"""nimble's MLX ParallelScorer behind the same /v1/systemone contract as Ollama.

  mlx_backend/serve.sh --config ~/sidm-models/bespoke-nimble-9b-q8/nimble-model.json [--port 11500]

POST /v1/systemone  {model, state, questions, keep_alive?, mlx_options?} -> {model, answers, usage, metrics}
                    mlx_options = {mode, field_batch_size} overrides the server defaults (--mode, --field-batch-size)
GET  /health

Each Jev `choice` question becomes a nimble `enum` field: key -> field name, `instructions` -> description,
criteria keys -> choices (in order), non-null criteria texts -> choice_descriptions. All fields are scored
from one prefill in a single batched pass (ParallelScorer mode "parallel").
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock


class BadRequest(ValueError):
    pass


def to_schema(questions):
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise BadRequest("questions must be an object with 1-64 entries")
    schema = {}
    for name, q in questions.items():
        if q.get("type") != "choice":
            raise BadRequest("question %r: only type 'choice' is supported by this backend" % name)
        criteria = q.get("criteria")
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
            raise BadRequest("question %r: criteria must be an object with 2-255 options" % name)
        field = {"type": "enum", "choices": list(criteria),
                 "description": q.get("instructions") or name.replace("_", " ")}
        described = {k: v for k, v in criteria.items() if isinstance(v, str) and v.strip()}
        if described:
            field["choice_descriptions"] = described
        schema[name] = field
    return schema


def to_answers(result, questions):
    answers = {}
    for name in questions:
        scores = result["fields"][name]["scores"]
        best = max(scores, key=scores.get)
        answers[name] = {"type": "choice", "choice": best, "probabilities": scores,
                         "confidence": scores[best]}  # top-1 probability (Ollama's definition is unpublished)
    return answers


class Handler(BaseHTTPRequestHandler):
    scorer = None
    model_name = None
    mode, field_batch_size = "parallel", None
    lock = Lock()  # one scoring at a time: the GPU is the bottleneck anyway

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": self.model_name})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            return self._send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            state = req.get("state")
            if state in (None, "", {}, []):
                raise BadRequest("state must be nonempty")
            context = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            questions = req.get("questions")
            schema = to_schema(questions)
            opts = req.get("mlx_options") or {}  # optional knobs: mode (parallel|cached_serial|independent), field_batch_size
            mode = opts.get("mode", self.mode)
            batch = opts.get("field_batch_size", self.field_batch_size)
            started = time.perf_counter()
            with self.lock:
                result = self.scorer.score(context, schema, mode=mode, field_batch_size=batch)
            metrics = dict(result["metrics"], server_seconds=time.perf_counter() - started)
            n_prompt = metrics["prefix_tokens"] + sum(metrics["suffix_tokens"])
            self._send(200, {"model": req.get("model", self.model_name), "answers": to_answers(result, questions),
                             "usage": {"input_tokens": n_prompt, "output_tokens": len(questions)},
                             "metrics": metrics, "temperature": result["temperature"]})
        except BadRequest as e:
            self._send(400, {"error": str(e)})
        except ValueError as e:  # nimble rejects e.g. prompts over max_input_tokens
            self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="nimble-model.json written by convert_model.py")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--mode", default="parallel", choices=["parallel", "cached_serial", "independent"])
    ap.add_argument("--field-batch-size", type=int, default=None, help="fields per batched pass (default: all)")
    args = ap.parse_args()
    config = json.loads(Path(args.config).expanduser().read_text())
    sys.path.insert(0, config["nimble_repo"])
    from nimble.scoring.parallel_scorer import ParallelScorer
    Handler.scorer = ParallelScorer(model_path=config["model_path"], max_input_tokens=config["max_input_tokens"],
                                    model_id=config["model_id"], revision=config["revision"])
    Handler.model_name = "%s@%s (q%s)" % (config["model_id"], config["revision"][:8], config["q_bits"])
    Handler.mode, Handler.field_batch_size = args.mode, args.field_batch_size
    print("serving %s on http://%s:%d (temperature %.3f)" % (Handler.model_name, args.host, args.port,
                                                              Handler.scorer.temperature), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
