"""Write a human-readable dump of the dataset: python -m sidm.preview [--data data/eval_raw.jsonl] [--out data/preview.txt]"""
import argparse
import json
from pathlib import Path


def render(row):
    tagged = " ".join("%s/%s" % (t, l[0].upper()) for t, l in zip(row["tokens"], row["token_labels"]))
    lines = ["#%d  %s" % (row["id"], row["query"]),
             "     category: %s" % (" > ".join(row["category_path"]) or "NONE"),
             "     filters:  %s" % (", ".join("%s=%s" % kv for kv in row["filters"].items()) or "-"),
             "     residual: %s" % (" ".join(row["residual_tokens"]) or "-"),
             "     words:    %s" % tagged]
    if row["flags"]:
        lines.append("     flags:    %s" % ", ".join(row["flags"]))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/eval_raw.jsonl")
    ap.add_argument("--out", default="data/preview.txt")
    args = ap.parse_args()
    rows = sorted((json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()), key=lambda r: r["id"])
    Path(args.out).write_text("\n".join(render(r) for r in rows))
    print("wrote %d rows to %s" % (len(rows), args.out))


if __name__ == "__main__":
    main()
