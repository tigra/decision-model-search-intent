# Results: dev split

- **Queries:** 100, evaluated by every run. The dev split has 100 rows in `data/eval_raw.jsonl`.
- **Decoding:** tuned per backend on the 100 dev rows (ids 0–99). Filters are masked by the predicted category's applicable attributes.
- **Runs:**
  - `embedded@ollama:nimble`: Ollama 0.35 `/v1/systemone` (llama.cpp, GGUF); questions scored one after another, split into several requests when the prompt exceeds the model context; model `nimble` (served as `not recorded`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.95; word-role `category` weight ×8
  - `router@ollama:nimble`: Ollama 0.35 `/v1/systemone` (llama.cpp, GGUF); questions scored one after another, split into several requests when the prompt exceeds the model context; model `nimble` (served as `not recorded`); category scheme top-level `cat_L1` + conditional group questions, 'no category' threshold 0.50; filter p ≥ 0.95; word-role `category` weight ×8
  - `embedded@mlx:nimble`: nimble's MLX `ParallelScorer` (8-bit, `lm_head` in bf16) via `mlx_backend/server.py`; one prefill, then fields batched or one at a time (mode recorded per query); model `nimble` (served as `not recorded`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.95; word-role `category` weight ×8
  - `embedded@ollama:tev1`: Ollama 0.35 `/v1/systemone` (llama.cpp, GGUF); questions scored one after another, split into several requests when the prompt exceeds the model context; model `tev1` (served as `tev1`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.70; word-role `category` weight ×16
  - `embedded@ollama:tev1:0.8b`: Ollama 0.35 `/v1/systemone` (llama.cpp, GGUF); questions scored one after another, split into several requests when the prompt exceeds the model context; model `tev1:0.8b` (served as `tev1:0.8b`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.70; word-role `category` weight ×8
  - `embedded@ollaya:jeb:4b`: Ollaya 0.9 local server (TypeSafe-compatible `/v1/systemone`), model as named in the run; model `jeb:4b` (served as `jeb:4b`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.70; word-role `category` weight ×16
  - `embedded@ollaya:winnow:e4b`: Ollaya 0.9 local server (TypeSafe-compatible `/v1/systemone`), model as named in the run; model `winnow:e4b` (served as `winnow:e4b`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.50; word-role `category` weight ×8
  - `embedded@ollaya:decider:2b`: Ollaya 0.9 local server (TypeSafe-compatible `/v1/systemone`), model as named in the run; model `decider:2b` (served as `decider:2b`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.70; word-role `category` weight ×8
  - `embedded@ollaya:decision:eos`: Ollaya 0.9 local server (TypeSafe-compatible `/v1/systemone`), model as named in the run; model `decision:eos` (served as `decision:eos`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.80; word-role `category` weight ×8
  - `embedded@ollaya:kev:0.8b`: Ollaya 0.9 local server (TypeSafe-compatible `/v1/systemone`), model as named in the run; model `kev:0.8b` (served as `kev:0.8b`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.80; filter p ≥ 0.00; word-role `category` weight ×8
  - `embedded@ollaya:laya:en`: Ollaya 0.9 local server (TypeSafe-compatible `/v1/systemone`), model as named in the run; model `laya:en` (served as `laya:en`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.98; word-role `category` weight ×8
  - `embedded@ollaya:laya:typed-decisions`: Ollaya 0.9 local server (TypeSafe-compatible `/v1/systemone`), model as named in the run; model `laya:typed-decisions` (served as `laya:typed-decisions`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.95; word-role `category` weight ×8
  - `embedded@ollaya:von`: Ollaya 0.9 local server (TypeSafe-compatible `/v1/systemone`), model as named in the run; model `von` (served as `von:latest`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.90; filter p ≥ 0.50; word-role `category` weight ×8
  - `embedded@ollaya:nli:modernbert-large`: Ollaya 0.9 local server (TypeSafe-compatible `/v1/systemone`), model as named in the run; model `nli:modernbert-large` (served as `nli:modernbert-large`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.50; word-role `category` weight ×8
  - `embedded@decider:strands-decider-2b`: AWS Strands Labs' Strands Decider 2B (`StrandsAgents/strands-decider-2B-hobson-v19`, MLX) via its own `/v1/systemone` server; the state is encoded once and only each question's suffix is added, batched; model `strands-decider-2b` (served as `strands-decider-2b`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.50; word-role `category` weight ×1
  - `embedded@jev:jev-latest`: TypeSafe's hosted Jev API (`https://api.typesafe.ai`); latency includes the network round trip; model `jev-latest` (served as `jev-1.13.0`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.95; word-role `category` weight ×8
- **Cells:** value (correct / counted) [95% Wilson interval]. F1 has no interval.

| Metric | Measure | Counted over | `embedded@ollama:nimble` | `router@ollama:nimble` | `embedded@mlx:nimble` | `embedded@ollama:tev1` | `embedded@ollama:tev1:0.8b` | `embedded@ollaya:jeb:4b` | `embedded@ollaya:winnow:e4b` | `embedded@ollaya:decider:2b` | `embedded@ollaya:decision:eos` | `embedded@ollaya:kev:0.8b` | `embedded@ollaya:laya:en` | `embedded@ollaya:laya:typed-decisions` | `embedded@ollaya:von` | `embedded@ollaya:nli:modernbert-large` | `embedded@decider:strands-decider-2b` | `embedded@jev:jev-latest` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Category, exact node | accuracy | all queries; the predicted node must equal the gold node (same depth; 'no category' is a value) | 0.910 (91/100) [0.84–0.95] | 0.850 (85/100) [0.77–0.91] | 0.920 (92/100) [0.85–0.96] | 0.910 (91/100) [0.84–0.95] | 0.750 (75/100) [0.66–0.82] | 0.900 (90/100) [0.83–0.94] | 0.960 (96/100) [0.90–0.98] | 0.820 (82/100) [0.73–0.88] | 0.760 (76/100) [0.67–0.83] | 0.760 (76/100) [0.67–0.83] | 0.760 (76/100) [0.67–0.83] | 0.730 (73/100) [0.64–0.81] | 0.550 (55/100) [0.45–0.64] | 0.540 (54/100) [0.44–0.63] | 0.760 (76/100) [0.67–0.83] | 0.970 (97/100) [0.92–0.99] |
| Category correct at L1 | accuracy | queries with a gold category; predicted path has the gold L1 node | 0.979 (93/95) [0.93–0.99] | 0.863 (82/95) [0.78–0.92] | 0.989 (94/95) [0.94–1.00] | 0.958 (91/95) [0.90–0.98] | 0.832 (79/95) [0.74–0.89] | 0.968 (92/95) [0.91–0.99] | 1.000 (95/95) [0.96–1.00] | 0.947 (90/95) [0.88–0.98] | 0.821 (78/95) [0.73–0.89] | 0.863 (82/95) [0.78–0.92] | 0.842 (80/95) [0.76–0.90] | 0.800 (76/95) [0.71–0.87] | 0.600 (57/95) [0.50–0.69] | 0.579 (55/95) [0.48–0.67] | 0.832 (79/95) [0.74–0.89] | 1.000 (95/95) [0.96–1.00] |
| Category correct at L2 | accuracy | queries whose gold category is at depth ≥ 2; predicted path has the gold L2 node | 0.977 (86/88) [0.92–0.99] | 0.886 (78/88) [0.80–0.94] | 1.000 (88/88) [0.96–1.00] | 0.989 (87/88) [0.94–1.00] | 0.818 (72/88) [0.72–0.88] | 0.966 (85/88) [0.90–0.99] | 1.000 (88/88) [0.96–1.00] | 0.932 (82/88) [0.86–0.97] | 0.818 (72/88) [0.72–0.88] | 0.773 (68/88) [0.67–0.85] | 0.875 (77/88) [0.79–0.93] | 0.841 (74/88) [0.75–0.90] | 0.591 (52/88) [0.49–0.69] | 0.602 (53/88) [0.50–0.70] | 0.875 (77/88) [0.79–0.93] | 1.000 (88/88) [0.96–1.00] |
| Category correct at L3 | accuracy | queries whose gold category is at depth 3; predicted node is the gold L3 node | 0.982 (54/55) [0.90–1.00] | 0.873 (48/55) [0.76–0.94] | 1.000 (55/55) [0.93–1.00] | 1.000 (55/55) [0.93–1.00] | 0.836 (46/55) [0.72–0.91] | 0.945 (52/55) [0.85–0.98] | 0.982 (54/55) [0.90–1.00] | 0.909 (50/55) [0.80–0.96] | 0.836 (46/55) [0.72–0.91] | 0.855 (47/55) [0.74–0.92] | 0.873 (48/55) [0.76–0.94] | 0.855 (47/55) [0.74–0.92] | 0.709 (39/55) [0.58–0.81] | 0.673 (37/55) [0.54–0.78] | 0.873 (48/55) [0.76–0.94] | 1.000 (55/55) [0.93–1.00] |
| 'No category' precision | precision | queries predicted as 'no category' | 1.000 (2/2) [0.34–1.00] | 0.333 (4/12) [0.14–0.61] | 0.667 (2/3) [0.21–0.94] | 1.000 (1/1) [0.21–1.00] | 1.000 (1/1) [0.21–1.00] | 1.000 (1/1) [0.21–1.00] | 1.000 (4/4) [0.51–1.00] | n/a (0/0) | 0.000 (0/1) [0.00–0.79] | 0.667 (2/3) [0.21–0.94] | n/a (0/0) | n/a (0/0) | 0.059 (1/17) [0.01–0.27] | 0.053 (1/19) [0.01–0.25] | n/a (0/0) | 1.000 (4/4) [0.51–1.00] |
| 'No category' recall | recall | queries whose gold is 'no category' | 0.400 (2/5) [0.12–0.77] | 0.800 (4/5) [0.38–0.96] | 0.400 (2/5) [0.12–0.77] | 0.200 (1/5) [0.04–0.62] | 0.200 (1/5) [0.04–0.62] | 0.200 (1/5) [0.04–0.62] | 0.800 (4/5) [0.38–0.96] | 0.000 (0/5) [-0.00–0.43] | 0.000 (0/5) [-0.00–0.43] | 0.400 (2/5) [0.12–0.77] | 0.000 (0/5) [-0.00–0.43] | 0.000 (0/5) [-0.00–0.43] | 0.200 (1/5) [0.04–0.62] | 0.200 (1/5) [0.04–0.62] | 0.000 (0/5) [-0.00–0.43] | 0.800 (4/5) [0.38–0.96] |
| Filters precision | precision (micro) | predicted attribute=value pairs, pooled over all queries | 0.922 (142/154) [0.87–0.95] | 0.902 (138/153) [0.84–0.94] | 0.922 (142/154) [0.87–0.95] | 0.824 (140/170) [0.76–0.87] | 0.693 (106/153) [0.62–0.76] | 0.865 (141/163) [0.80–0.91] | 0.810 (136/168) [0.74–0.86] | 0.801 (129/161) [0.73–0.86] | 0.910 (71/78) [0.83–0.96] | 0.525 (107/204) [0.46–0.59] | 0.697 (53/76) [0.59–0.79] | 0.706 (12/17) [0.47–0.87] | 0.311 (28/90) [0.22–0.41] | 0.250 (49/196) [0.19–0.32] | 0.787 (126/160) [0.72–0.84] | 0.964 (132/137) [0.92–0.98] |
| Filters recall | recall (micro) | gold attribute=value pairs, pooled over all queries | 0.940 (142/151) [0.89–0.97] | 0.914 (138/151) [0.86–0.95] | 0.940 (142/151) [0.89–0.97] | 0.927 (140/151) [0.87–0.96] | 0.702 (106/151) [0.62–0.77] | 0.934 (141/151) [0.88–0.96] | 0.901 (136/151) [0.84–0.94] | 0.854 (129/151) [0.79–0.90] | 0.470 (71/151) [0.39–0.55] | 0.709 (107/151) [0.63–0.78] | 0.351 (53/151) [0.28–0.43] | 0.079 (12/151) [0.05–0.13] | 0.185 (28/151) [0.13–0.25] | 0.325 (49/151) [0.25–0.40] | 0.834 (126/151) [0.77–0.89] | 0.874 (132/151) [0.81–0.92] |
| Filters F1 | F1 (micro) | harmonic mean of the two rows above | 0.931 | 0.908 | 0.931 | 0.872 | 0.697 | 0.898 | 0.853 | 0.827 | 0.620 | 0.603 | 0.467 | 0.143 | 0.232 | 0.282 | 0.810 | 0.917 |
| Filter set exact match | accuracy | all queries; predicted filter set equals the gold set (incl. both empty) | 0.800 (80/100) [0.71–0.87] | 0.770 (77/100) [0.68–0.84] | 0.820 (82/100) [0.73–0.88] | 0.690 (69/100) [0.59–0.77] | 0.340 (34/100) [0.25–0.44] | 0.770 (77/100) [0.68–0.84] | 0.660 (66/100) [0.56–0.75] | 0.610 (61/100) [0.51–0.70] | 0.320 (32/100) [0.24–0.42] | 0.290 (29/100) [0.21–0.39] | 0.240 (24/100) [0.17–0.33] | 0.160 (16/100) [0.10–0.24] | 0.140 (14/100) [0.09–0.22] | 0.070 (7/100) [0.03–0.14] | 0.550 (55/100) [0.45–0.64] | 0.790 (79/100) [0.70–0.86] |
| Word-role accuracy | accuracy | all words of all queries; role category / filter / residual | 0.841 (481/572) [0.81–0.87] | 0.836 (478/572) [0.80–0.86] | 0.871 (498/572) [0.84–0.90] | 0.710 (406/572) [0.67–0.75] | 0.327 (187/572) [0.29–0.37] | 0.760 (435/572) [0.72–0.79] | 0.696 (398/572) [0.66–0.73] | 0.558 (319/572) [0.52–0.60] | 0.329 (188/572) [0.29–0.37] | 0.329 (188/572) [0.29–0.37] | 0.329 (188/572) [0.29–0.37] | 0.329 (188/572) [0.29–0.37] | 0.329 (188/572) [0.29–0.37] | 0.329 (188/572) [0.29–0.37] | 0.692 (396/572) [0.65–0.73] | 0.878 (502/572) [0.85–0.90] |
| Residual words precision | precision | words predicted 'residual' | 0.677 (105/155) [0.60–0.75] | 0.648 (105/162) [0.57–0.72] | 0.729 (102/140) [0.65–0.80] | 0.792 (103/130) [0.71–0.85] | 0.000 (0/1) [0.00–0.79] | 0.715 (98/137) [0.63–0.78] | 1.000 (8/8) [0.68–1.00] | 1.000 (14/14) [0.78–1.00] | n/a (0/0) | n/a (0/0) | n/a (0/0) | n/a (0/0) | n/a (0/0) | n/a (0/0) | 0.778 (28/36) [0.62–0.88] | 0.944 (85/90) [0.88–0.98] |
| Residual words recall | recall | words whose gold role is 'residual' | 0.808 (105/130) [0.73–0.87] | 0.808 (105/130) [0.73–0.87] | 0.785 (102/130) [0.71–0.85] | 0.792 (103/130) [0.71–0.85] | 0.000 (0/130) [0.00–0.03] | 0.754 (98/130) [0.67–0.82] | 0.062 (8/130) [0.03–0.12] | 0.108 (14/130) [0.07–0.17] | 0.000 (0/130) [0.00–0.03] | 0.000 (0/130) [0.00–0.03] | 0.000 (0/130) [0.00–0.03] | 0.000 (0/130) [0.00–0.03] | 0.000 (0/130) [0.00–0.03] | 0.000 (0/130) [0.00–0.03] | 0.215 (28/130) [0.15–0.29] | 0.654 (85/130) [0.57–0.73] |
| Residual words F1 | F1 | harmonic mean of the two rows above | 0.737 | 0.719 | 0.756 | 0.792 | 0.000 | 0.734 | 0.116 | 0.194 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.337 | 0.773 |
| **Whole query exactly right** | accuracy | all queries; category exact, filter set exact, and the set of residual word positions exact | 0.440 (44/100) [0.35–0.54] | 0.350 (35/100) [0.26–0.45] | 0.430 (43/100) [0.34–0.53] | 0.420 (42/100) [0.33–0.52] | 0.100 (10/100) [0.06–0.17] | 0.370 (37/100) [0.28–0.47] | 0.290 (29/100) [0.21–0.39] | 0.240 (24/100) [0.17–0.33] | 0.110 (11/100) [0.06–0.19] | 0.110 (11/100) [0.06–0.19] | 0.100 (10/100) [0.06–0.17] | 0.060 (6/100) [0.03–0.12] | 0.060 (6/100) [0.03–0.12] | 0.040 (4/100) [0.02–0.10] | 0.220 (22/100) [0.15–0.31] | 0.570 (57/100) [0.47–0.66] |
| **Weighted query score** | mean (0–1) | all queries; per query the weighted mean of category credit (0.45; partial for the right branch), filters F0.5 (0.30), residual words F2 (0.17) and the other word roles (0.08), over the parts the query involves (`sidm/score.py`); 95% interval of the mean | 0.883 [0.854–0.912] | 0.834 [0.796–0.872] | 0.889 [0.861–0.917] | 0.841 [0.800–0.882] | 0.631 [0.579–0.683] | 0.848 [0.813–0.882] | 0.830 [0.800–0.860] | 0.762 [0.719–0.805] | 0.618 [0.565–0.670] | 0.618 [0.569–0.667] | 0.576 [0.518–0.633] | 0.468 [0.416–0.521] | 0.403 [0.343–0.463] | 0.395 [0.337–0.454] | 0.726 [0.674–0.778] | 0.916 [0.889–0.943] |
| Latency p50 / p95 | seconds | all queries; one request each, sequential, warm model | 16.1 s / 17.4 s | 15.4 s / 16.9 s | 17.4 s / 19.3 s | 10.2 s / 11.5 s | 2.3 s / 2.7 s | 10.7 s / 12.6 s | 6.4 s / 6.8 s | 75.1 s / 121.4 s | 35.1 s / 42.4 s | 31.4 s / 41.2 s | 1.1 s / 1.3 s | 14.6 s / 18.0 s | 13.0 s / 15.3 s | 5.7 s / 6.7 s | 3.9 s / 4.5 s | 0.3 s / 0.4 s |
| Server prefill / field evaluation | mean seconds | mlx backend only (nimble's own timings) | n/a | n/a | 13.14 s / 3.23 s | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Questions / input tokens | mean per query | server-reported input tokens (Ollama: every question charged the full prompt; mlx: tokens actually processed) | 23.7 / 128k | 24.7 / 134k | 23.7 / 7k | 23.7 / 25k | 23.7 / 25k | 23.7 / 8k | 23.7 / 9k | 23.7 / 8k | 23.7 / 10k | 23.7 / 7k | 23.7 / 7k | 23.7 / 7k | 23.7 / 6k | 23.7 / 44k | 23.7 / 4k | 23.7 / 5k |

## Paired comparison against `embedded@ollama:nimble` (exact McNemar test, same queries)

| Run | Metric | correct only with `embedded@ollama:nimble` | correct only with the other | p-value |
|---|---|---|---|---|
| `router@ollama:nimble` | Category, exact node | 9 | 3 | 0.1460 |
| `router@ollama:nimble` | Whole query exactly right | 10 | 1 | 0.0117 |
| `embedded@mlx:nimble` | Category, exact node | 1 | 2 | 1.0000 |
| `embedded@mlx:nimble` | Whole query exactly right | 6 | 5 | 1.0000 |
| `embedded@ollama:tev1` | Category, exact node | 5 | 5 | 1.0000 |
| `embedded@ollama:tev1` | Whole query exactly right | 15 | 13 | 0.8506 |
| `embedded@ollama:tev1:0.8b` | Category, exact node | 20 | 4 | 0.0015 |
| `embedded@ollama:tev1:0.8b` | Whole query exactly right | 35 | 1 | 0.0000 |
| `embedded@ollaya:jeb:4b` | Category, exact node | 5 | 4 | 1.0000 |
| `embedded@ollaya:jeb:4b` | Whole query exactly right | 11 | 4 | 0.1185 |
| `embedded@ollaya:winnow:e4b` | Category, exact node | 1 | 6 | 0.1250 |
| `embedded@ollaya:winnow:e4b` | Whole query exactly right | 26 | 11 | 0.0201 |
| `embedded@ollaya:decider:2b` | Category, exact node | 11 | 2 | 0.0225 |
| `embedded@ollaya:decider:2b` | Whole query exactly right | 28 | 8 | 0.0012 |
| `embedded@ollaya:decision:eos` | Category, exact node | 18 | 3 | 0.0015 |
| `embedded@ollaya:decision:eos` | Whole query exactly right | 37 | 4 | 0.0000 |
| `embedded@ollaya:kev:0.8b` | Category, exact node | 20 | 5 | 0.0041 |
| `embedded@ollaya:kev:0.8b` | Whole query exactly right | 37 | 4 | 0.0000 |
| `embedded@ollaya:laya:en` | Category, exact node | 17 | 2 | 0.0007 |
| `embedded@ollaya:laya:en` | Whole query exactly right | 36 | 2 | 0.0000 |
| `embedded@ollaya:laya:typed-decisions` | Category, exact node | 20 | 2 | 0.0001 |
| `embedded@ollaya:laya:typed-decisions` | Whole query exactly right | 38 | 0 | 0.0000 |
| `embedded@ollaya:von` | Category, exact node | 39 | 3 | 0.0000 |
| `embedded@ollaya:von` | Whole query exactly right | 39 | 1 | 0.0000 |
| `embedded@ollaya:nli:modernbert-large` | Category, exact node | 39 | 2 | 0.0000 |
| `embedded@ollaya:nli:modernbert-large` | Whole query exactly right | 42 | 2 | 0.0000 |
| `embedded@decider:strands-decider-2b` | Category, exact node | 17 | 2 | 0.0007 |
| `embedded@decider:strands-decider-2b` | Whole query exactly right | 33 | 11 | 0.0013 |
| `embedded@jev:jev-latest` | Category, exact node | 0 | 6 | 0.0312 |
| `embedded@jev:jev-latest` | Whole query exactly right | 6 | 19 | 0.0146 |

## All pairs: Category, exact node (exact McNemar test)

Cell (row, column) = queries correct only with the row run : only with the column run, and the p-value. **Bold** = significant at p < 0.05; the better run of the pair is the one with the larger count.

| | `router@ollama:nimble` | `embedded@mlx:nimble` | `embedded@ollama:tev1` | `embedded@ollama:tev1:0.8b` | `embedded@ollaya:jeb:4b` | `embedded@ollaya:winnow:e4b` | `embedded@ollaya:decider:2b` | `embedded@ollaya:decision:eos` | `embedded@ollaya:kev:0.8b` | `embedded@ollaya:laya:en` | `embedded@ollaya:laya:typed-decisions` | `embedded@ollaya:von` | `embedded@ollaya:nli:modernbert-large` | `embedded@decider:strands-decider-2b` | `embedded@jev:jev-latest` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `embedded@ollama:nimble` | 9:3 p=0.1460 | 1:2 p=1.0000 | 5:5 p=1.0000 | **20:4 p=0.0015** | 5:4 p=1.0000 | 1:6 p=0.1250 | **11:2 p=0.0225** | **18:3 p=0.0015** | **20:5 p=0.0041** | **17:2 p=0.0007** | **20:2 p=0.0001** | **39:3 p=<0.0001** | **39:2 p=<0.0001** | **17:2 p=0.0007** | **0:6 p=0.0312** |
| `router@ollama:nimble` |  | 3:10 p=0.0923 | 7:13 p=0.2632 | 21:11 p=0.1102 | 5:10 p=0.3018 | **1:12 p=0.0034** | 11:8 p=0.6476 | 19:10 p=0.1360 | 19:10 p=0.1360 | 18:9 p=0.1221 | **19:7 p=0.0290** | **38:8 p=<0.0001** | **38:7 p=<0.0001** | 16:7 p=0.0931 | **0:12 p=0.0005** |
| `embedded@mlx:nimble` |  |  | 4:3 p=1.0000 | **21:4 p=0.0009** | 5:3 p=0.7266 | 1:5 p=0.2188 | **10:0 p=0.0020** | **18:2 p=0.0004** | **21:5 p=0.0025** | **16:0 p=<0.0001** | **19:0 p=<0.0001** | **39:2 p=<0.0001** | **38:0 p=<0.0001** | **16:0 p=<0.0001** | 0:5 p=0.0625 |
| `embedded@ollama:tev1` |  |  |  | **20:4 p=0.0015** | 4:3 p=1.0000 | 2:7 p=0.1797 | **11:2 p=0.0225** | **17:2 p=0.0007** | **21:6 p=0.0059** | **16:1 p=0.0003** | **19:1 p=<0.0001** | **37:1 p=<0.0001** | **38:1 p=<0.0001** | **16:1 p=0.0003** | 1:7 p=0.0703 |
| `embedded@ollama:tev1:0.8b` |  |  |  |  | **3:18 p=0.0015** | **2:23 p=<0.0001** | 8:15 p=0.2100 | 9:10 p=1.0000 | 12:13 p=1.0000 | 14:15 p=1.0000 | 16:14 p=0.8555 | **26:6 p=0.0005** | **31:10 p=0.0015** | 13:14 p=1.0000 | **1:23 p=<0.0001** |
| `embedded@ollaya:jeb:4b` |  |  |  |  |  | 1:7 p=0.0703 | 12:4 p=0.0768 | **17:3 p=0.0026** | **20:6 p=0.0094** | **16:2 p=0.0013** | **19:2 p=0.0002** | **37:2 p=<0.0001** | **38:2 p=<0.0001** | **16:2 p=0.0013** | **0:7 p=0.0156** |
| `embedded@ollaya:winnow:e4b` |  |  |  |  |  |  | **15:1 p=0.0005** | **21:1 p=<0.0001** | **22:2 p=<0.0001** | **20:0 p=<0.0001** | **24:1 p=<0.0001** | **42:1 p=<0.0001** | **43:1 p=<0.0001** | **21:1 p=<0.0001** | 0:1 p=1.0000 |
| `embedded@ollaya:decider:2b` |  |  |  |  |  |  |  | 11:5 p=0.2101 | 16:10 p=0.3269 | 10:4 p=0.1796 | **13:4 p=0.0490** | **33:6 p=<0.0001** | **31:3 p=<0.0001** | 10:4 p=0.1796 | **0:15 p=<0.0001** |
| `embedded@ollaya:decision:eos` |  |  |  |  |  |  |  |  | 10:10 p=1.0000 | 13:13 p=1.0000 | 14:11 p=0.6900 | **29:8 p=0.0008** | **32:10 p=0.0009** | 9:9 p=1.0000 | **0:21 p=<0.0001** |
| `embedded@ollaya:kev:0.8b` |  |  |  |  |  |  |  |  |  | 17:17 p=1.0000 | 17:14 p=0.7201 | **31:10 p=0.0015** | **31:9 p=0.0007** | 14:14 p=1.0000 | **1:22 p=<0.0001** |
| `embedded@ollaya:laya:en` |  |  |  |  |  |  |  |  |  |  | 10:7 p=0.6291 | **29:8 p=0.0008** | **28:6 p=0.0002** | 9:9 p=1.0000 | **0:21 p=<0.0001** |
| `embedded@ollaya:laya:typed-decisions` |  |  |  |  |  |  |  |  |  |  |  | **32:14 p=0.0114** | **30:11 p=0.0043** | 5:8 p=0.5811 | **0:24 p=<0.0001** |
| `embedded@ollaya:von` |  |  |  |  |  |  |  |  |  |  |  |  | 17:16 p=1.0000 | **9:30 p=0.0011** | **0:42 p=<0.0001** |
| `embedded@ollaya:nli:modernbert-large` |  |  |  |  |  |  |  |  |  |  |  |  |  | **8:30 p=0.0005** | **0:43 p=<0.0001** |
| `embedded@decider:strands-decider-2b` |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **0:21 p=<0.0001** |

## All pairs: Whole query exactly right (exact McNemar test)

Cell (row, column) = queries correct only with the row run : only with the column run, and the p-value. **Bold** = significant at p < 0.05; the better run of the pair is the one with the larger count.

| | `router@ollama:nimble` | `embedded@mlx:nimble` | `embedded@ollama:tev1` | `embedded@ollama:tev1:0.8b` | `embedded@ollaya:jeb:4b` | `embedded@ollaya:winnow:e4b` | `embedded@ollaya:decider:2b` | `embedded@ollaya:decision:eos` | `embedded@ollaya:kev:0.8b` | `embedded@ollaya:laya:en` | `embedded@ollaya:laya:typed-decisions` | `embedded@ollaya:von` | `embedded@ollaya:nli:modernbert-large` | `embedded@decider:strands-decider-2b` | `embedded@jev:jev-latest` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `embedded@ollama:nimble` | **10:1 p=0.0117** | 6:5 p=1.0000 | 15:13 p=0.8506 | **35:1 p=<0.0001** | 11:4 p=0.1185 | **26:11 p=0.0201** | **28:8 p=0.0012** | **37:4 p=<0.0001** | **37:4 p=<0.0001** | **36:2 p=<0.0001** | **38:0 p=<0.0001** | **39:1 p=<0.0001** | **42:2 p=<0.0001** | **33:11 p=0.0013** | **6:19 p=0.0146** |
| `router@ollama:nimble` |  | 5:13 p=0.0963 | 13:20 p=0.2962 | **29:4 p=<0.0001** | 8:10 p=0.8145 | 23:17 p=0.4296 | 24:13 p=0.0989 | **31:7 p=0.0001** | **30:6 p=<0.0001** | **29:4 p=<0.0001** | **32:3 p=<0.0001** | **31:2 p=<0.0001** | **33:2 p=<0.0001** | 27:14 p=0.0596 | **6:28 p=0.0002** |
| `embedded@mlx:nimble` |  |  | 14:13 p=1.0000 | **34:1 p=<0.0001** | 11:5 p=0.2101 | **26:12 p=0.0336** | **28:9 p=0.0026** | **37:5 p=<0.0001** | **37:5 p=<0.0001** | **36:3 p=<0.0001** | **38:1 p=<0.0001** | **39:2 p=<0.0001** | **41:2 p=<0.0001** | **32:11 p=0.0019** | **7:21 p=0.0125** |
| `embedded@ollama:tev1` |  |  |  | **34:2 p=<0.0001** | 12:7 p=0.3593 | **21:8 p=0.0241** | **23:5 p=0.0009** | **34:3 p=<0.0001** | **34:3 p=<0.0001** | **34:2 p=<0.0001** | **37:1 p=<0.0001** | **37:1 p=<0.0001** | **38:0 p=<0.0001** | **27:7 p=0.0008** | **6:21 p=0.0059** |
| `embedded@ollama:tev1:0.8b` |  |  |  |  | **2:29 p=<0.0001** | **3:22 p=0.0002** | **4:18 p=0.0043** | 3:4 p=1.0000 | 6:7 p=1.0000 | 6:6 p=1.0000 | 8:4 p=0.3877 | 9:5 p=0.4240 | 8:2 p=0.1094 | **5:17 p=0.0169** | **0:47 p=<0.0001** |
| `embedded@ollaya:jeb:4b` |  |  |  |  |  | 20:12 p=0.2153 | **21:8 p=0.0241** | **31:5 p=<0.0001** | **30:4 p=<0.0001** | **30:3 p=<0.0001** | **32:1 p=<0.0001** | **33:2 p=<0.0001** | **34:1 p=<0.0001** | **27:12 p=0.0237** | **3:23 p=<0.0001** |
| `embedded@ollaya:winnow:e4b` |  |  |  |  |  |  | 9:4 p=0.2668 | **20:2 p=0.0001** | **19:1 p=<0.0001** | **20:1 p=<0.0001** | **23:0 p=<0.0001** | **23:0 p=<0.0001** | **25:0 p=<0.0001** | 13:6 p=0.1671 | **2:30 p=<0.0001** |
| `embedded@ollaya:decider:2b` |  |  |  |  |  |  |  | **17:4 p=0.0072** | **14:1 p=0.0010** | **16:2 p=0.0013** | **19:1 p=<0.0001** | **19:1 p=<0.0001** | **20:0 p=<0.0001** | 11:9 p=0.8238 | **2:35 p=<0.0001** |
| `embedded@ollaya:decision:eos` |  |  |  |  |  |  |  |  | 5:5 p=1.0000 | 5:4 p=1.0000 | 8:3 p=0.2266 | 9:4 p=0.2668 | 9:2 p=0.0654 | **4:15 p=0.0192** | **2:48 p=<0.0001** |
| `embedded@ollaya:kev:0.8b` |  |  |  |  |  |  |  |  |  | 7:6 p=1.0000 | 9:4 p=0.2668 | 8:3 p=0.2266 | **7:0 p=0.0156** | **3:14 p=0.0127** | **1:47 p=<0.0001** |
| `embedded@ollaya:laya:en` |  |  |  |  |  |  |  |  |  |  | 6:2 p=0.2891 | 6:2 p=0.2891 | 8:2 p=0.1094 | **3:15 p=0.0075** | **2:49 p=<0.0001** |
| `embedded@ollaya:laya:typed-decisions` |  |  |  |  |  |  |  |  |  |  |  | 2:2 p=1.0000 | 6:4 p=0.7539 | **2:18 p=0.0004** | **0:51 p=<0.0001** |
| `embedded@ollaya:von` |  |  |  |  |  |  |  |  |  |  |  |  | 4:2 p=0.6875 | **3:19 p=0.0009** | **0:51 p=<0.0001** |
| `embedded@ollaya:nli:modernbert-large` |  |  |  |  |  |  |  |  |  |  |  |  |  | **1:19 p=<0.0001** | **0:53 p=<0.0001** |
| `embedded@decider:strands-decider-2b` |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **5:40 p=<0.0001** |

## All pairs: weighted query score (paired sign-flip test)

Cell (row, column) = the row run's mean score minus the column run's, and the p-value. **Bold** = significant at p < 0.05.

| | `router@ollama:nimble` | `embedded@mlx:nimble` | `embedded@ollama:tev1` | `embedded@ollama:tev1:0.8b` | `embedded@ollaya:jeb:4b` | `embedded@ollaya:winnow:e4b` | `embedded@ollaya:decider:2b` | `embedded@ollaya:decision:eos` | `embedded@ollaya:kev:0.8b` | `embedded@ollaya:laya:en` | `embedded@ollaya:laya:typed-decisions` | `embedded@ollaya:von` | `embedded@ollaya:nli:modernbert-large` | `embedded@decider:strands-decider-2b` | `embedded@jev:jev-latest` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `embedded@ollama:nimble` | **+0.049 p=0.0105** | -0.006 p=0.5641 | **+0.042 p=0.0420** | **+0.252 p=<0.0001** | **+0.035 p=0.0222** | **+0.053 p=0.0052** | **+0.121 p=<0.0001** | **+0.265 p=<0.0001** | **+0.265 p=<0.0001** | **+0.307 p=<0.0001** | **+0.414 p=<0.0001** | **+0.480 p=<0.0001** | **+0.488 p=<0.0001** | **+0.157 p=<0.0001** | **-0.033 p=0.0337** |
| `router@ollama:nimble` |  | **-0.055 p=0.0068** | -0.007 p=0.8077 | **+0.203 p=<0.0001** | -0.013 p=0.5400 | +0.004 p=0.8576 | **+0.073 p=0.0071** | **+0.217 p=<0.0001** | **+0.217 p=<0.0001** | **+0.259 p=<0.0001** | **+0.366 p=<0.0001** | **+0.431 p=<0.0001** | **+0.439 p=<0.0001** | **+0.108 p=0.0006** | **-0.082 p=0.0004** |
| `embedded@mlx:nimble` |  |  | **+0.048 p=0.0100** | **+0.258 p=<0.0001** | **+0.041 p=0.0066** | **+0.059 p=0.0010** | **+0.128 p=<0.0001** | **+0.271 p=<0.0001** | **+0.271 p=<0.0001** | **+0.314 p=<0.0001** | **+0.421 p=<0.0001** | **+0.486 p=<0.0001** | **+0.494 p=<0.0001** | **+0.163 p=<0.0001** | -0.027 p=0.0677 |
| `embedded@ollama:tev1` |  |  |  | **+0.210 p=<0.0001** | -0.007 p=0.6971 | +0.011 p=0.6273 | **+0.080 p=0.0007** | **+0.223 p=<0.0001** | **+0.223 p=<0.0001** | **+0.266 p=<0.0001** | **+0.373 p=<0.0001** | **+0.438 p=<0.0001** | **+0.446 p=<0.0001** | **+0.115 p=<0.0001** | **-0.075 p=0.0008** |
| `embedded@ollama:tev1:0.8b` |  |  |  |  | **-0.217 p=<0.0001** | **-0.199 p=<0.0001** | **-0.131 p=<0.0001** | +0.013 p=0.6389 | +0.013 p=0.6314 | +0.055 p=0.1029 | **+0.162 p=<0.0001** | **+0.228 p=<0.0001** | **+0.235 p=<0.0001** | **-0.095 p=0.0040** | **-0.285 p=<0.0001** |
| `embedded@ollaya:jeb:4b` |  |  |  |  |  | +0.018 p=0.3898 | **+0.086 p=0.0005** | **+0.230 p=<0.0001** | **+0.230 p=<0.0001** | **+0.272 p=<0.0001** | **+0.379 p=<0.0001** | **+0.444 p=<0.0001** | **+0.452 p=<0.0001** | **+0.122 p=<0.0001** | **-0.068 p=<0.0001** |
| `embedded@ollaya:winnow:e4b` |  |  |  |  |  |  | **+0.068 p=0.0011** | **+0.212 p=<0.0001** | **+0.212 p=<0.0001** | **+0.254 p=<0.0001** | **+0.362 p=<0.0001** | **+0.427 p=<0.0001** | **+0.435 p=<0.0001** | **+0.104 p=0.0003** | **-0.086 p=<0.0001** |
| `embedded@ollaya:decider:2b` |  |  |  |  |  |  |  | **+0.144 p=<0.0001** | **+0.144 p=<0.0001** | **+0.186 p=<0.0001** | **+0.293 p=<0.0001** | **+0.358 p=<0.0001** | **+0.366 p=<0.0001** | +0.036 p=0.1303 | **-0.155 p=<0.0001** |
| `embedded@ollaya:decision:eos` |  |  |  |  |  |  |  |  | -0.000 p=0.9990 | +0.042 p=0.1703 | **+0.149 p=<0.0001** | **+0.215 p=<0.0001** | **+0.222 p=<0.0001** | **-0.108 p=0.0008** | **-0.298 p=<0.0001** |
| `embedded@ollaya:kev:0.8b` |  |  |  |  |  |  |  |  |  | +0.042 p=0.2166 | **+0.149 p=<0.0001** | **+0.215 p=<0.0001** | **+0.222 p=<0.0001** | **-0.108 p=0.0009** | **-0.298 p=<0.0001** |
| `embedded@ollaya:laya:en` |  |  |  |  |  |  |  |  |  |  | **+0.107 p=0.0002** | **+0.172 p=<0.0001** | **+0.180 p=<0.0001** | **-0.150 p=<0.0001** | **-0.341 p=<0.0001** |
| `embedded@ollaya:laya:typed-decisions` |  |  |  |  |  |  |  |  |  |  |  | +0.065 p=0.0637 | +0.073 p=0.0529 | **-0.257 p=<0.0001** | **-0.448 p=<0.0001** |
| `embedded@ollaya:von` |  |  |  |  |  |  |  |  |  |  |  |  | +0.008 p=0.8214 | **-0.323 p=<0.0001** | **-0.513 p=<0.0001** |
| `embedded@ollaya:nli:modernbert-large` |  |  |  |  |  |  |  |  |  |  |  |  |  | **-0.331 p=<0.0001** | **-0.521 p=<0.0001** |
| `embedded@decider:strands-decider-2b` |  |  |  |  |  |  |  |  |  |  |  |  |  |  | **-0.190 p=<0.0001** |

## Weighted query score: how much the ranking depends on the weights

Rank of each run over 300 random weight vectors that keep the order category ≥ filters ≥ residual words ≥ other word roles.

| Run | Score | Rank range | Most often |
|---|---|---|---|
| `embedded@jev:jev-latest` | 0.916 | 1–1 | 1 |
| `embedded@mlx:nimble` | 0.889 | 2–3 | 2 |
| `embedded@ollama:nimble` | 0.883 | 2–4 | 3 |
| `embedded@ollaya:jeb:4b` | 0.848 | 4–6 | 4 |
| `embedded@ollama:tev1` | 0.841 | 4–7 | 6 |
| `router@ollama:nimble` | 0.834 | 4–7 | 6 |
| `embedded@ollaya:winnow:e4b` | 0.830 | 2–7 | 7 |
| `embedded@ollaya:decider:2b` | 0.762 | 8–9 | 8 |
| `embedded@decider:strands-decider-2b` | 0.726 | 8–9 | 9 |
| `embedded@ollama:tev1:0.8b` | 0.631 | 10–11 | 10 |
| `embedded@ollaya:kev:0.8b` | 0.618 | 10–12 | 12 |
| `embedded@ollaya:decision:eos` | 0.618 | 11–12 | 11 |
| `embedded@ollaya:laya:en` | 0.576 | 13–13 | 13 |
| `embedded@ollaya:laya:typed-decisions` | 0.468 | 14–14 | 14 |
| `embedded@ollaya:von` | 0.403 | 15–15 | 15 |
| `embedded@ollaya:nli:modernbert-large` | 0.395 | 16–16 | 16 |
