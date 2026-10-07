# Results: eval split

- **Queries:** 1000, evaluated by every run. The eval split has 1000 rows in `data/eval_raw.jsonl`.
- **Decoding:** tuned per backend on the 100 dev rows (ids 0–99). Filters are masked by the predicted category's applicable attributes.
- **Runs:**
  - `embedded@ollama:nimble`: Ollama 0.35 `/v1/systemone` (llama.cpp, GGUF); questions scored one after another, split into several requests when the prompt exceeds the model context; model `nimble` (served as `not recorded`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.95; word-role `category` weight ×8
  - `router@ollama:nimble`: Ollama 0.35 `/v1/systemone` (llama.cpp, GGUF); questions scored one after another, split into several requests when the prompt exceeds the model context; model `nimble` (served as `not recorded`); category scheme top-level `cat_L1` + conditional group questions, 'no category' threshold 0.50; filter p ≥ 0.95; word-role `category` weight ×8
  - `embedded@ollama:tev1`: Ollama 0.35 `/v1/systemone` (llama.cpp, GGUF); questions scored one after another, split into several requests when the prompt exceeds the model context; model `tev1` (served as `tev1`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.70; word-role `category` weight ×16
  - `embedded@ollama:tev1:0.8b`: Ollama 0.35 `/v1/systemone` (llama.cpp, GGUF); questions scored one after another, split into several requests when the prompt exceeds the model context; model `tev1:0.8b` (served as `tev1:0.8b`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.70; word-role `category` weight ×8
  - `embedded@ollaya:jeb:4b`: Ollaya 0.9 local server (TypeSafe-compatible `/v1/systemone`), model as named in the run; model `jeb:4b` (served as `jeb:4b`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.70; word-role `category` weight ×16
  - `embedded@ollaya:winnow:e4b`: Ollaya 0.9 local server (TypeSafe-compatible `/v1/systemone`), model as named in the run; model `winnow:e4b` (served as `winnow:e4b`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.50; word-role `category` weight ×8
  - `embedded@decider:strands-decider-2b`: AWS Strands Labs' Strands Decider 2B (`StrandsAgents/strands-decider-2B-hobson-v19`, MLX) via its own `/v1/systemone` server; the state is encoded once and only each question's suffix is added, batched; model `strands-decider-2b` (served as `strands-decider-2b`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.50; word-role `category` weight ×1
  - `embedded@jev:jev-latest`: TypeSafe's hosted Jev API (`https://api.typesafe.ai`); latency includes the network round trip; model `jev-latest` (served as `jev-1.13.0`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.95; word-role `category` weight ×8
  - `embedded@openai:gpt-6-luna`: OpenAI's hosted Decisions API (`POST /v1/decisions`, gpt-6-luna); the state is sent as JSON text; latency includes the network round trip; model `gpt-6-luna` (served as `gpt-6-luna`); category scheme group questions with an `other_product` escape, no `cat_L1`, 'no category' threshold 0.70; filter p ≥ 0.80; word-role `category` weight ×8
- **Cells:** value (correct / counted) [95% Wilson interval]. F1 has no interval.

| Metric | Measure | Counted over | `embedded@ollama:nimble` | `router@ollama:nimble` | `embedded@ollama:tev1` | `embedded@ollama:tev1:0.8b` | `embedded@ollaya:jeb:4b` | `embedded@ollaya:winnow:e4b` | `embedded@decider:strands-decider-2b` | `embedded@jev:jev-latest` | `embedded@openai:gpt-6-luna` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Category, exact node | accuracy | all queries; the predicted node must equal the gold node (same depth; 'no category' is a value) | 0.917 (917/1000) [0.90–0.93] | 0.836 (836/1000) [0.81–0.86] | 0.942 (942/1000) [0.93–0.95] | 0.727 (727/1000) [0.70–0.75] | 0.918 (918/1000) [0.90–0.93] | 0.966 (966/1000) [0.95–0.98] | 0.743 (743/1000) [0.72–0.77] | 0.987 (987/1000) [0.98–0.99] | 0.987 (987/1000) [0.98–0.99] |
| Category correct at L1 | accuracy | queries with a gold category; predicted path has the gold L1 node | 0.977 (930/952) [0.97–0.98] | 0.867 (825/952) [0.84–0.89] | 0.987 (940/952) [0.98–0.99] | 0.809 (770/952) [0.78–0.83] | 0.973 (926/952) [0.96–0.98] | 0.985 (938/952) [0.98–0.99] | 0.803 (764/952) [0.78–0.83] | 0.996 (948/952) [0.99–1.00] | 0.998 (950/952) [0.99–1.00] |
| Category correct at L2 | accuracy | queries whose gold category is at depth ≥ 2; predicted path has the gold L2 node | 0.983 (851/866) [0.97–0.99] | 0.893 (773/866) [0.87–0.91] | 0.990 (857/866) [0.98–0.99] | 0.807 (699/866) [0.78–0.83] | 0.975 (844/866) [0.96–0.98] | 0.988 (856/866) [0.98–0.99] | 0.864 (748/866) [0.84–0.88] | 0.997 (863/866) [0.99–1.00] | 1.000 (866/866) [1.00–1.00] |
| Category correct at L3 | accuracy | queries whose gold category is at depth 3; predicted node is the gold L3 node | 0.984 (482/490) [0.97–0.99] | 0.857 (420/490) [0.82–0.89] | 0.978 (479/490) [0.96–0.99] | 0.796 (390/490) [0.76–0.83] | 0.955 (468/490) [0.93–0.97] | 0.965 (473/490) [0.95–0.98] | 0.824 (404/490) [0.79–0.86] | 0.994 (487/490) [0.98–1.00] | 0.992 (486/490) [0.98–1.00] |
| 'No category' precision | precision | queries predicted as 'no category' | 0.682 (15/22) [0.47–0.84] | 0.250 (23/92) [0.17–0.35] | 0.889 (24/27) [0.72–0.96] | n/a (0/0) | 0.900 (9/10) [0.60–0.98] | 0.957 (44/46) [0.85–0.99] | n/a (0/0) | 0.980 (48/49) [0.89–1.00] | 0.960 (48/50) [0.87–0.99] |
| 'No category' recall | recall | queries whose gold is 'no category' | 0.312 (15/48) [0.20–0.45] | 0.479 (23/48) [0.34–0.62] | 0.500 (24/48) [0.36–0.64] | 0.000 (0/48) [0.00–0.07] | 0.188 (9/48) [0.10–0.32] | 0.917 (44/48) [0.80–0.97] | 0.000 (0/48) [0.00–0.07] | 1.000 (48/48) [0.93–1.00] | 1.000 (48/48) [0.93–1.00] |
| Filters precision | precision (micro) | predicted attribute=value pairs, pooled over all queries | 0.904 (1399/1547) [0.89–0.92] | 0.913 (1325/1452) [0.90–0.93] | 0.786 (1449/1843) [0.77–0.80] | 0.644 (1013/1574) [0.62–0.67] | 0.860 (1393/1619) [0.84–0.88] | 0.817 (1313/1607) [0.80–0.84] | 0.760 (1275/1677) [0.74–0.78] | 0.963 (1341/1392) [0.95–0.97] | 0.929 (1302/1402) [0.91–0.94] |
| Filters recall | recall (micro) | gold attribute=value pairs, pooled over all queries | 0.920 (1399/1520) [0.91–0.93] | 0.872 (1325/1520) [0.85–0.89] | 0.953 (1449/1520) [0.94–0.96] | 0.666 (1013/1520) [0.64–0.69] | 0.916 (1393/1520) [0.90–0.93] | 0.864 (1313/1520) [0.85–0.88] | 0.839 (1275/1520) [0.82–0.86] | 0.882 (1341/1520) [0.87–0.90] | 0.857 (1302/1520) [0.84–0.87] |
| Filters F1 | F1 (micro) | harmonic mean of the two rows above | 0.912 | 0.892 | 0.862 | 0.655 | 0.888 | 0.840 | 0.798 | 0.921 | 0.891 |
| Filter set exact match | accuracy | all queries; predicted filter set equals the gold set (incl. both empty) | 0.770 (770/1000) [0.74–0.80] | 0.742 (742/1000) [0.71–0.77] | 0.612 (612/1000) [0.58–0.64] | 0.326 (326/1000) [0.30–0.36] | 0.720 (720/1000) [0.69–0.75] | 0.619 (619/1000) [0.59–0.65] | 0.508 (508/1000) [0.48–0.54] | 0.800 (800/1000) [0.77–0.82] | 0.726 (726/1000) [0.70–0.75] |
| Word-role accuracy | accuracy | all words of all queries; role category / filter / residual | 0.841 (4688/5576) [0.83–0.85] | 0.839 (4681/5576) [0.83–0.85] | 0.684 (3814/5576) [0.67–0.70] | 0.316 (1760/5576) [0.30–0.33] | 0.755 (4211/5576) [0.74–0.77] | 0.699 (3898/5576) [0.69–0.71] | 0.655 (3650/5576) [0.64–0.67] | 0.854 (4762/5576) [0.84–0.86] | 0.884 (4930/5576) [0.88–0.89] |
| Residual words precision | precision | words predicted 'residual' | 0.688 (1056/1536) [0.66–0.71] | 0.681 (1056/1551) [0.66–0.70] | 0.783 (952/1216) [0.76–0.81] | 0.000 (0/1) [0.00–0.79] | 0.707 (905/1280) [0.68–0.73] | 1.000 (135/135) [0.97–1.00] | 0.909 (239/263) [0.87–0.94] | 0.946 (738/780) [0.93–0.96] | 0.981 (813/829) [0.97–0.99] |
| Residual words recall | recall | words whose gold role is 'residual' | 0.786 (1056/1344) [0.76–0.81] | 0.786 (1056/1344) [0.76–0.81] | 0.708 (952/1344) [0.68–0.73] | 0.000 (0/1344) [0.00–0.00] | 0.673 (905/1344) [0.65–0.70] | 0.100 (135/1344) [0.09–0.12] | 0.178 (239/1344) [0.16–0.20] | 0.549 (738/1344) [0.52–0.58] | 0.605 (813/1344) [0.58–0.63] |
| Residual words F1 | F1 | harmonic mean of the two rows above | 0.733 | 0.730 | 0.744 | 0.000 | 0.690 | 0.183 | 0.297 | 0.695 | 0.748 |
| **Whole query exactly right** | accuracy | all queries; category exact, filter set exact, and the set of residual word positions exact | 0.424 (424/1000) [0.39–0.45] | 0.373 (373/1000) [0.34–0.40] | 0.336 (336/1000) [0.31–0.37] | 0.090 (90/1000) [0.07–0.11] | 0.365 (365/1000) [0.34–0.40] | 0.245 (245/1000) [0.22–0.27] | 0.193 (193/1000) [0.17–0.22] | 0.535 (535/1000) [0.50–0.57] | 0.512 (512/1000) [0.48–0.54] |
| **Weighted query score** | mean (0–1) | all queries; per query the weighted mean of category credit (0.45; partial for the right branch), filters F0.5 (0.30), residual words F2 (0.17) and the other word roles (0.08), over the parts the query involves (`sidm/score.py`); 95% interval of the mean | 0.880 [0.869–0.890] | 0.824 [0.810–0.839] | 0.850 [0.841–0.859] | 0.602 [0.586–0.619] | 0.853 [0.842–0.863] | 0.826 [0.817–0.835] | 0.694 [0.678–0.711] | 0.920 [0.913–0.927] | 0.919 [0.912–0.926] |
| Latency p50 / p95 | seconds | all queries; one request each, sequential, warm model | 15.6 s / 17.3 s | 15.9 s / 18.0 s | 9.9 s / 10.9 s | 2.2 s / 2.4 s | 10.8 s / 13.3 s | 6.2 s / 6.8 s | 3.9 s / 4.7 s | 0.3 s / 0.4 s | 0.4 s / 0.8 s |
| Server prefill / field evaluation | mean seconds | mlx backend only (nimble's own timings) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Questions / input tokens | mean per query | server-reported input tokens (Ollama: every question charged the full prompt; mlx: tokens actually processed) | 23.6 / 127k | 24.6 / 133k | 23.6 / 19k | 23.6 / 19k | 23.6 / 8k | 23.6 / 9k | 23.6 / 4k | 23.6 / 5k | 23.6 / 5k |

## Paired comparison against `embedded@ollama:nimble` (exact McNemar test, same queries)

| Run | Metric | correct only with `embedded@ollama:nimble` | correct only with the other | p-value |
|---|---|---|---|---|
| `router@ollama:nimble` | Category, exact node | 112 | 31 | 0.0000 |
| `router@ollama:nimble` | Whole query exactly right | 79 | 28 | 0.0000 |
| `embedded@ollama:tev1` | Category, exact node | 19 | 44 | 0.0022 |
| `embedded@ollama:tev1` | Whole query exactly right | 176 | 88 | 0.0000 |
| `embedded@ollama:tev1:0.8b` | Category, exact node | 212 | 22 | 0.0000 |
| `embedded@ollama:tev1:0.8b` | Whole query exactly right | 356 | 22 | 0.0000 |
| `embedded@ollaya:jeb:4b` | Category, exact node | 38 | 39 | 1.0000 |
| `embedded@ollaya:jeb:4b` | Whole query exactly right | 132 | 73 | 0.0000 |
| `embedded@ollaya:winnow:e4b` | Category, exact node | 14 | 63 | 0.0000 |
| `embedded@ollaya:winnow:e4b` | Whole query exactly right | 273 | 94 | 0.0000 |
| `embedded@decider:strands-decider-2b` | Category, exact node | 184 | 10 | 0.0000 |
| `embedded@decider:strands-decider-2b` | Whole query exactly right | 311 | 80 | 0.0000 |
| `embedded@jev:jev-latest` | Category, exact node | 4 | 74 | 0.0000 |
| `embedded@jev:jev-latest` | Whole query exactly right | 104 | 215 | 0.0000 |
| `embedded@openai:gpt-6-luna` | Category, exact node | 2 | 72 | 0.0000 |
| `embedded@openai:gpt-6-luna` | Whole query exactly right | 126 | 214 | 0.0000 |

## All pairs: Category, exact node (exact McNemar test)

Cell (row, column) = queries correct only with the row run : only with the column run, and the p-value. **Bold** = significant at p < 0.05; the better run of the pair is the one with the larger count.

| | `router@ollama:nimble` | `embedded@ollama:tev1` | `embedded@ollama:tev1:0.8b` | `embedded@ollaya:jeb:4b` | `embedded@ollaya:winnow:e4b` | `embedded@decider:strands-decider-2b` | `embedded@jev:jev-latest` | `embedded@openai:gpt-6-luna` |
|---|---|---|---|---|---|---|---|---|
| `embedded@ollama:nimble` | **112:31 p=<0.0001** | **19:44 p=0.0022** | **212:22 p=<0.0001** | 38:39 p=1.0000 | **14:63 p=<0.0001** | **184:10 p=<0.0001** | **4:74 p=<0.0001** | **2:72 p=<0.0001** |
| `router@ollama:nimble` |  | **24:130 p=<0.0001** | **210:101 p=<0.0001** | **38:120 p=<0.0001** | **9:139 p=<0.0001** | **148:55 p=<0.0001** | **2:153 p=<0.0001** | **1:152 p=<0.0001** |
| `embedded@ollama:tev1` |  |  | **225:10 p=<0.0001** | **36:12 p=0.0007** | **14:38 p=0.0012** | **207:8 p=<0.0001** | **2:47 p=<0.0001** | **2:47 p=<0.0001** |
| `embedded@ollama:tev1:0.8b` |  |  |  | **14:205 p=<0.0001** | **8:247 p=<0.0001** | 122:138 p=0.3523 | **4:264 p=<0.0001** | **3:263 p=<0.0001** |
| `embedded@ollaya:jeb:4b` |  |  |  |  | **11:59 p=<0.0001** | **184:9 p=<0.0001** | **3:72 p=<0.0001** | **3:72 p=<0.0001** |
| `embedded@ollaya:winnow:e4b` |  |  |  |  |  | **229:6 p=<0.0001** | **4:25 p=0.0001** | **3:24 p=<0.0001** |
| `embedded@decider:strands-decider-2b` |  |  |  |  |  |  | **4:248 p=<0.0001** | **0:244 p=<0.0001** |
| `embedded@jev:jev-latest` |  |  |  |  |  |  |  | 6:6 p=1.0000 |

## All pairs: Whole query exactly right (exact McNemar test)

Cell (row, column) = queries correct only with the row run : only with the column run, and the p-value. **Bold** = significant at p < 0.05; the better run of the pair is the one with the larger count.

| | `router@ollama:nimble` | `embedded@ollama:tev1` | `embedded@ollama:tev1:0.8b` | `embedded@ollaya:jeb:4b` | `embedded@ollaya:winnow:e4b` | `embedded@decider:strands-decider-2b` | `embedded@jev:jev-latest` | `embedded@openai:gpt-6-luna` |
|---|---|---|---|---|---|---|---|---|
| `embedded@ollama:nimble` | **79:28 p=<0.0001** | **176:88 p=<0.0001** | **356:22 p=<0.0001** | **132:73 p=<0.0001** | **273:94 p=<0.0001** | **311:80 p=<0.0001** | **104:215 p=<0.0001** | **126:214 p=<0.0001** |
| `router@ollama:nimble` |  | **162:125 p=0.0334** | **315:32 p=<0.0001** | 120:112 p=0.6459 | **242:114 p=<0.0001** | **270:90 p=<0.0001** | **91:253 p=<0.0001** | **108:247 p=<0.0001** |
| `embedded@ollama:tev1` |  |  | **269:23 p=<0.0001** | **87:116 p=0.0491** | **178:87 p=<0.0001** | **220:77 p=<0.0001** | **66:265 p=<0.0001** | **80:256 p=<0.0001** |
| `embedded@ollama:tev1:0.8b` |  |  |  | **25:300 p=<0.0001** | **14:169 p=<0.0001** | **38:141 p=<0.0001** | **4:449 p=<0.0001** | **8:430 p=<0.0001** |
| `embedded@ollaya:jeb:4b` |  |  |  |  | **200:80 p=<0.0001** | **250:78 p=<0.0001** | **54:224 p=<0.0001** | **73:220 p=<0.0001** |
| `embedded@ollaya:winnow:e4b` |  |  |  |  |  | **121:69 p=0.0002** | **17:307 p=<0.0001** | **32:299 p=<0.0001** |
| `embedded@decider:strands-decider-2b` |  |  |  |  |  |  | **40:382 p=<0.0001** | **33:352 p=<0.0001** |
| `embedded@jev:jev-latest` |  |  |  |  |  |  |  | 128:105 p=0.1494 |

## All pairs: weighted query score (paired sign-flip test)

Cell (row, column) = the row run's mean score minus the column run's, and the p-value. **Bold** = significant at p < 0.05.

| | `router@ollama:nimble` | `embedded@ollama:tev1` | `embedded@ollama:tev1:0.8b` | `embedded@ollaya:jeb:4b` | `embedded@ollaya:winnow:e4b` | `embedded@decider:strands-decider-2b` | `embedded@jev:jev-latest` | `embedded@openai:gpt-6-luna` |
|---|---|---|---|---|---|---|---|---|
| `embedded@ollama:nimble` | **+0.055 p=<0.0001** | **+0.030 p=<0.0001** | **+0.277 p=<0.0001** | **+0.027 p=<0.0001** | **+0.053 p=<0.0001** | **+0.185 p=<0.0001** | **-0.041 p=<0.0001** | **-0.040 p=<0.0001** |
| `router@ollama:nimble` |  | **-0.026 p=0.0007** | **+0.222 p=<0.0001** | **-0.028 p=0.0001** | -0.002 p=0.7963 | **+0.130 p=<0.0001** | **-0.096 p=<0.0001** | **-0.095 p=<0.0001** |
| `embedded@ollama:tev1` |  |  | **+0.248 p=<0.0001** | -0.003 p=0.5710 | **+0.024 p=<0.0001** | **+0.156 p=<0.0001** | **-0.070 p=<0.0001** | **-0.069 p=<0.0001** |
| `embedded@ollama:tev1:0.8b` |  |  |  | **-0.250 p=<0.0001** | **-0.224 p=<0.0001** | **-0.092 p=<0.0001** | **-0.318 p=<0.0001** | **-0.317 p=<0.0001** |
| `embedded@ollaya:jeb:4b` |  |  |  |  | **+0.026 p=<0.0001** | **+0.158 p=<0.0001** | **-0.068 p=<0.0001** | **-0.066 p=<0.0001** |
| `embedded@ollaya:winnow:e4b` |  |  |  |  |  | **+0.132 p=<0.0001** | **-0.094 p=<0.0001** | **-0.093 p=<0.0001** |
| `embedded@decider:strands-decider-2b` |  |  |  |  |  |  | **-0.226 p=<0.0001** | **-0.225 p=<0.0001** |
| `embedded@jev:jev-latest` |  |  |  |  |  |  |  | +0.001 p=0.7142 |

## Weighted query score: how much the ranking depends on the weights

Rank of each run over 300 random weight vectors that keep the order category ≥ filters ≥ residual words ≥ other word roles.

| Run | Score | Rank range | Most often |
|---|---|---|---|
| `embedded@jev:jev-latest` | 0.920 | 1–2 | 2 |
| `embedded@openai:gpt-6-luna` | 0.919 | 1–2 | 1 |
| `embedded@ollama:nimble` | 0.880 | 3–5 | 3 |
| `embedded@ollaya:jeb:4b` | 0.853 | 4–6 | 4 |
| `embedded@ollama:tev1` | 0.850 | 4–6 | 5 |
| `embedded@ollaya:winnow:e4b` | 0.826 | 3–7 | 7 |
| `router@ollama:nimble` | 0.824 | 4–7 | 6 |
| `embedded@decider:strands-decider-2b` | 0.694 | 8–8 | 8 |
| `embedded@ollama:tev1:0.8b` | 0.602 | 9–9 | 9 |
