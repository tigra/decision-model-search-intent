# Search intent parsing: `embedded@ollama:nimble` via /v1/systemone

1000 queries, one request per query (23.6 questions on average).

## Accuracy

| Metric | **tuned decoding** (group-max category, filter p≥0.95, masked) | tuned, unmasked filters | untuned (plain argmax) |
|---|---|---|---|
| Category exact node acc | 0.917 | 0.917 | 0.919 |
| Category L1 acc (given gold depth ≥ 1) | 0.977 | 0.977 | 0.973 |
| Category L2 acc (given gold depth ≥ 2) | 0.983 | 0.983 | 0.983 |
| Category L3 acc (given gold depth ≥ 3) | 0.984 | 0.984 | 0.984 |
| Category hierarchical F1 | 0.937 | 0.937 | 0.938 |
| 'No category' P / R (tp/fp/fn) | 0.68 / 0.31 (15/7/33) | 0.68 / 0.31 (15/7/33) | 0.63 / 0.40 (19/11/29) |
| Filters micro P / R / F1 | 0.904 / 0.920 / 0.912 | 0.899 / 0.920 / 0.910 | 0.723 / 0.975 / 0.830 |
| Filter set exact match | 0.770 | 0.762 | 0.587 |
| Word-role accuracy | 0.841 | 0.841 | 0.820 |
| Word-role macro F1 | 0.834 | 0.834 | 0.816 |
| Residual words P / R / F1 | 0.688 / 0.786 / 0.733 | 0.688 / 0.786 / 0.733 | 0.636 / 0.830 / 0.720 |
| Residual set exact match | 0.527 | 0.527 | 0.483 |
| **Full query exact match** | **0.424** | **0.420** | **0.318** |

### Per-attribute filter scores (masked)

| attribute | support | P | R | F1 |
|---|---|---|---|---|
| feature | 220 | 0.995 | 0.982 | 0.989 |
| color | 211 | 0.921 | 0.995 | 0.957 |
| material | 185 | 0.969 | 0.843 | 0.902 |
| style | 182 | 0.995 | 1.000 | 0.997 |
| room | 147 | 0.770 | 0.959 | 0.855 |
| leg_color | 142 | 0.863 | 0.979 | 0.917 |
| leg_material | 132 | 0.674 | 0.909 | 0.774 |
| bed_size | 103 | 1.000 | 0.990 | 0.995 |
| width | 83 | 1.000 | 0.277 | 0.434 |
| shape | 66 | 1.000 | 0.939 | 0.969 |
| firmness | 36 | 1.000 | 0.972 | 0.986 |
| seat_count | 13 | 0.929 | 1.000 | 0.963 |

### Word roles (gold → predicted)

| gold \ pred | category | filter | residual |
|---|---|---|---|
| category | 1678 | 46 | 30 |
| filter | 74 | 1954 | 450 |
| residual | 105 | 183 | 1056 |

### Top category confusions (gold → predicted)

- None → tables: 9
- None → dressers: 6
- None → coffee_tables: 5
- None → drop_leaf_table: 4
- drafting_tables → console_tables: 4
- None → sofas: 4
- desks → desk: 4
- mattresses → hybrid_mattress: 3
- curio_cabinet → None: 2
- seating → office_chair: 2
- desk → coffee_tables: 2
- cabinets → curio_cabinet: 2
- desks → tables: 2
- loveseat → seating: 2
- seating → sofas: 2

## Latency (sequential, warm model)

| mean | p50 | p90 | p95 | p99 | max | queries/s | ms per question | mean input tokens |
|---|---|---|---|---|---|---|---|---|
| 15.642s | 15.564s | 16.792s | 17.270s | 18.333s | 19.028s | 0.06 | 663.5 | 126963 |

p50 latency by query length: 1-3 words: 14.590s (n=185), 4-6 words: 15.359s (n=505), 7-9 words: 16.252s (n=258), 10-12 words: 17.240s (n=47), 13-15 words: 18.237s (n=5)

Truncated queries (more than the word-question budget): 0

## 20 worst examples

- `furniture with natural wood legs oak legs hexagonal deals for tall people`  
  gold: cat=None filters={'leg_color': 'natural', 'leg_material': 'oak', 'shape': 'hexagonal'} roles=rfffffffrrrr  
  pred: cat=drop_leaf_table filters={'leg_material': 'oak', 'shape': 'hexagonal'} roles=crcrfrrfrrrr
- `two seater sofa 42 in under 500 dollars`  
  gold: cat=loveseat filters={'width': '30_to_48in'} roles=cccffrrr  
  pred: cat=seating filters={} roles=ffcfrffr
- `furniture with acrylic legs vintage style silver legs for small space`  
  gold: cat=None filters={'leg_material': 'acrylic', 'style': 'vintage', 'leg_color': 'silver'} roles=rffffrffrrr  
  pred: cat=coffee_tables filters={'leg_material': 'acrylic', 'leg_color': 'silver', 'style': 'vintage'} roles=crffffcfrfr
- `wall shelf with acrylic legs and natural wood legs for tall people under 500 dollars`  
  gold: cat=wall_shelf filters={'leg_material': 'acrylic', 'leg_color': 'natural'} roles=ccfffffffrrrrrr  
  pred: cat=wall_shelf filters={'leg_color': 'natural'} roles=ccrffrcffrcrffr
- `beige furniture with wooden legs traditional style pet friendly sturdy`  
  gold: cat=None filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=frffffrrrr  
  pred: cat=sofas filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=fcrffffffr
- `loveseat with acrylic legs under 30 inch two seater sofa 23 inches`  
  gold: cat=loveseat filters={'leg_material': 'acrylic', 'width': 'under_30in'} roles=cffffffcccrr  
  pred: cat=loveseat filters={'leg_material': 'acrylic', 'seat_count': 'seats_2'} roles=crfffffffcff
- `cabinet for my cat with natural wood legs like pottery barn in home office`  
  gold: cat=cabinets filters={'room': 'office', 'leg_color': 'natural'} roles=crrrffffrrrfff  
  pred: cat=None filters={'leg_color': 'natural', 'room': 'office'} roles=crrcrfffrrrrrc
- `dresser with drawers brass legs 114 cm under 500 dollars`  
  gold: cat=dressers filters={'leg_color': 'brass', 'width': '30_to_48in', 'feature': 'with_drawers'} roles=cffffffrrr  
  pred: cat=dressers filters={'leg_material': 'metal', 'leg_color': 'brass', 'feature': 'with_drawers'} roles=crrfffrffr
- `buffet for home office with storag`  
  gold: cat=sideboards filters={'room': 'office'} roles=cfffrr  
  pred: cat=sideboards filters={'feature': 'with_storage', 'room': 'office'} roles=crrcff
- `furniture with metal legs and white legs`  
  gold: cat=None filters={'leg_material': 'metal', 'leg_color': 'white'} roles=rffffff  
  pred: cat=tables filters={'leg_material': 'metal', 'leg_color': 'white'} roles=crffrrf
- `seats for kids microfiber gold legs free shipping`  
  gold: cat=seating filters={'room': 'kids_room', 'leg_color': 'gold', 'material': 'microfiber'} roles=cfffffrr  
  pred: cat=sofas filters={'material': 'microfiber', 'leg_material': 'walnut', 'leg_color': 'gold', 'room': 'kids_room'} roles=crrfcfrr
- `white legs leather blue furniture`  
  gold: cat=None filters={'leg_color': 'white', 'material': 'leather', 'color': 'blue'} roles=ffffr  
  pred: cat=sofas filters={'material': 'leather', 'leg_material': 'oak', 'leg_color': 'white'} roles=rfffc
- `furniture with walnut legs ergonomic used`  
  gold: cat=None filters={'leg_material': 'walnut'} roles=rfffrr  
  pred: cat=office_chair filters={'leg_material': 'walnut'} roles=crffff
- `office furniture for kids with hairpin legs boucle`  
  gold: cat=desks filters={'room': 'kids_room', 'leg_material': 'hairpin', 'material': 'boucle'} roles=ccffffff  
  pred: cat=desk filters={'material': 'boucle', 'leg_material': 'hairpin', 'room': 'kids_room'} roles=ccrrrffc
- `furniture with white legs marble bohemian`  
  gold: cat=None filters={'leg_color': 'white', 'material': 'marble', 'style': 'bohemian'} roles=rfffff  
  pred: cat=coffee_tables filters={'material': 'marble', 'leg_material': 'oak', 'leg_color': 'white', 'style': 'bohemian'} roles=crrfff
- `furniture with natural wood legs and chrome base navy`  
  gold: cat=None filters={'leg_color': 'natural', 'leg_material': 'chrome', 'color': 'navy'} roles=rffffrfff  
  pred: cat=dressers filters={'color': 'navy', 'leg_color': 'natural'} roles=crfffrfrf
- `rocking chair with natural wood legs oak legs for entryway`  
  gold: cat=rocking_chair filters={'leg_color': 'natural', 'leg_material': 'oak', 'room': 'entryway'} roles=ccffffffff  
  pred: cat=rocking_chair filters={'leg_material': 'oak', 'room': 'entryway'} roles=ccrcrfrfrf
- `storage furniture with glass doors chrome legs`  
  gold: cat=storage filters={'leg_material': 'chrome', 'feature': 'glass_doors'} roles=ccfffff  
  pred: cat=curio_cabinet filters={'leg_material': 'chrome', 'leg_color': 'silver', 'feature': 'glass_doors'} roles=ccrrfff
- `furniture with hairpin legs`  
  gold: cat=None filters={'leg_material': 'hairpin'} roles=rfff  
  pred: cat=desk filters={'leg_material': 'hairpin'} roles=crcf
- `cabinet with glass doors bohemian 46 in ergonomic`  
  gold: cat=cabinets filters={'feature': 'glass_doors', 'width': '30_to_48in', 'style': 'bohemian'} roles=cffffffr  
  pred: cat=curio_cabinet filters={'style': 'bohemian', 'feature': 'glass_doors'} roles=cfffffrc
