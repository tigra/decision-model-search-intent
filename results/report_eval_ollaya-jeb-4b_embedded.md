# Search intent parsing: `embedded@ollaya:jeb:4b` via /v1/systemone

1000 queries, one request per query (23.6 questions on average).

## Accuracy

| Metric | **tuned decoding** (group-max category, filter p≥0.95, masked) | tuned, unmasked filters | untuned (plain argmax) |
|---|---|---|---|
| Category exact node acc | 0.918 | 0.918 | 0.930 |
| Category L1 acc (given gold depth ≥ 1) | 0.973 | 0.973 | 0.974 |
| Category L2 acc (given gold depth ≥ 2) | 0.975 | 0.975 | 0.975 |
| Category L3 acc (given gold depth ≥ 3) | 0.955 | 0.955 | 0.955 |
| Category hierarchical F1 | 0.931 | 0.931 | 0.944 |
| 'No category' P / R (tp/fp/fn) | 0.90 / 0.19 (9/1/39) | 0.90 / 0.19 (9/1/39) | 1.00 / 0.44 (21/0/27) |
| Filters micro P / R / F1 | 0.860 / 0.916 / 0.888 | 0.855 / 0.925 / 0.888 | 0.774 / 0.949 / 0.852 |
| Filter set exact match | 0.720 | 0.717 | 0.630 |
| Word-role accuracy | 0.755 | 0.755 | 0.769 |
| Word-role macro F1 | 0.747 | 0.747 | 0.758 |
| Residual words P / R / F1 | 0.707 / 0.673 / 0.690 | 0.707 / 0.673 / 0.690 | 0.554 / 0.722 / 0.627 |
| Residual set exact match | 0.486 | 0.486 | 0.325 |
| **Full query exact match** | **0.365** | **0.364** | **0.230** |

### Per-attribute filter scores (masked)

| attribute | support | P | R | F1 |
|---|---|---|---|---|
| feature | 220 | 0.927 | 0.977 | 0.951 |
| color | 211 | 0.783 | 0.991 | 0.874 |
| material | 185 | 0.907 | 0.741 | 0.815 |
| style | 182 | 0.995 | 0.995 | 0.995 |
| room | 147 | 0.716 | 0.946 | 0.815 |
| leg_color | 142 | 0.847 | 0.937 | 0.890 |
| leg_material | 132 | 0.695 | 0.879 | 0.776 |
| bed_size | 103 | 1.000 | 0.951 | 0.975 |
| width | 83 | 0.982 | 0.663 | 0.791 |
| shape | 66 | 0.970 | 0.970 | 0.970 |
| firmness | 36 | 1.000 | 1.000 | 1.000 |
| seat_count | 13 | 0.769 | 0.769 | 0.769 |

### Word roles (gold → predicted)

| gold \ pred | category | filter | residual |
|---|---|---|---|
| category | 1717 | 7 | 30 |
| filter | 544 | 1589 | 345 |
| residual | 169 | 270 | 905 |

### Top category confusions (gold → predicted)

- None → tables: 27
- chaise_lounges → daybeds: 8
- curio_cabinet → cabinets: 5
- None → storage: 5
- None → desks: 5
- sleeper_sofa → daybeds: 5
- beds → mattresses: 2
- accent_cabinet → side_tables: 2
- loveseat → sofas: 2
- step_stool → side_tables: 2
- desks → desk: 2
- seating → console_tables: 1
- seating → None: 1
- lounge_chair → daybeds: 1
- step_stool → dining_tables: 1

## Latency (sequential, warm model)

| mean | p50 | p90 | p95 | p99 | max | queries/s | ms per question | mean input tokens |
|---|---|---|---|---|---|---|---|---|
| 11.036s | 10.769s | 12.763s | 13.287s | 14.518s | 17.106s | 0.09 | 468.1 | 8328 |

p50 latency by query length: 1-3 words: 9.720s (n=185), 4-6 words: 10.641s (n=505), 7-9 words: 12.010s (n=258), 10-12 words: 13.710s (n=47), 13-15 words: 15.413s (n=5)

Truncated queries (more than the word-question budget): 0

## 20 worst examples

- `furniture with acrylic legs vintage style silver legs for small space`  
  gold: cat=None filters={'leg_material': 'acrylic', 'style': 'vintage', 'leg_color': 'silver'} roles=rffffrffrrr  
  pred: cat=tables filters={'leg_material': 'acrylic', 'leg_color': 'silver', 'style': 'vintage'} roles=crfcfcfcrff
- `two seater sofa 42 in under 500 dollars`  
  gold: cat=loveseat filters={'width': '30_to_48in'} roles=cccffrrr  
  pred: cat=sofas filters={'seat_count': 'seats_2'} roles=ffcfrffr
- `furniture with natural wood legs oak legs hexagonal deals for tall people`  
  gold: cat=None filters={'leg_color': 'natural', 'leg_material': 'oak', 'shape': 'hexagonal'} roles=rfffffffrrrr  
  pred: cat=tables filters={'shape': 'hexagonal'} roles=crffcfcfrrcr
- `beige furniture with wooden legs traditional style pet friendly sturdy`  
  gold: cat=None filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=frffffrrrr  
  pred: cat=tables filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=fcrfffcfff
- `seats stacking dining room brass legs sturdy`  
  gold: cat=seating filters={'feature': 'stackable', 'leg_color': 'brass', 'room': 'dining_room'} roles=cfffffr  
  pred: cat=dining_tables filters={'leg_material': 'metal', 'leg_color': 'brass', 'feature': 'stackable', 'room': 'dining_room'} roles=ccccfcf
- `furniture with natural wood legs and chrome base navy`  
  gold: cat=None filters={'leg_color': 'natural', 'leg_material': 'chrome', 'color': 'navy'} roles=rffffrfff  
  pred: cat=tables filters={'color': 'navy'} roles=crffcrfrf
- `chaise lounge with hairpin legs outdoor under $300`  
  gold: cat=chaise_lounges filters={'room': 'outdoor', 'leg_material': 'hairpin'} roles=ccffffrr  
  pred: cat=daybeds filters={'room': 'outdoor'} roles=ccrfcfff
- `lounger white legs on wheels tempered glass fast delivery`  
  gold: cat=lounge_chair filters={'leg_color': 'white', 'feature': 'wheels', 'material': 'glass'} roles=cffffffrr  
  pred: cat=daybeds filters={'color': 'white', 'feature': 'wheels'} roles=cfcrfffrr
- `china cabinet with glass doors for small apartment luxury`  
  gold: cat=curio_cabinet filters={'feature': 'glass_doors'} roles=ccfffrrrr  
  pred: cat=cabinets filters={'feature': 'glass_doors'} roles=ccrfcrffc
- `sofa bed rustic living room 22 in sturdy near me`  
  gold: cat=sleeper_sofa filters={'style': 'rustic', 'room': 'living_room', 'width': 'under_30in'} roles=ccfffffrrr  
  pred: cat=daybeds filters={'style': 'rustic', 'room': 'living_room'} roles=ccfrrfrfrr
- `china cabinet rustic 315 cm pink handmade 2025`  
  gold: cat=curio_cabinet filters={'style': 'rustic', 'width': 'over_90in', 'color': 'pink'} roles=ccffffrr  
  pred: cat=cabinets filters={'color': 'pink', 'style': 'rustic'} roles=cccccffr
- `dresser with drawers brass legs 114 cm under 500 dollars`  
  gold: cat=dressers filters={'leg_color': 'brass', 'width': '30_to_48in', 'feature': 'with_drawers'} roles=cffffffrrr  
  pred: cat=dressers filters={'leg_material': 'metal', 'leg_color': 'brass', 'feature': 'with_drawers'} roles=crcfcffffr
- `bed tufted solid wood bedroom 2024`  
  gold: cat=beds filters={'feature': 'tufted', 'room': 'bedroom', 'material': 'solid_wood'} roles=cffffr  
  pred: cat=mattresses filters={'feature': 'tufted'} roles=cfcfcr
- `china cabinet living room 37 in`  
  gold: cat=curio_cabinet filters={'room': 'living_room', 'width': '30_to_48in'} roles=ccffff  
  pred: cat=cabinets filters={'room': 'living_room'} roles=ccrrfr
- `drop leaf table brass legs walnut legs foldable in stock pet friendly`  
  gold: cat=drop_leaf_table filters={'leg_color': 'brass', 'leg_material': 'walnut', 'feature': 'foldable'} roles=cccfffffrrrr  
  pred: cat=drop_leaf_table filters={'leg_material': 'walnut', 'leg_color': 'brass', 'feature': 'foldable'} roles=rccfcfcfrcff
- `accent chair with gold legs and marble for small apartment`  
  gold: cat=accent_chair filters={'leg_color': 'gold', 'material': 'marble'} roles=ccfffffrrr  
  pred: cat=accent_chair filters={'leg_color': 'gold', 'room': 'living_room'} roles=ccrffrfrff
- `furniture with chrome legs glam round`  
  gold: cat=None filters={'leg_material': 'chrome', 'style': 'glam', 'shape': 'round'} roles=rfffff  
  pred: cat=tables filters={'leg_material': 'chrome', 'leg_color': 'silver', 'style': 'glam', 'shape': 'round'} roles=crffcf
- `wall shelf with acrylic legs and natural wood legs for tall people under 500 dollars`  
  gold: cat=wall_shelf filters={'leg_material': 'acrylic', 'leg_color': 'natural'} roles=ccfffffffrrrrrr  
  pred: cat=wall_shelf filters={'leg_color': 'natural'} roles=ccrffrfffrfrffr
- `adjustable base full foldable heavy duty comfortable`  
  gold: cat=adjustable_bases filters={'feature': 'foldable', 'bed_size': 'full'} roles=ccffrrr  
  pred: cat=adjustable_bases filters={'feature': 'foldable'} roles=fccffff
- `furniture with metal legs and white legs`  
  gold: cat=None filters={'leg_material': 'metal', 'leg_color': 'white'} roles=rffffff  
  pred: cat=tables filters={'color': 'white', 'leg_material': 'metal', 'leg_color': 'white'} roles=crffrff
