# Search intent parsing: `embedded@decider:strands-decider-2b` via /v1/systemone

1000 queries, one request per query (23.6 questions on average).

## Accuracy

| Metric | **tuned decoding** (group-max category, filter p≥0.95, masked) | tuned, unmasked filters | untuned (plain argmax) |
|---|---|---|---|
| Category exact node acc | 0.743 | 0.743 | 0.743 |
| Category L1 acc (given gold depth ≥ 1) | 0.803 | 0.803 | 0.803 |
| Category L2 acc (given gold depth ≥ 2) | 0.864 | 0.864 | 0.864 |
| Category L3 acc (given gold depth ≥ 3) | 0.824 | 0.824 | 0.824 |
| Category hierarchical F1 | 0.760 | 0.760 | 0.760 |
| 'No category' P / R (tp/fp/fn) | 1.00 / 0.00 (0/0/48) | 1.00 / 0.00 (0/0/48) | 1.00 / 0.00 (0/0/48) |
| Filters micro P / R / F1 | 0.760 / 0.839 / 0.798 | 0.726 / 0.872 / 0.793 | 0.687 / 0.858 / 0.763 |
| Filter set exact match | 0.508 | 0.465 | 0.417 |
| Word-role accuracy | 0.655 | 0.655 | 0.655 |
| Word-role macro F1 | 0.569 | 0.569 | 0.569 |
| Residual words P / R / F1 | 0.909 / 0.178 / 0.297 | 0.909 / 0.178 / 0.297 | 0.909 / 0.178 / 0.297 |
| Residual set exact match | 0.441 | 0.441 | 0.441 |
| **Full query exact match** | **0.193** | **0.167** | **0.151** |

### Per-attribute filter scores (masked)

| attribute | support | P | R | F1 |
|---|---|---|---|---|
| feature | 220 | 0.922 | 0.914 | 0.918 |
| color | 211 | 0.803 | 0.929 | 0.862 |
| material | 185 | 0.726 | 0.886 | 0.798 |
| style | 182 | 0.989 | 0.995 | 0.992 |
| room | 147 | 0.397 | 0.952 | 0.560 |
| leg_color | 142 | 0.992 | 0.930 | 0.960 |
| leg_material | 132 | 0.771 | 0.917 | 0.837 |
| bed_size | 103 | 0.786 | 0.427 | 0.553 |
| width | 83 | 0.704 | 0.229 | 0.345 |
| shape | 66 | 0.981 | 0.803 | 0.883 |
| firmness | 36 | 1.000 | 0.444 | 0.615 |
| seat_count | 13 | 0.800 | 0.615 | 0.696 |

### Word roles (gold → predicted)

| gold \ pred | category | filter | residual |
|---|---|---|---|
| category | 941 | 790 | 23 |
| filter | 7 | 2470 | 1 |
| residual | 66 | 1039 | 239 |

### Top category confusions (gold → predicted)

- None → coffee_tables: 29
- pub_table → desk: 18
- None → bed_frames: 14
- seating → coffee_tables: 14
- tables → desk: 13
- mattresses → coffee_tables: 9
- desks → coffee_tables: 8
- cabinets → filing_cabinets: 7
- curio_cabinet → cabinets: 7
- memory_foam_mattress → coffee_tables: 7
- bookcases → coffee_tables: 6
- seating → desk: 5
- accent_chair → coffee_tables: 5
- chaise_lounges → coffee_tables: 5
- storage → coffee_tables: 5

## Latency (sequential, warm model)

| mean | p50 | p90 | p95 | p99 | max | queries/s | ms per question | mean input tokens |
|---|---|---|---|---|---|---|---|---|
| 4.011s | 3.947s | 4.562s | 4.716s | 5.108s | 5.595s | 0.25 | 170.1 | 4037 |

p50 latency by query length: 1-3 words: 3.553s (n=185), 4-6 words: 3.913s (n=505), 7-9 words: 4.392s (n=258), 10-12 words: 4.884s (n=47), 13-15 words: 5.419s (n=5)

Truncated queries (more than the word-question budget): 0

## 20 worst examples

- `furniture with engineered wood and acrylic legs like west elm heavy duty`  
  gold: cat=None filters={'material': 'engineered_wood', 'leg_material': 'acrylic'} roles=rrffrffrrrrr  
  pred: cat=bed_frames filters={} roles=cfffffffffff
- `memory foam mattress extra firm twin for small apartmet`  
  gold: cat=memory_foam_mattress filters={'firmness': 'extra_firm', 'bed_size': 'twin'} roles=cccfffrrr  
  pred: cat=sofas filters={'room': 'bedroom'} roles=fffffffff
- `furniture with acrylic legs vintage style silver legs for small space`  
  gold: cat=None filters={'leg_material': 'acrylic', 'style': 'vintage', 'leg_color': 'silver'} roles=rffffrffrrr  
  pred: cat=bed_frames filters={'material': 'plastic', 'style': 'vintage'} roles=cffffffffff
- `kids bed with storage california king japandi on sale like pottery barn`  
  gold: cat=kids_beds filters={'feature': 'with_storage', 'bed_size': 'california_king', 'style': 'japandi'} roles=ccfffffrrrrr  
  pred: cat=kids_beds filters={'style': 'japandi', 'bed_size': 'king', 'feature': 'with_storage', 'room': 'bedroom'} roles=ffffffffffff
- `memory foam mattress king under 500 dollars gift`  
  gold: cat=memory_foam_mattress filters={'bed_size': 'king'} roles=cccfrrrr  
  pred: cat=sofas filters={'room': 'bedroom'} roles=fffffffr
- `office furniture with walnut legs modern made in usa for studio`  
  gold: cat=desks filters={'leg_material': 'walnut', 'style': 'modern'} roles=ccffffrrrrr  
  pred: cat=coffee_tables filters={'material': 'walnut', 'leg_material': 'walnut', 'style': 'modern', 'room': 'office'} roles=fcfffffffff
- `step stool white reclining pet friendly like west elm`  
  gold: cat=step_stool filters={'color': 'white', 'feature': 'reclining'} roles=ccffrrrrr  
  pred: cat=coffee_tables filters={'color': 'white'} roles=fffffffff
- `office furniture white farmhouse desk ikea under 300 dark brown legs`  
  gold: cat=desks filters={'leg_color': 'dark_brown', 'color': 'white', 'style': 'farmhouse'} roles=ccffcrrrfff  
  pred: cat=coffee_tables filters={'color': 'white', 'leg_color': 'dark_brown', 'style': 'farmhouse', 'width': 'under_30in', 'room': 'office'} roles=fffffffffff
- `memory foam mattress twin extra firm coolint`  
  gold: cat=memory_foam_mattress filters={'bed_size': 'twin', 'feature': 'cooling', 'firmness': 'extra_firm'} roles=cccffff  
  pred: cat=sofas filters={'room': 'bedroom'} roles=fffffff
- `two seater sofa 42 in under 500 dollars`  
  gold: cat=loveseat filters={'width': '30_to_48in'} roles=cccffrrr  
  pred: cat=sofas filters={'seat_count': 'seats_2'} roles=ffcfffff
- `china cabinet with glass doors for small apartment luxury`  
  gold: cat=curio_cabinet filters={'feature': 'glass_doors'} roles=ccfffrrrr  
  pred: cat=cabinets filters={'material': 'glass', 'feature': 'glass_doors'} roles=fffffffff
- `wall shelf with acrylic legs and natural wood legs for tall people under 500 dollars`  
  gold: cat=wall_shelf filters={'leg_material': 'acrylic', 'leg_color': 'natural'} roles=ccfffffffrrrrrr  
  pred: cat=wall_shelf filters={'leg_color': 'natural'} roles=fffffffffffffff
- `floating shelf with natural wood legs under 30 in ergonomic easy to assemble`  
  gold: cat=wall_shelf filters={'width': 'under_30in', 'leg_material': 'wood', 'leg_color': 'natural'} roles=ccrffffffrrrr  
  pred: cat=desk filters={'leg_material': 'wood', 'leg_color': 'natural', 'width': 'under_30in'} roles=fffffffffffff
- `furniture minimalist engineered wood easy to assemble like pottery barn`  
  gold: cat=None filters={'style': 'minimalist', 'material': 'engineered_wood'} roles=rfffrrrrrr  
  pred: cat=bed_frames filters={'material': 'engineered_wood', 'style': 'minimalist'} roles=cfffffffff
- `pantry cabinet glam style kitchen pantry office engineered wood deals for small apartment`  
  gold: cat=pantry_cabinet filters={'material': 'engineered_wood', 'style': 'glam', 'room': 'office'} roles=ccffccfffrrrr  
  pred: cat=pantry_cabinet filters={'style': 'glam'} roles=fffffffffrfff
- `accent cabinet 30 inch made in usa fast delivery`  
  gold: cat=accent_cabinet filters={'width': '30_to_48in'} roles=ccffrrrrr  
  pred: cat=accent_cabinet filters={'width': 'under_30in'} roles=fffffffff
- `memory foam mattress soft foldable full set`  
  gold: cat=memory_foam_mattress filters={'firmness': 'soft', 'feature': 'foldable', 'bed_size': 'full'} roles=cccfffr  
  pred: cat=sofas filters={'feature': 'foldable', 'room': 'bedroom'} roles=ffcffff
- `bed frame full bedroom in stock for small apartment`  
  gold: cat=bed_frames filters={'bed_size': 'full', 'room': 'bedroom'} roles=ccffrrrrr  
  pred: cat=bed_frames filters={'room': 'bedroom'} roles=fffffffff
- `l shaped desk curved pet friendly under 300 pine`  
  gold: cat=l_shaped_desk filters={'material': 'pine', 'shape': 'curved'} roles=cccfrrrrf  
  pred: cat=l_shaped_desk filters={'material': 'pine', 'shape': 'curved', 'width': 'under_30in'} roles=fffffffff
- `pub table seats 8 like pottery barn`  
  gold: cat=pub_table filters={'seat_count': 'seats_8'} roles=ccffrrr  
  pred: cat=desk filters={} roles=rffffff
