# Search intent parsing: `embedded@ollama:tev1:0.8b` via /v1/systemone

1000 queries, one request per query (23.6 questions on average).

## Accuracy

| Metric | **tuned decoding** (group-max category, filter p≥0.95, masked) | tuned, unmasked filters | untuned (plain argmax) |
|---|---|---|---|
| Category exact node acc | 0.727 | 0.727 | 0.728 |
| Category L1 acc (given gold depth ≥ 1) | 0.809 | 0.809 | 0.809 |
| Category L2 acc (given gold depth ≥ 2) | 0.807 | 0.807 | 0.807 |
| Category L3 acc (given gold depth ≥ 3) | 0.796 | 0.796 | 0.796 |
| Category hierarchical F1 | 0.759 | 0.759 | 0.760 |
| 'No category' P / R (tp/fp/fn) | 1.00 / 0.00 (0/0/48) | 1.00 / 0.00 (0/0/48) | 0.33 / 0.02 (1/2/47) |
| Filters micro P / R / F1 | 0.644 / 0.666 / 0.655 | 0.618 / 0.691 / 0.653 | 0.472 / 0.810 / 0.596 |
| Filter set exact match | 0.326 | 0.307 | 0.197 |
| Word-role accuracy | 0.316 | 0.316 | 0.467 |
| Word-role macro F1 | 0.161 | 0.161 | 0.372 |
| Residual words P / R / F1 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.253 / 0.030 / 0.053 |
| Residual set exact match | 0.373 | 0.373 | 0.347 |
| **Full query exact match** | **0.090** | **0.083** | **0.056** |

### Per-attribute filter scores (masked)

| attribute | support | P | R | F1 |
|---|---|---|---|---|
| feature | 220 | 0.574 | 0.864 | 0.690 |
| color | 211 | 0.750 | 0.498 | 0.598 |
| material | 185 | 0.296 | 0.341 | 0.317 |
| style | 182 | 0.896 | 0.808 | 0.850 |
| room | 147 | 0.560 | 0.925 | 0.697 |
| leg_color | 142 | 0.887 | 0.831 | 0.858 |
| leg_material | 132 | 0.601 | 0.856 | 0.706 |
| bed_size | 103 | 0.774 | 0.398 | 0.526 |
| width | 83 | 0.808 | 0.253 | 0.385 |
| shape | 66 | 0.965 | 0.833 | 0.894 |
| firmness | 36 | 0.950 | 0.528 | 0.679 |
| seat_count | 13 | 0.833 | 0.385 | 0.526 |

### Word roles (gold → predicted)

| gold \ pred | category | filter | residual |
|---|---|---|---|
| category | 1754 | 0 | 0 |
| filter | 2471 | 6 | 1 |
| residual | 1341 | 3 | 0 |

### Top category confusions (gold → predicted)

- None → coffee_tables: 41
- cabinets → filing_cabinets: 13
- nightstands → standing_desk: 12
- step_stool → stools: 10
- dining_chair → bunk_bed: 8
- bookcases → l_shaped_desk: 8
- accent_cabinet → adjustable_bases: 8
- bed_frames → coffee_tables: 7
- dressers → desk: 7
- seating → coffee_tables: 6
- drafting_tables → mattress: 6
- accent_chair → trestle_table: 5
- platform_bed → sleeper_sofa: 5
- bean_bags → tables: 5
- ladder_shelf → writing_desk: 5

## Latency (sequential, warm model)

| mean | p50 | p90 | p95 | p99 | max | queries/s | ms per question | mean input tokens |
|---|---|---|---|---|---|---|---|---|
| 2.200s | 2.170s | 2.337s | 2.425s | 2.543s | 2.675s | 0.45 | 93.3 | 19264 |

p50 latency by query length: 1-3 words: 2.098s (n=185), 4-6 words: 2.162s (n=505), 7-9 words: 2.277s (n=258), 10-12 words: 2.426s (n=47), 13-15 words: 2.588s (n=5)

Truncated queries (more than the word-question budget): 0

## 20 worst examples

- `wall shelf with acrylic legs and natural wood legs for tall people under 500 dollars`  
  gold: cat=wall_shelf filters={'leg_material': 'acrylic', 'leg_color': 'natural'} roles=ccfffffffrrrrrr  
  pred: cat=hybrid_mattress filters={} roles=ccccccccccccccc
- `furniture minimalist engineered wood easy to assemble like pottery barn`  
  gold: cat=None filters={'style': 'minimalist', 'material': 'engineered_wood'} roles=rfffrrrrrr  
  pred: cat=coffee_tables filters={'material': 'solid_wood', 'leg_material': 'wood'} roles=cccccccccc
- `furniture with engineered wood and acrylic legs like west elm heavy duty`  
  gold: cat=None filters={'material': 'engineered_wood', 'leg_material': 'acrylic'} roles=rrffrffrrrrr  
  pred: cat=desk filters={'material': 'fabric', 'leg_material': 'acrylic'} roles=cccccccccccc
- `cabinet for my cat with natural wood legs like pottery barn in home office`  
  gold: cat=cabinets filters={'room': 'office', 'leg_color': 'natural'} roles=crrrffffrrrfff  
  pred: cat=cabinets filters={'material': 'leather', 'leg_material': 'wood', 'leg_color': 'natural', 'feature': 'reclining', 'room': 'office'} roles=cccccccccccccc
- `office furniture white farmhouse desk ikea under 300 dark brown legs`  
  gold: cat=desks filters={'leg_color': 'dark_brown', 'color': 'white', 'style': 'farmhouse'} roles=ccffcrrrfff  
  pred: cat=desk filters={'material': 'leather', 'leg_color': 'dark_brown', 'style': 'farmhouse', 'shape': 'oval', 'feature': 'reclining', 'room': 'office'} roles=ccccccccccc
- `furniture with natural wood legs oak legs hexagonal deals for tall people`  
  gold: cat=None filters={'leg_color': 'natural', 'leg_material': 'oak', 'shape': 'hexagonal'} roles=rfffffffrrrr  
  pred: cat=coffee_tables filters={'leg_material': 'oak', 'leg_color': 'natural', 'style': 'scandinavian', 'shape': 'hexagonal'} roles=cccccccccccc
- `furniture with acrylic legs vintage style silver legs for small space`  
  gold: cat=None filters={'leg_material': 'acrylic', 'style': 'vintage', 'leg_color': 'silver'} roles=rffffrffrrr  
  pred: cat=coffee_tables filters={'material': 'fabric', 'leg_material': 'acrylic', 'leg_color': 'silver', 'style': 'vintage'} roles=ccccccccccc
- `office chair rustic acacia natural wood legs like pottery barn`  
  gold: cat=office_chair filters={'leg_color': 'natural', 'style': 'rustic', 'material': 'acacia'} roles=ccfffffrrr  
  pred: cat=office_chair filters={'material': 'leather', 'leg_material': 'wood', 'style': 'rustic', 'feature': 'reclining', 'room': 'office'} roles=cccccccccc
- `seating with dark brown legs and orange color made of acacia`  
  gold: cat=seating filters={'material': 'acacia', 'leg_color': 'dark_brown', 'color': 'orange'} roles=crfffrffrrf  
  pred: cat=coffee_tables filters={'leg_color': 'dark_brown'} roles=ccccccccccc
- `floating shelf with natural wood legs under 30 in ergonomic easy to assemble`  
  gold: cat=wall_shelf filters={'width': 'under_30in', 'leg_material': 'wood', 'leg_color': 'natural'} roles=ccrffffffrrrr  
  pred: cat=coffee_tables filters={'material': 'leather', 'leg_material': 'wood', 'leg_color': 'natural', 'width': 'under_30in'} roles=ccccccccccccc
- `office furniture with walnut legs modern made in usa for studio`  
  gold: cat=desks filters={'leg_material': 'walnut', 'style': 'modern'} roles=ccffffrrrrr  
  pred: cat=coffee_tables filters={'material': 'velvet', 'leg_material': 'walnut', 'style': 'modern', 'feature': 'reclining', 'room': 'office'} roles=ccccccccccc
- `trestle table scandinavian style white legs under 30 inch 69 cm for small spaces ergonomic`  
  gold: cat=trestle_table filters={'width': 'under_30in', 'style': 'scandinavian', 'leg_color': 'white'} roles=ccfffffffffrrrr  
  pred: cat=trestle_table filters={'color': 'white', 'leg_color': 'white', 'style': 'scandinavian', 'width': 'under_30in'} roles=ccccccccccccccc
- `step stool dark brown legs walnut legs living room like pottery barn`  
  gold: cat=step_stool filters={'leg_color': 'dark_brown', 'room': 'living_room', 'leg_material': 'walnut'} roles=ccfffffffrrr  
  pred: cat=stools filters={'material': 'velvet', 'leg_material': 'walnut', 'leg_color': 'dark_brown', 'feature': 'foldable', 'room': 'living_room'} roles=cccccccccccc
- `office chair with wheels oak legs natural wood legs used`  
  gold: cat=office_chair filters={'leg_color': 'natural', 'leg_material': 'oak', 'feature': 'wheels'} roles=ccfffffffr  
  pred: cat=office_chair filters={'feature': 'reclining', 'room': 'office'} roles=cccccccccc
- `cheap step stool with natural wood legs and metal legs for office`  
  gold: cat=step_stool filters={'leg_color': 'natural', 'leg_material': 'metal', 'room': 'office'} roles=rccfffffffff  
  pred: cat=step_stool filters={'feature': 'reclining', 'room': 'office'} roles=cccccccccccc
- `cheap dining chair japandi with casters dark brown legs deals`  
  gold: cat=dining_chair filters={'feature': 'wheels', 'style': 'japandi', 'leg_color': 'dark_brown'} roles=rccffffffr  
  pred: cat=bunk_bed filters={'color': 'brown', 'material': 'leather', 'style': 'japandi', 'feature': 'wheels'} roles=cccccccccc
- `beige furniture with wooden legs traditional style pet friendly sturdy`  
  gold: cat=None filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=frffffrrrr  
  pred: cat=coffee_tables filters={'color': 'beige', 'material': 'leather', 'leg_material': 'wood', 'style': 'traditional'} roles=cccccccccc
- `furniture square with acrylic legs charcoal easy to assemble reviews`  
  gold: cat=None filters={'shape': 'square', 'leg_material': 'acrylic', 'color': 'charcoal'} roles=rfffffrrrr  
  pred: cat=coffee_tables filters={'color': 'charcoal', 'material': 'fabric', 'leg_material': 'acrylic', 'shape': 'square'} roles=cccccccccc
- `step stoo with silver legs and wooden legs minimalist style`  
  gold: cat=step_stool filters={'leg_color': 'silver', 'leg_material': 'wood', 'style': 'minimalist'} roles=ccfffffffr  
  pred: cat=coffee_tables filters={'material': 'leather', 'leg_material': 'wood'} roles=cccccccccc
- `sofa entryway white legs 67 in easy to assemble`  
  gold: cat=sofas filters={'room': 'entryway', 'leg_color': 'white', 'width': '48_to_72in'} roles=cfffffrrr  
  pred: cat=coffee_tables filters={'color': 'white', 'leg_color': 'white', 'feature': 'adjustable_height', 'room': 'entryway'} roles=ccccccccc
