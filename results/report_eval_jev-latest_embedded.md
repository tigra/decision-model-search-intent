# Search intent parsing: `embedded@jev:jev-latest` via /v1/systemone

1000 queries, one request per query (23.6 questions on average).

## Accuracy

| Metric | **tuned decoding** (group-max category, filter p≥0.95, masked) | tuned, unmasked filters | untuned (plain argmax) |
|---|---|---|---|
| Category exact node acc | 0.987 | 0.987 | 0.987 |
| Category L1 acc (given gold depth ≥ 1) | 0.996 | 0.996 | 0.996 |
| Category L2 acc (given gold depth ≥ 2) | 0.997 | 0.997 | 0.997 |
| Category L3 acc (given gold depth ≥ 3) | 0.994 | 0.994 | 0.994 |
| Category hierarchical F1 | 0.994 | 0.994 | 0.994 |
| 'No category' P / R (tp/fp/fn) | 0.98 / 1.00 (48/1/0) | 0.98 / 1.00 (48/1/0) | 0.98 / 1.00 (48/1/0) |
| Filters micro P / R / F1 | 0.963 / 0.882 / 0.921 | 0.963 / 0.882 / 0.921 | 0.819 / 0.980 / 0.892 |
| Filter set exact match | 0.800 | 0.799 | 0.701 |
| Word-role accuracy | 0.854 | 0.854 | 0.862 |
| Word-role macro F1 | 0.830 | 0.830 | 0.839 |
| Residual words P / R / F1 | 0.946 / 0.549 / 0.695 | 0.946 / 0.549 / 0.695 | 0.939 / 0.560 / 0.702 |
| Residual set exact match | 0.651 | 0.651 | 0.657 |
| **Full query exact match** | **0.535** | **0.535** | **0.452** |

### Per-attribute filter scores (masked)

| attribute | support | P | R | F1 |
|---|---|---|---|---|
| feature | 220 | 0.954 | 0.950 | 0.952 |
| color | 211 | 0.980 | 0.948 | 0.964 |
| material | 185 | 1.000 | 0.714 | 0.833 |
| style | 182 | 1.000 | 0.995 | 0.997 |
| room | 147 | 0.956 | 0.741 | 0.835 |
| leg_color | 142 | 1.000 | 0.894 | 0.944 |
| leg_material | 132 | 0.794 | 0.818 | 0.806 |
| bed_size | 103 | 1.000 | 0.981 | 0.990 |
| width | 83 | 1.000 | 0.819 | 0.901 |
| shape | 66 | 1.000 | 0.909 | 0.952 |
| firmness | 36 | 1.000 | 0.944 | 0.971 |
| seat_count | 13 | 0.750 | 0.923 | 0.828 |

### Word roles (gold → predicted)

| gold \ pred | category | filter | residual |
|---|---|---|---|
| category | 1725 | 29 | 0 |
| filter | 137 | 2299 | 42 |
| residual | 92 | 514 | 738 |

### Top category confusions (gold → predicted)

- beds → kids_beds: 2
- desks → desk: 2
- curio_cabinet → cabinets: 1
- sectional_sofa → console_tables: 1
- drafting_tables → tables: 1
- chairs → dining_chair: 1
- pantry_cabinet → cabinets: 1
- beds → None: 1
- seating → sofas: 1
- filing_cabinets → cabinets: 1
- seating → dining_chair: 1

## Latency (sequential, warm model)

| mean | p50 | p90 | p95 | p99 | max | queries/s | ms per question | mean input tokens |
|---|---|---|---|---|---|---|---|---|
| 0.344s | 0.332s | 0.389s | 0.414s | 0.514s | 0.790s | 2.91 | 14.6 | 5304 |

p50 latency by query length: 1-3 words: 0.332s (n=185), 4-6 words: 0.329s (n=505), 7-9 words: 0.334s (n=258), 10-12 words: 0.344s (n=47), 13-15 words: 0.342s (n=5)

Truncated queries (more than the word-question budget): 0

## 20 worst examples

- `latex mattress twin xl extra firm cooling heavy duty for tall people`  
  gold: cat=latex_mattress filters={'bed_size': 'twin_xl', 'firmness': 'extra_firm', 'feature': 'cooling'} roles=ccfffffrrrrr  
  pred: cat=latex_mattress filters={'bed_size': 'twin_xl', 'firmness': 'extra_firm', 'feature': 'cooling'} roles=fcffffffffff
- `trundle bed coastal for kids full for small apartment`  
  gold: cat=trundle_bed filters={'style': 'coastal', 'room': 'kids_room', 'bed_size': 'full'} roles=ccffffrrr  
  pred: cat=trundle_bed filters={'style': 'coastal', 'bed_size': 'full'} roles=ccfrrffff
- `file cabinet natural wood legs easy to assemble`  
  gold: cat=filing_cabinets filters={'leg_color': 'natural'} roles=ccfffrrr  
  pred: cat=filing_cabinets filters={'leg_material': 'wood'} roles=ccffcfff
- `console table natural wood legs under 30 inch made in usa`  
  gold: cat=console_tables filters={'leg_color': 'natural', 'width': 'under_30in'} roles=ccffffffrrr  
  pred: cat=console_tables filters={'leg_material': 'wood', 'leg_color': 'natural'} roles=ccffcffffff
- `ladder shelf for kids with drawers for tall people`  
  gold: cat=ladder_shelf filters={'room': 'kids_room', 'feature': 'with_drawers'} roles=ccffffrrr  
  pred: cat=ladder_shelf filters={'feature': 'with_drawers'} roles=cccrfffff
- `floating shelf with natural wood legs under 30 in ergonomic easy to assemble`  
  gold: cat=wall_shelf filters={'width': 'under_30in', 'leg_material': 'wood', 'leg_color': 'natural'} roles=ccrffffffrrrr  
  pred: cat=wall_shelf filters={'leg_material': 'wood', 'width': 'under_30in'} roles=ccfffffffffff
- `murphy bed tufted for small spaces heavy duty`  
  gold: cat=murphy_beds filters={'feature': 'tufted'} roles=ccfrrrrr  
  pred: cat=murphy_beds filters={} roles=ccffffff
- `trestle table scandinavian style white legs under 30 inch 69 cm for small spaces ergonomic`  
  gold: cat=trestle_table filters={'width': 'under_30in', 'style': 'scandinavian', 'leg_color': 'white'} roles=ccfffffffffrrrr  
  pred: cat=trestle_table filters={'leg_color': 'white', 'style': 'scandinavian'} roles=ccfffcfffffffff
- `pantry cabinet glam style kitchen pantry office engineered wood deals for small apartment`  
  gold: cat=pantry_cabinet filters={'material': 'engineered_wood', 'style': 'glam', 'room': 'office'} roles=ccffccfffrrrr  
  pred: cat=pantry_cabinet filters={'material': 'engineered_wood', 'style': 'glam'} roles=ccfcfcfffrfff
- `seats stacking dining room brass legs sturdy`  
  gold: cat=seating filters={'feature': 'stackable', 'leg_color': 'brass', 'room': 'dining_room'} roles=cfffffr  
  pred: cat=dining_chair filters={'leg_material': 'metal', 'leg_color': 'brass', 'feature': 'stackable', 'room': 'dining_room'} roles=cccffff
- `sofa for kids for studio`  
  gold: cat=sofas filters={'room': 'kids_room'} roles=cffrr  
  pred: cat=sofas filters={} roles=crrff
- `furniture with acrylic legs vintage style silver legs for small space`  
  gold: cat=None filters={'leg_material': 'acrylic', 'style': 'vintage', 'leg_color': 'silver'} roles=rffffrffrrr  
  pred: cat=None filters={'leg_material': 'acrylic', 'leg_color': 'silver', 'style': 'vintage'} roles=cffffffffff
- `pub table with boucle seats 3 wood legs for small spaces`  
  gold: cat=pub_table filters={'material': 'boucle', 'leg_material': 'wood', 'seat_count': 'seats_3'} roles=ccffffffrrr  
  pred: cat=pub_table filters={'leg_material': 'wood'} roles=ccfffffffff
- `china cabinet with glass doors for small apartment luxury`  
  gold: cat=curio_cabinet filters={'feature': 'glass_doors'} roles=ccfffrrrr  
  pred: cat=curio_cabinet filters={'feature': 'glass_doors'} roles=ccffcffff
- `wall shelf with acrylic legs and natural wood legs for tall people under 500 dollars`  
  gold: cat=wall_shelf filters={'leg_material': 'acrylic', 'leg_color': 'natural'} roles=ccfffffffrrrrrr  
  pred: cat=wall_shelf filters={} roles=ccfffffffffrfrr
- `console table hairpin legs 111 inch set`  
  gold: cat=console_tables filters={'leg_material': 'hairpin', 'width': 'over_90in'} roles=ccffffr  
  pred: cat=console_tables filters={} roles=ccccffc
- `seating with dark brown legs and orange color made of acacia`  
  gold: cat=seating filters={'material': 'acacia', 'leg_color': 'dark_brown', 'color': 'orange'} roles=crfffrffrrf  
  pred: cat=seating filters={'color': 'orange', 'leg_color': 'dark_brown'} roles=cffffffffff
- `bed for kids twin contemporary`  
  gold: cat=beds filters={'bed_size': 'twin', 'style': 'contemporary', 'room': 'kids_room'} roles=cffff  
  pred: cat=kids_beds filters={'style': 'contemporary', 'bed_size': 'twin'} roles=crrff
- `bunk bed with drawrs for small spaces full`  
  gold: cat=bunk_bed filters={'bed_size': 'full', 'feature': 'with_drawers'} roles=ccffrrrf  
  pred: cat=bunk_bed filters={'feature': 'with_drawers'} roles=ccfcffff
- `beige furniture with wooden legs traditional style pet friendly sturdy`  
  gold: cat=None filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=frffffrrrr  
  pred: cat=None filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=fcffffffff
