# Search intent parsing: `embedded@jev:jev-1.14.0` via /v1/systemone

1000 queries, one request per query (23.6 questions on average).

## Accuracy

| Metric | **tuned decoding** (group-max category, filter p≥0.95, masked) | tuned, unmasked filters | untuned (plain argmax) |
|---|---|---|---|
| Category exact node acc | 0.991 | 0.991 | 0.991 |
| Category L1 acc (given gold depth ≥ 1) | 0.999 | 0.999 | 0.999 |
| Category L2 acc (given gold depth ≥ 2) | 1.000 | 1.000 | 1.000 |
| Category L3 acc (given gold depth ≥ 3) | 0.998 | 0.998 | 0.998 |
| Category hierarchical F1 | 0.997 | 0.997 | 0.997 |
| 'No category' P / R (tp/fp/fn) | 0.98 / 1.00 (48/1/0) | 0.98 / 1.00 (48/1/0) | 0.98 / 1.00 (48/1/0) |
| Filters micro P / R / F1 | 0.952 / 0.882 / 0.916 | 0.952 / 0.882 / 0.916 | 0.802 / 0.975 / 0.880 |
| Filter set exact match | 0.783 | 0.783 | 0.671 |
| Word-role accuracy | 0.868 | 0.868 | 0.861 |
| Word-role macro F1 | 0.844 | 0.844 | 0.842 |
| Residual words P / R / F1 | 0.952 / 0.562 / 0.707 | 0.952 / 0.562 / 0.707 | 0.945 / 0.580 / 0.719 |
| Residual set exact match | 0.654 | 0.654 | 0.666 |
| **Full query exact match** | **0.519** | **0.519** | **0.447** |

### Per-attribute filter scores (masked)

| attribute | support | P | R | F1 |
|---|---|---|---|---|
| feature | 220 | 0.981 | 0.941 | 0.961 |
| color | 211 | 0.971 | 0.953 | 0.962 |
| material | 185 | 1.000 | 0.681 | 0.810 |
| style | 182 | 1.000 | 1.000 | 1.000 |
| room | 147 | 0.973 | 0.741 | 0.842 |
| leg_color | 142 | 0.874 | 0.930 | 0.901 |
| leg_material | 132 | 0.785 | 0.856 | 0.819 |
| bed_size | 103 | 1.000 | 0.981 | 0.990 |
| width | 83 | 1.000 | 0.735 | 0.847 |
| shape | 66 | 1.000 | 0.924 | 0.961 |
| firmness | 36 | 1.000 | 0.972 | 0.986 |
| seat_count | 13 | 0.750 | 0.923 | 0.828 |

### Word roles (gold → predicted)

| gold \ pred | category | filter | residual |
|---|---|---|---|
| category | 1719 | 35 | 0 |
| filter | 74 | 2366 | 38 |
| residual | 98 | 490 | 756 |

### Top category confusions (gold → predicted)

- beds → kids_beds: 2
- desks → desk: 2
- curio_cabinet → cabinets: 1
- chairs → dining_chair: 1
- beds → None: 1
- seating → sofas: 1
- seating → dining_chair: 1

## Latency (sequential, warm model)

| mean | p50 | p90 | p95 | p99 | max | queries/s | ms per question | mean input tokens |
|---|---|---|---|---|---|---|---|---|
| 0.271s | 0.262s | 0.298s | 0.321s | 0.556s | 0.746s | 3.69 | 11.5 | 5304 |

p50 latency by query length: 1-3 words: 0.260s (n=185), 4-6 words: 0.259s (n=505), 7-9 words: 0.266s (n=258), 10-12 words: 0.274s (n=47), 13-15 words: 0.283s (n=5)

Truncated queries (more than the word-question budget): 0

## 20 worst examples

- `latex mattress twin xl extra firm cooling heavy duty for tall people`  
  gold: cat=latex_mattress filters={'bed_size': 'twin_xl', 'firmness': 'extra_firm', 'feature': 'cooling'} roles=ccfffffrrrrr  
  pred: cat=latex_mattress filters={'bed_size': 'twin_xl', 'firmness': 'extra_firm', 'feature': 'cooling'} roles=fcffffffffff
- `console table natural wood legs under 30 inch made in usa`  
  gold: cat=console_tables filters={'leg_color': 'natural', 'width': 'under_30in'} roles=ccffffffrrr  
  pred: cat=console_tables filters={'leg_material': 'wood'} roles=ccfffffffff
- `ladder shelf for kids with drawers for tall people`  
  gold: cat=ladder_shelf filters={'room': 'kids_room', 'feature': 'with_drawers'} roles=ccffffrrr  
  pred: cat=ladder_shelf filters={'feature': 'with_drawers'} roles=ccfrcffff
- `gaming chair living room brass legs tufted made in usa for studio`  
  gold: cat=gaming_chair filters={'room': 'living_room', 'leg_color': 'brass', 'feature': 'tufted'} roles=ccfffffrrrrr  
  pred: cat=gaming_chair filters={'leg_material': 'metal', 'leg_color': 'brass', 'feature': 'tufted', 'room': 'living_room'} roles=ccffffffffff
- `murphy bed tufted for small spaces heavy duty`  
  gold: cat=murphy_beds filters={'feature': 'tufted'} roles=ccfrrrrr  
  pred: cat=murphy_beds filters={} roles=ccffffff
- `trestle table scandinavian style white legs under 30 inch 69 cm for small spaces ergonomic`  
  gold: cat=trestle_table filters={'width': 'under_30in', 'style': 'scandinavian', 'leg_color': 'white'} roles=ccfffffffffrrrr  
  pred: cat=trestle_table filters={'leg_color': 'white', 'style': 'scandinavian'} roles=ccfffcfffffffff
- `loveseat with acrylic legs under 30 inch two seater sofa 23 inches`  
  gold: cat=loveseat filters={'leg_material': 'acrylic', 'width': 'under_30in'} roles=cffffffcccrr  
  pred: cat=loveseat filters={'leg_material': 'acrylic', 'seat_count': 'seats_2'} roles=cffffffffcff
- `accent cabinet entryway linen near me for small apartmet`  
  gold: cat=accent_cabinet filters={'room': 'entryway', 'material': 'linen'} roles=ccffrrrrr  
  pred: cat=accent_cabinet filters={'room': 'entryway'} roles=ccfcrrfff
- `sofa for kids for studio`  
  gold: cat=sofas filters={'room': 'kids_room'} roles=cffrr  
  pred: cat=sofas filters={} roles=crrff
- `furniture with acrylic legs vintage style silver legs for small space`  
  gold: cat=None filters={'leg_material': 'acrylic', 'style': 'vintage', 'leg_color': 'silver'} roles=rffffrffrrr  
  pred: cat=None filters={'leg_material': 'acrylic', 'leg_color': 'silver', 'style': 'vintage'} roles=cffffffffff
- `latex mattress fast delivery for small spaces`  
  gold: cat=latex_mattress filters={} roles=ccrrrrr  
  pred: cat=latex_mattress filters={} roles=fcrffff
- `pub table with boucle seats 3 wood legs for small spaces`  
  gold: cat=pub_table filters={'material': 'boucle', 'leg_material': 'wood', 'seat_count': 'seats_3'} roles=ccffffffrrr  
  pred: cat=pub_table filters={'leg_material': 'wood'} roles=ccfffffffff
- `l shaped desk curved pet friendly under 300 pine`  
  gold: cat=l_shaped_desk filters={'material': 'pine', 'shape': 'curved'} roles=cccfrrrrf  
  pred: cat=l_shaped_desk filters={'material': 'pine', 'shape': 'curved'} roles=fccffffff
- `ladder shelf with walnut and chrome legs farmhouse style ikea`  
  gold: cat=ladder_shelf filters={'material': 'walnut', 'leg_material': 'chrome', 'style': 'farmhouse'} roles=ccrfrfffrr  
  pred: cat=ladder_shelf filters={'color': 'brown', 'material': 'walnut', 'leg_material': 'chrome', 'leg_color': 'silver', 'style': 'farmhouse'} roles=ccfffffffr
- `trundle bed coastal for kids full for small apartment`  
  gold: cat=trundle_bed filters={'style': 'coastal', 'room': 'kids_room', 'bed_size': 'full'} roles=ccffffrrr  
  pred: cat=trundle_bed filters={'style': 'coastal', 'bed_size': 'full'} roles=ccffrffff
- `file cabinet natural wood legs easy to assemble`  
  gold: cat=filing_cabinets filters={'leg_color': 'natural'} roles=ccfffrrr  
  pred: cat=filing_cabinets filters={'leg_material': 'wood', 'leg_color': 'natural'} roles=ccffcfff
- `seating with dark brown legs and orange color made of acacia`  
  gold: cat=seating filters={'material': 'acacia', 'leg_color': 'dark_brown', 'color': 'orange'} roles=crfffrffrrf  
  pred: cat=seating filters={'color': 'orange', 'leg_color': 'dark_brown'} roles=cffffffffff
- `bunk bed with drawrs for small spaces full`  
  gold: cat=bunk_bed filters={'bed_size': 'full', 'feature': 'with_drawers'} roles=ccffrrrf  
  pred: cat=bunk_bed filters={'feature': 'with_drawers'} roles=cccfffff
- `beige furniture with wooden legs traditional style pet friendly sturdy`  
  gold: cat=None filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=frffffrrrr  
  pred: cat=None filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=fcffffffff
- `floating shelf with natural wood legs under 30 in ergonomic easy to assemble`  
  gold: cat=wall_shelf filters={'width': 'under_30in', 'leg_material': 'wood', 'leg_color': 'natural'} roles=ccrffffffrrrr  
  pred: cat=wall_shelf filters={'leg_material': 'wood', 'leg_color': 'natural', 'width': 'under_30in'} roles=ccfffffffffff
