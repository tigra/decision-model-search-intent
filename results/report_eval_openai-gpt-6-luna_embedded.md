# Search intent parsing: `embedded@openai:gpt-6-luna` via /v1/systemone

1000 queries, one request per query (23.6 questions on average).

## Accuracy

| Metric | **tuned decoding** (group-max category, filter p≥0.95, masked) | tuned, unmasked filters | untuned (plain argmax) |
|---|---|---|---|
| Category exact node acc | 0.987 | 0.987 | 0.986 |
| Category L1 acc (given gold depth ≥ 1) | 0.998 | 0.998 | 0.997 |
| Category L2 acc (given gold depth ≥ 2) | 1.000 | 1.000 | 1.000 |
| Category L3 acc (given gold depth ≥ 3) | 0.992 | 0.992 | 0.992 |
| Category hierarchical F1 | 0.996 | 0.996 | 0.995 |
| 'No category' P / R (tp/fp/fn) | 0.96 / 1.00 (48/2/0) | 0.96 / 1.00 (48/2/0) | 0.94 / 1.00 (48/3/0) |
| Filters micro P / R / F1 | 0.929 / 0.857 / 0.891 | 0.925 / 0.857 / 0.890 | 0.814 / 0.963 / 0.882 |
| Filter set exact match | 0.726 | 0.721 | 0.687 |
| Word-role accuracy | 0.884 | 0.884 | 0.870 |
| Word-role macro F1 | 0.865 | 0.865 | 0.854 |
| Residual words P / R / F1 | 0.981 / 0.605 / 0.748 | 0.981 / 0.605 / 0.748 | 0.953 / 0.619 / 0.751 |
| Residual set exact match | 0.697 | 0.697 | 0.691 |
| **Full query exact match** | **0.512** | **0.508** | **0.467** |

### Per-attribute filter scores (masked)

| attribute | support | P | R | F1 |
|---|---|---|---|---|
| feature | 220 | 0.981 | 0.927 | 0.953 |
| color | 211 | 0.877 | 0.943 | 0.909 |
| material | 185 | 1.000 | 0.541 | 0.702 |
| style | 182 | 1.000 | 0.989 | 0.994 |
| room | 147 | 0.855 | 0.884 | 0.870 |
| leg_color | 142 | 0.783 | 0.866 | 0.823 |
| leg_material | 132 | 0.934 | 0.864 | 0.898 |
| bed_size | 103 | 1.000 | 0.981 | 0.990 |
| width | 83 | 0.980 | 0.602 | 0.746 |
| shape | 66 | 1.000 | 0.848 | 0.918 |
| firmness | 36 | 1.000 | 0.944 | 0.971 |
| seat_count | 13 | 0.786 | 0.846 | 0.815 |

### Word roles (gold → predicted)

| gold \ pred | category | filter | residual |
|---|---|---|---|
| category | 1697 | 48 | 9 |
| filter | 51 | 2420 | 7 |
| residual | 60 | 471 | 813 |

### Top category confusions (gold → predicted)

- loveseat → sofas: 2
- curio_cabinet → cabinets: 2
- beds → kids_beds: 2
- chairs → dining_chair: 1
- tables → dining_tables: 1
- beds → None: 1
- desks → None: 1
- seating → sofas: 1
- desks → desk: 1
- seating → dining_chair: 1

## Latency (sequential, warm model)

| mean | p50 | p90 | p95 | p99 | max | queries/s | ms per question | mean input tokens |
|---|---|---|---|---|---|---|---|---|
| 0.468s | 0.439s | 0.611s | 0.816s | 1.157s | 2.530s | 2.14 | 19.8 | 5227 |

p50 latency by query length: 1-3 words: 0.433s (n=185), 4-6 words: 0.435s (n=505), 7-9 words: 0.446s (n=258), 10-12 words: 0.458s (n=47), 13-15 words: 0.455s (n=5)

Truncated queries (more than the word-question budget): 0

## 20 worst examples

- `latex mattress twin xl extra firm cooling heavy duty for tall people`  
  gold: cat=latex_mattress filters={'bed_size': 'twin_xl', 'firmness': 'extra_firm', 'feature': 'cooling'} roles=ccfffffrrrrr  
  pred: cat=latex_mattress filters={'bed_size': 'twin_xl', 'firmness': 'extra_firm', 'feature': 'cooling'} roles=fcffffffffff
- `furniture with acrylic legs vintage style silver legs for small space`  
  gold: cat=None filters={'leg_material': 'acrylic', 'style': 'vintage', 'leg_color': 'silver'} roles=rffffrffrrr  
  pred: cat=None filters={'leg_color': 'silver', 'style': 'vintage'} roles=cffffffffff
- `china cabinet with glass doors for small apartment luxury`  
  gold: cat=curio_cabinet filters={'feature': 'glass_doors'} roles=ccfffrrrr  
  pred: cat=cabinets filters={'feature': 'glass_doors'} roles=ccffcfffr
- `ladder shelf with walnut and chrome legs farmhouse style ikea`  
  gold: cat=ladder_shelf filters={'material': 'walnut', 'leg_material': 'chrome', 'style': 'farmhouse'} roles=ccrfrfffrr  
  pred: cat=ladder_shelf filters={'leg_color': 'silver', 'style': 'farmhouse'} roles=ccfffffffr
- `gaming chair living room brass legs tufted made in usa for studio`  
  gold: cat=gaming_chair filters={'room': 'living_room', 'leg_color': 'brass', 'feature': 'tufted'} roles=ccfffffrrrrr  
  pred: cat=gaming_chair filters={'leg_color': 'brass', 'feature': 'tufted', 'room': 'living_room'} roles=cccfffffffff
- `furniture with engineered wood and acrylic legs like west elm heavy duty`  
  gold: cat=None filters={'material': 'engineered_wood', 'leg_material': 'acrylic'} roles=rrffrffrrrrr  
  pred: cat=None filters={'leg_material': 'acrylic'} roles=cffffffrrrff
- `china cabinet with drawers traditional for small apartment`  
  gold: cat=curio_cabinet filters={'feature': 'with_drawers', 'style': 'traditional'} roles=ccfffrrr  
  pred: cat=cabinets filters={'style': 'traditional', 'feature': 'with_drawers'} roles=ccfcffff
- `two seater sofa 42 in under 500 dollars`  
  gold: cat=loveseat filters={'width': '30_to_48in'} roles=cccffrrr  
  pred: cat=sofas filters={'width': '30_to_48in', 'seat_count': 'seats_2'} roles=fccfffrr
- `used dining room chair`  
  gold: cat=chairs filters={'room': 'dining_room'} roles=rffc  
  pred: cat=dining_chair filters={} roles=rccc
- `seating with dark brown legs and orange color made of acacia`  
  gold: cat=seating filters={'material': 'acacia', 'leg_color': 'dark_brown', 'color': 'orange'} roles=crfffrffrrf  
  pred: cat=seating filters={'color': 'orange', 'leg_color': 'dark_brown'} roles=cffffffffff
- `ladder shelf for kids with drawers for tall people`  
  gold: cat=ladder_shelf filters={'room': 'kids_room', 'feature': 'with_drawers'} roles=ccffffrrr  
  pred: cat=ladder_shelf filters={'feature': 'with_drawers'} roles=ccfffcfff
- `beige furniture with wooden legs traditional style pet friendly sturdy`  
  gold: cat=None filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=frffffrrrr  
  pred: cat=None filters={'color': 'beige', 'leg_material': 'wood', 'leg_color': 'natural', 'style': 'traditional'} roles=fcfffffffr
- `dining table with white legs 58 inch for small apartment`  
  gold: cat=dining_tables filters={'leg_color': 'white', 'width': '48_to_72in'} roles=ccfffffrrr  
  pred: cat=dining_tables filters={'color': 'white', 'leg_color': 'white'} roles=ccffffffff
- `murphy bed tufted for small spaces heavy duty`  
  gold: cat=murphy_beds filters={'feature': 'tufted'} roles=ccfrrrrr  
  pred: cat=murphy_beds filters={'feature': 'tufted'} roles=ccffffff
- `latex mattress queen set easy to assemble cooling`  
  gold: cat=latex_mattress filters={'feature': 'cooling', 'bed_size': 'queen'} roles=ccfrrrrf  
  pred: cat=latex_mattress filters={'bed_size': 'queen', 'feature': 'cooling'} roles=fcfcffff
- `office furniture with walnut legs modern made in usa for studio`  
  gold: cat=desks filters={'leg_material': 'walnut', 'style': 'modern'} roles=ccffffrrrrr  
  pred: cat=desks filters={'leg_material': 'walnut', 'style': 'modern'} roles=ccfffffffff
- `bed linen for my cat`  
  gold: cat=beds filters={'material': 'linen'} roles=cfrrr  
  pred: cat=None filters={} roles=ccfrr
- `trestle table scandinavian style white legs under 30 inch 69 cm for small spaces ergonomic`  
  gold: cat=trestle_table filters={'width': 'under_30in', 'style': 'scandinavian', 'leg_color': 'white'} roles=ccfffffffffrrrr  
  pred: cat=trestle_table filters={'color': 'white', 'leg_color': 'white', 'style': 'scandinavian'} roles=ccffffffffffffr
- `seating for entryway reclining sofa`  
  gold: cat=seating filters={'room': 'entryway', 'feature': 'reclining'} roles=crffc  
  pred: cat=sofas filters={'room': 'entryway'} roles=cffcc
- `adjustable base easy to assemble under 300`  
  gold: cat=adjustable_bases filters={} roles=ccrrrrr  
  pred: cat=adjustable_bases filters={} roles=fcffffr
