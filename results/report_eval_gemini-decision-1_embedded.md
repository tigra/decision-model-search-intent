# Search intent parsing: `embedded@gemini:gemini-decision-1` via /v1/systemone

1000 queries, one request per query (23.6 questions on average).

## Accuracy

| Metric | **tuned decoding** (group-max category, filter p≥0.95, masked) | tuned, unmasked filters | untuned (plain argmax) |
|---|---|---|---|
| Category exact node acc | 0.981 | 0.981 | 0.984 |
| Category L1 acc (given gold depth ≥ 1) | 0.998 | 0.998 | 0.998 |
| Category L2 acc (given gold depth ≥ 2) | 0.999 | 0.999 | 0.999 |
| Category L3 acc (given gold depth ≥ 3) | 0.992 | 0.992 | 0.992 |
| Category hierarchical F1 | 0.989 | 0.989 | 0.992 |
| 'No category' P / R (tp/fp/fn) | 0.98 / 0.88 (42/1/6) | 0.98 / 0.88 (42/1/6) | 0.98 / 0.94 (45/1/3) |
| Filters micro P / R / F1 | 0.913 / 0.946 / 0.929 | 0.912 / 0.946 / 0.929 | 0.873 / 0.975 / 0.921 |
| Filter set exact match | 0.809 | 0.807 | 0.779 |
| Word-role accuracy | 0.910 | 0.910 | 0.880 |
| Word-role macro F1 | 0.904 | 0.904 | 0.876 |
| Residual words P / R / F1 | 0.949 / 0.778 / 0.855 | 0.949 / 0.778 / 0.855 | 0.935 / 0.783 / 0.852 |
| Residual set exact match | 0.771 | 0.771 | 0.769 |
| **Full query exact match** | **0.622** | **0.621** | **0.593** |

### Per-attribute filter scores (masked)

| attribute | support | P | R | F1 |
|---|---|---|---|---|
| feature | 220 | 0.911 | 0.977 | 0.943 |
| color | 211 | 0.977 | 0.995 | 0.986 |
| material | 185 | 0.938 | 0.811 | 0.870 |
| style | 182 | 0.984 | 1.000 | 0.992 |
| room | 147 | 0.728 | 0.966 | 0.830 |
| leg_color | 142 | 0.870 | 0.944 | 0.905 |
| leg_material | 132 | 0.846 | 0.871 | 0.858 |
| bed_size | 103 | 1.000 | 1.000 | 1.000 |
| width | 83 | 0.986 | 0.880 | 0.930 |
| shape | 66 | 1.000 | 0.985 | 0.992 |
| firmness | 36 | 1.000 | 1.000 | 1.000 |
| seat_count | 13 | 0.812 | 1.000 | 0.897 |

### Word roles (gold → predicted)

| gold \ pred | category | filter | residual |
|---|---|---|---|
| category | 1607 | 136 | 11 |
| filter | 9 | 2424 | 45 |
| residual | 55 | 244 | 1045 |

### Top category confusions (gold → predicted)

- None → seating: 3
- seating → office_chair: 2
- loveseat → sofas: 2
- None → tables: 2
- desks → desk: 2
- curio_cabinet → accent_cabinet: 1
- sleigh_bed → sleeper_sofa: 1
- chairs → dining_chair: 1
- beds → None: 1
- None → ottomans: 1
- seating → sofas: 1
- beds → kids_beds: 1
- seating → dining_chair: 1

## Latency (sequential, warm model)

| mean | p50 | p90 | p95 | p99 | max | queries/s | ms per question | mean input tokens |
|---|---|---|---|---|---|---|---|---|
| 0.479s | 0.440s | 0.635s | 0.718s | 1.048s | 1.981s | 2.09 | 20.3 | 3899 |

p50 latency by query length: 1-3 words: 0.431s (n=185), 4-6 words: 0.430s (n=505), 7-9 words: 0.456s (n=258), 10-12 words: 0.463s (n=47), 13-15 words: 0.480s (n=5)

Truncated queries (more than the word-question budget): 0

## 20 worst examples

- `furniture with hexagonal top and chrome legs`  
  gold: cat=None filters={'leg_material': 'chrome', 'shape': 'hexagonal'} roles=rrffrff  
  pred: cat=tables filters={'leg_material': 'chrome', 'leg_color': 'silver', 'shape': 'hexagonal'} roles=cffcfff
- `beige furniture with wooden legs traditional style pet friendly sturdy`  
  gold: cat=None filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=frffffrrrr  
  pred: cat=seating filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=fcfffffffr
- `two seater sofa coastal style`  
  gold: cat=loveseat filters={'style': 'coastal'} roles=cccfr  
  pred: cat=sofas filters={'style': 'coastal', 'seat_count': 'seats_2'} roles=ffcff
- `furniture with acrylic legs vintage style silver legs for small space`  
  gold: cat=None filters={'leg_material': 'acrylic', 'style': 'vintage', 'leg_color': 'silver'} roles=rffffrffrrr  
  pred: cat=None filters={'leg_material': 'acrylic', 'leg_color': 'silver', 'style': 'vintage'} roles=cffffffffff
- `ladder shelf with walnut and chrome legs farmhouse style ikea`  
  gold: cat=ladder_shelf filters={'material': 'walnut', 'leg_material': 'chrome', 'style': 'farmhouse'} roles=ccrfrfffrr  
  pred: cat=ladder_shelf filters={'leg_material': 'chrome', 'leg_color': 'silver', 'style': 'farmhouse'} roles=ccfffffffr
- `seating with dark brown legs and orange color made of acacia`  
  gold: cat=seating filters={'material': 'acacia', 'leg_color': 'dark_brown', 'color': 'orange'} roles=crfffrffrrf  
  pred: cat=seating filters={'color': 'orange', 'leg_color': 'dark_brown'} roles=cffffffffff
- `latex mattress queen set easy to assemble cooling`  
  gold: cat=latex_mattress filters={'feature': 'cooling', 'bed_size': 'queen'} roles=ccfrrrrf  
  pred: cat=latex_mattress filters={'bed_size': 'queen', 'feature': 'cooling'} roles=fcfcffff
- `loveseat with acrylic legs under 30 inch two seater sofa 23 inches`  
  gold: cat=loveseat filters={'leg_material': 'acrylic', 'width': 'under_30in'} roles=cffffffcccrr  
  pred: cat=loveseat filters={'leg_material': 'acrylic', 'width': 'under_30in', 'seat_count': 'seats_2'} roles=cffffffffcff
- `office furniture for kids with hairpin legs boucle`  
  gold: cat=desks filters={'room': 'kids_room', 'leg_material': 'hairpin', 'material': 'boucle'} roles=ccffffff  
  pred: cat=desks filters={'material': 'boucle', 'leg_material': 'hairpin', 'style': 'mid_century'} roles=fcrrffff
- `seats boucle white legs office near me`  
  gold: cat=seating filters={'leg_color': 'white', 'material': 'boucle', 'room': 'office'} roles=cffffrr  
  pred: cat=office_chair filters={'color': 'white', 'material': 'boucle', 'room': 'office'} roles=cffffrr
- `drop leaf table brass legs walnut legs foldable in stock pet friendly`  
  gold: cat=drop_leaf_table filters={'leg_color': 'brass', 'leg_material': 'walnut', 'feature': 'foldable'} roles=cccfffffrrrr  
  pred: cat=drop_leaf_table filters={'feature': 'foldable'} roles=cccfffffrrff
- `pub table with boucle seats 3 wood legs for small spaces`  
  gold: cat=pub_table filters={'material': 'boucle', 'leg_material': 'wood', 'seat_count': 'seats_3'} roles=ccffffffrrr  
  pred: cat=pub_table filters={'leg_material': 'wood', 'seat_count': 'seats_3'} roles=ccfffffffff
- `two seater sofa 42 in under 500 dollars`  
  gold: cat=loveseat filters={'width': '30_to_48in'} roles=cccffrrr  
  pred: cat=sofas filters={'width': '30_to_48in', 'seat_count': 'seats_2'} roles=fccffrrr
- `furniture with chrome legs glam round`  
  gold: cat=None filters={'leg_material': 'chrome', 'style': 'glam', 'shape': 'round'} roles=rfffff  
  pred: cat=tables filters={'leg_material': 'chrome', 'leg_color': 'silver', 'style': 'glam', 'shape': 'round'} roles=cfffff
- `office chair with marble top for kids pink`  
  gold: cat=office_chair filters={'material': 'marble', 'room': 'kids_room', 'color': 'pink'} roles=ccffffff  
  pred: cat=office_chair filters={'color': 'pink'} roles=ccffcrff
- `office furniture solid wood glam desk`  
  gold: cat=desks filters={'material': 'solid_wood', 'style': 'glam'} roles=ccfffc  
  pred: cat=desk filters={'material': 'solid_wood', 'style': 'glam', 'room': 'office'} roles=fcfffc
- `storage furniture under 30 inch 74 cm affordable`  
  gold: cat=storage filters={'width': 'under_30in'} roles=ccfffrrr  
  pred: cat=storage filters={'width': 'under_30in', 'feature': 'with_storage'} roles=fcfffffr
- `floating shelf with natural wood legs under 30 in ergonomic easy to assemble`  
  gold: cat=wall_shelf filters={'width': 'under_30in', 'leg_material': 'wood', 'leg_color': 'natural'} roles=ccrffffffrrrr  
  pred: cat=wall_shelf filters={'leg_material': 'wood', 'leg_color': 'natural', 'width': 'under_30in'} roles=ccffffffffrff
- `best computer desk for kids with acrylic legs metal`  
  gold: cat=computer_desk filters={'leg_material': 'acrylic', 'room': 'kids_room', 'material': 'metal'} roles=rccffffff  
  pred: cat=computer_desk filters={'room': 'kids_room'} roles=rccrrffff
- `storage furniture 73 inch pet friendly`  
  gold: cat=storage filters={'width': '72_to_90in'} roles=ccffrr  
  pred: cat=storage filters={'width': '72_to_90in', 'feature': 'with_storage'} roles=fcffff
