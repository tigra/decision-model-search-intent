# Search intent parsing: `embedded@ollama:tev1` via /v1/systemone

1000 queries, one request per query (23.6 questions on average).

## Accuracy

| Metric | **tuned decoding** (group-max category, filter p≥0.95, masked) | tuned, unmasked filters | untuned (plain argmax) |
|---|---|---|---|
| Category exact node acc | 0.942 | 0.942 | 0.947 |
| Category L1 acc (given gold depth ≥ 1) | 0.987 | 0.987 | 0.977 |
| Category L2 acc (given gold depth ≥ 2) | 0.990 | 0.990 | 0.987 |
| Category L3 acc (given gold depth ≥ 3) | 0.978 | 0.978 | 0.976 |
| Category hierarchical F1 | 0.958 | 0.958 | 0.963 |
| 'No category' P / R (tp/fp/fn) | 0.89 / 0.50 (24/3/24) | 0.89 / 0.50 (24/3/24) | 0.72 / 0.79 (38/15/10) |
| Filters micro P / R / F1 | 0.786 / 0.953 / 0.862 | 0.766 / 0.956 / 0.851 | 0.713 / 0.973 / 0.823 |
| Filter set exact match | 0.612 | 0.580 | 0.534 |
| Word-role accuracy | 0.684 | 0.684 | 0.795 |
| Word-role macro F1 | 0.687 | 0.687 | 0.793 |
| Residual words P / R / F1 | 0.783 / 0.708 / 0.744 | 0.783 / 0.708 / 0.744 | 0.598 / 0.842 / 0.699 |
| Residual set exact match | 0.549 | 0.549 | 0.440 |
| **Full query exact match** | **0.336** | **0.322** | **0.253** |

### Per-attribute filter scores (masked)

| attribute | support | P | R | F1 |
|---|---|---|---|---|
| feature | 220 | 0.817 | 0.995 | 0.898 |
| color | 211 | 0.832 | 0.986 | 0.902 |
| material | 185 | 0.808 | 0.957 | 0.876 |
| style | 182 | 1.000 | 0.995 | 0.997 |
| room | 147 | 0.516 | 0.980 | 0.676 |
| leg_color | 142 | 0.773 | 0.986 | 0.867 |
| leg_material | 132 | 0.629 | 0.939 | 0.754 |
| bed_size | 103 | 1.000 | 0.990 | 0.995 |
| width | 83 | 0.827 | 0.518 | 0.637 |
| shape | 66 | 1.000 | 0.955 | 0.977 |
| firmness | 36 | 1.000 | 1.000 | 1.000 |
| seat_count | 13 | 0.800 | 0.923 | 0.857 |

### Word roles (gold → predicted)

| gold \ pred | category | filter | residual |
|---|---|---|---|
| category | 1746 | 4 | 4 |
| filter | 1102 | 1116 | 260 |
| residual | 251 | 141 | 952 |

### Top category confusions (gold → predicted)

- None → storage: 11
- curio_cabinet → cabinets: 6
- None → tables: 6
- filing_cabinets → cabinets: 6
- None → mattresses: 2
- beds → None: 2
- seating → office_chair: 2
- seating → sofas: 2
- loveseat → sofas: 2
- beds → kids_beds: 2
- desks → desk: 2
- None → coffee_tables: 2
- None → seating: 2
- step_stool → dining_chair: 1
- tables → dining_tables: 1

## Latency (sequential, warm model)

| mean | p50 | p90 | p95 | p99 | max | queries/s | ms per question | mean input tokens |
|---|---|---|---|---|---|---|---|---|
| 13.075s | 12.961s | 14.143s | 15.585s | 16.958s | 17.440s | 0.08 | 554.6 | 19264 |

p50 latency by query length: 1-3 words: 12.660s (n=185), 4-6 words: 12.770s (n=505), 7-9 words: 13.356s (n=258), 10-12 words: 14.111s (n=47), 13-15 words: 15.259s (n=5)

Truncated queries (more than the word-question budget): 0

## 20 worst examples

- `furniture with acrylic legs vintage style silver legs for small space`  
  gold: cat=None filters={'leg_material': 'acrylic', 'style': 'vintage', 'leg_color': 'silver'} roles=rffffrffrrr  
  pred: cat=storage filters={'leg_material': 'acrylic', 'leg_color': 'silver', 'style': 'vintage', 'width': 'under_30in'} roles=crfcfffcrrr
- `cabinet with glass doors bohemian 46 in ergonomic`  
  gold: cat=cabinets filters={'feature': 'glass_doors', 'width': '30_to_48in', 'style': 'bohemian'} roles=cffffffr  
  pred: cat=cabinets filters={'material': 'glass', 'style': 'bohemian', 'feature': 'glass_doors'} roles=crccccrr
- `furniture with metal legs and white legs`  
  gold: cat=None filters={'leg_material': 'metal', 'leg_color': 'white'} roles=rffffff  
  pred: cat=desks filters={'color': 'white', 'leg_material': 'metal', 'leg_color': 'white'} roles=crfcrfc
- `beige furniture with wooden legs traditional style pet friendly sturdy`  
  gold: cat=None filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=frffffrrrr  
  pred: cat=storage filters={'color': 'beige', 'leg_material': 'wood', 'leg_color': 'natural', 'style': 'traditional'} roles=fcrffffffr
- `furniture mid century modern with wood legs`  
  gold: cat=None filters={'style': 'mid_century', 'leg_material': 'wood'} roles=rffffff  
  pred: cat=storage filters={'leg_material': 'wood', 'leg_color': 'natural', 'style': 'mid_century'} roles=ccccrff
- `bed tufted solid wood bedroom 2024`  
  gold: cat=beds filters={'feature': 'tufted', 'room': 'bedroom', 'material': 'solid_wood'} roles=cffffr  
  pred: cat=None filters={'material': 'solid_wood', 'leg_material': 'wood', 'feature': 'tufted', 'room': 'bedroom'} roles=cccccr
- `china cabinet living room 37 in`  
  gold: cat=curio_cabinet filters={'room': 'living_room', 'width': '30_to_48in'} roles=ccffff  
  pred: cat=cabinets filters={'room': 'living_room'} roles=cccccr
- `loveseat with chrome legs over 90 inch traditional style`  
  gold: cat=loveseat filters={'width': 'over_90in', 'style': 'traditional', 'leg_material': 'chrome'} roles=cffffffff  
  pred: cat=loveseat filters={'leg_material': 'chrome', 'leg_color': 'silver', 'style': 'traditional', 'width': 'over_90in'} roles=crfcrfccc
- `two seater sofa 42 in under 500 dollars`  
  gold: cat=loveseat filters={'width': '30_to_48in'} roles=cccffrrr  
  pred: cat=sofas filters={'seat_count': 'seats_2'} roles=ccccrfrr
- `l shaped desk curved pet friendly under 300 pine`  
  gold: cat=l_shaped_desk filters={'material': 'pine', 'shape': 'curved'} roles=cccfrrrrf  
  pred: cat=l_shaped_desk filters={'material': 'pine', 'width': 'under_30in'} roles=ccccfcfff
- `china cabinet with glass doors for small apartment luxury`  
  gold: cat=curio_cabinet filters={'feature': 'glass_doors'} roles=ccfffrrrr  
  pred: cat=cabinets filters={'material': 'glass', 'feature': 'glass_doors'} roles=cccccrrfr
- `tv stand with drawers over 99 inch handmade`  
  gold: cat=tv_stands filters={'width': 'over_90in', 'feature': 'with_drawers'} roles=ccfffffr  
  pred: cat=tv_stands filters={'width': 'over_90in', 'feature': 'with_drawers', 'room': 'living_room'} roles=cccccccc
- `kids bed with storage california king japandi on sale like pottery barn`  
  gold: cat=kids_beds filters={'feature': 'with_storage', 'bed_size': 'california_king', 'style': 'japandi'} roles=ccfffffrrrrr  
  pred: cat=kids_beds filters={'style': 'japandi', 'bed_size': 'california_king', 'feature': 'with_storage', 'room': 'kids_room'} roles=ccccrccrrrcr
- `dining table 66" with glass doors seats 2`  
  gold: cat=dining_tables filters={'width': '48_to_72in', 'feature': 'glass_doors', 'seat_count': 'seats_2'} roles=ccffffff  
  pred: cat=dining_tables filters={'material': 'glass', 'seat_count': 'seats_2', 'feature': 'glass_doors', 'room': 'dining_room'} roles=ccfccccf
- `storage furniture bedroom 89 inch on wheels`  
  gold: cat=storage filters={'room': 'bedroom', 'width': '72_to_90in', 'feature': 'wheels'} roles=ccfffff  
  pred: cat=None filters={'feature': 'wheels', 'room': 'bedroom'} roles=ccfccrc
- `china cabinet rustic 315 cm pink handmade 2025`  
  gold: cat=curio_cabinet filters={'style': 'rustic', 'width': 'over_90in', 'color': 'pink'} roles=ccffffrr  
  pred: cat=cabinets filters={'color': 'pink', 'style': 'rustic'} roles=cccccfcr
- `best seating with metal and black legs`  
  gold: cat=seating filters={'material': 'metal', 'leg_color': 'black'} roles=rcfffff  
  pred: cat=seating filters={'color': 'black', 'leg_material': 'metal', 'leg_color': 'black'} roles=ccrfrfc
- `seats stacking dining room brass legs sturdy`  
  gold: cat=seating filters={'feature': 'stackable', 'leg_color': 'brass', 'room': 'dining_room'} roles=cfffffr  
  pred: cat=dining_tables filters={'leg_color': 'brass', 'feature': 'stackable', 'room': 'dining_room'} roles=ccccfcc
- `dresser with drawers brass legs 114 cm under 500 dollars`  
  gold: cat=dressers filters={'leg_color': 'brass', 'width': '30_to_48in', 'feature': 'with_drawers'} roles=cffffffrrr  
  pred: cat=dressers filters={'leg_color': 'brass', 'feature': 'with_drawers'} roles=cccffccffr
- `furniture glam oak legs`  
  gold: cat=None filters={'style': 'glam', 'leg_material': 'oak'} roles=rfff  
  pred: cat=mattresses filters={} roles=ccff
