# Search intent parsing: `embedded@kev:kev-latest` via /v1/systemone

1000 queries, one request per query (23.6 questions on average).

## Accuracy

| Metric | **tuned decoding** (group-max category, filter p≥0.95, masked) | tuned, unmasked filters | untuned (plain argmax) |
|---|---|---|---|
| Category exact node acc | 0.874 | 0.874 | 0.871 |
| Category L1 acc (given gold depth ≥ 1) | 0.936 | 0.936 | 0.956 |
| Category L2 acc (given gold depth ≥ 2) | 0.958 | 0.958 | 0.962 |
| Category L3 acc (given gold depth ≥ 3) | 0.865 | 0.865 | 0.869 |
| Category hierarchical F1 | 0.921 | 0.921 | 0.924 |
| 'No category' P / R (tp/fp/fn) | 0.59 / 0.85 (41/29/7) | 0.59 / 0.85 (41/29/7) | 0.76 / 0.58 (28/9/20) |
| Filters micro P / R / F1 | 0.893 / 0.754 / 0.817 | 0.890 / 0.754 / 0.816 | 0.558 / 0.968 / 0.708 |
| Filter set exact match | 0.610 | 0.608 | 0.335 |
| Word-role accuracy | 0.581 | 0.581 | 0.822 |
| Word-role macro F1 | 0.584 | 0.584 | 0.809 |
| Residual words P / R / F1 | 0.790 / 0.638 / 0.706 | 0.790 / 0.638 / 0.706 | 0.678 / 0.749 / 0.711 |
| Residual set exact match | 0.535 | 0.535 | 0.481 |
| **Full query exact match** | **0.309** | **0.308** | **0.138** |

### Per-attribute filter scores (masked)

| attribute | support | P | R | F1 |
|---|---|---|---|---|
| feature | 220 | 0.994 | 0.786 | 0.878 |
| color | 211 | 0.839 | 0.768 | 0.802 |
| material | 185 | 0.938 | 0.486 | 0.641 |
| style | 182 | 1.000 | 0.791 | 0.883 |
| room | 147 | 0.869 | 0.769 | 0.816 |
| leg_color | 142 | 0.824 | 0.887 | 0.854 |
| leg_material | 132 | 0.664 | 0.674 | 0.669 |
| bed_size | 103 | 1.000 | 0.922 | 0.960 |
| width | 83 | 0.951 | 0.699 | 0.806 |
| shape | 66 | 1.000 | 0.818 | 0.900 |
| firmness | 36 | 1.000 | 0.833 | 0.909 |
| seat_count | 13 | 0.600 | 0.923 | 0.727 |

### Word roles (gold → predicted)

| gold \ pred | category | filter | residual |
|---|---|---|---|
| category | 1666 | 11 | 77 |
| filter | 1614 | 713 | 151 |
| residual | 377 | 109 | 858 |

### Top category confusions (gold → predicted)

- seating → None: 15
- gaming_desk → tables: 14
- chesterfield → sofas: 9
- l_shaped_desk → tables: 8
- nesting_coffee_table → coffee_tables: 8
- desks → None: 6
- desk → tables: 4
- memory_foam_mattress → mattress: 4
- nightstands → side_tables: 3
- hybrid_mattress → mattress: 3
- platform_bed → bed_frames: 3
- seating → office_chair: 2
- step_stool → stools: 2
- sectional_sofa → sofas: 2
- tables → dining_tables: 2

## Latency (sequential, warm model)

| mean | p50 | p90 | p95 | p99 | max | queries/s | ms per question | mean input tokens |
|---|---|---|---|---|---|---|---|---|
| 6.709s | 6.622s | 7.164s | 7.278s | 7.519s | 7.941s | 0.15 | 284.6 | 3164 |

p50 latency by query length: 1-3 words: 6.384s (n=185), 4-6 words: 6.614s (n=505), 7-9 words: 7.052s (n=258), 10-12 words: 7.388s (n=47), 13-15 words: 7.725s (n=5)

Truncated queries (more than the word-question budget): 0

## 20 worst examples

- `beige furniture with wooden legs traditional style pet friendly sturdy`  
  gold: cat=None filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=frffffrrrr  
  pred: cat=None filters={'leg_material': 'wood', 'leg_color': 'natural', 'style': 'traditional'} roles=ccrccccccc
- `trestle table scandinavian style white legs under 30 inch 69 cm for small spaces ergonomic`  
  gold: cat=trestle_table filters={'width': 'under_30in', 'style': 'scandinavian', 'leg_color': 'white'} roles=ccfffffffffrrrr  
  pred: cat=trestle_table filters={'color': 'white', 'leg_color': 'white', 'style': 'scandinavian'} roles=ccccccccffcrccc
- `office furniture for kids with hairpin legs boucle`  
  gold: cat=desks filters={'room': 'kids_room', 'leg_material': 'hairpin', 'material': 'boucle'} roles=ccffffff  
  pred: cat=None filters={} roles=rcrcrccc
- `l shaped desk japandi beige square luxury for small spaces`  
  gold: cat=l_shaped_desk filters={'style': 'japandi', 'color': 'beige', 'shape': 'square'} roles=cccfffrrrr  
  pred: cat=tables filters={} roles=cccccccccc
- `drop leaf table brass legs walnut legs foldable in stock pet friendly`  
  gold: cat=drop_leaf_table filters={'leg_color': 'brass', 'leg_material': 'walnut', 'feature': 'foldable'} roles=cccfffffrrrr  
  pred: cat=drop_leaf_table filters={} roles=rcccccccrrcc
- `cheap step stool with natural wood legs and metal legs for office`  
  gold: cat=step_stool filters={'leg_color': 'natural', 'leg_material': 'metal', 'room': 'office'} roles=rccfffffffff  
  pred: cat=step_stool filters={'room': 'office'} roles=rccccccccccc
- `furniture with engineered wood and acrylic legs like west elm heavy duty`  
  gold: cat=None filters={'material': 'engineered_wood', 'leg_material': 'acrylic'} roles=rrffrffrrrrr  
  pred: cat=office_chair filters={'material': 'engineered_wood', 'leg_material': 'acrylic'} roles=cccccccrrrcc
- `cabinet with glass doors bohemian 46 in ergonomic`  
  gold: cat=cabinets filters={'feature': 'glass_doors', 'width': '30_to_48in', 'style': 'bohemian'} roles=cffffffr  
  pred: cat=curio_cabinet filters={'style': 'bohemian', 'feature': 'glass_doors'} roles=crccccrc
- `l shaped desk curved pet friendly under 300 pine`  
  gold: cat=l_shaped_desk filters={'material': 'pine', 'shape': 'curved'} roles=cccfrrrrf  
  pred: cat=tables filters={} roles=ccccccffc
- `vanity table with oak legs and natural wood legs for office`  
  gold: cat=vanity_tables filters={'room': 'office', 'leg_color': 'natural', 'leg_material': 'oak'} roles=ccfffffffff  
  pred: cat=vanity_tables filters={'room': 'office'} roles=ccrcccccccf
- `furniture square with acrylic legs charcoal easy to assemble reviews`  
  gold: cat=None filters={'shape': 'square', 'leg_material': 'acrylic', 'color': 'charcoal'} roles=rfffffrrrr  
  pred: cat=coffee_tables filters={'color': 'charcoal', 'leg_material': 'acrylic', 'shape': 'square'} roles=ccrccccrcr
- `seating with chrome legs art deco for studio`  
  gold: cat=seating filters={'leg_material': 'chrome', 'style': 'art_deco'} roles=cfffffrr  
  pred: cat=None filters={'style': 'art_deco'} roles=cfcccccc
- `wall shelf with acrylic legs and natural wood legs for tall people under 500 dollars`  
  gold: cat=wall_shelf filters={'leg_material': 'acrylic', 'leg_color': 'natural'} roles=ccfffffffrrrrrr  
  pred: cat=wall_shelf filters={'leg_material': 'acrylic'} roles=ccccccccfrcrcrr
- `gaming chair living room brass legs tufted made in usa for studio`  
  gold: cat=gaming_chair filters={'room': 'living_room', 'leg_color': 'brass', 'feature': 'tufted'} roles=ccfffffrrrrr  
  pred: cat=gaming_chair filters={'leg_material': 'metal', 'feature': 'tufted', 'room': 'living_room'} roles=cccccccrrcrc
- `gaming desk cream adjustable height metal`  
  gold: cat=gaming_desk filters={'color': 'cream', 'feature': 'adjustable_height', 'material': 'metal'} roles=ccffff  
  pred: cat=tables filters={'color': 'cream', 'leg_material': 'metal'} roles=cccccc
- `pub table white legs wood legs under $300`  
  gold: cat=pub_table filters={'leg_color': 'white', 'leg_material': 'wood'} roles=ccffffrr  
  pred: cat=pub_table filters={'color': 'white'} roles=rccccccr
- `step stool dark brown legs walnut legs living room like pottery barn`  
  gold: cat=step_stool filters={'leg_color': 'dark_brown', 'room': 'living_room', 'leg_material': 'walnut'} roles=ccfffffffrrr  
  pred: cat=step_stool filters={'leg_material': 'walnut', 'leg_color': 'dark_brown', 'room': 'living_room'} roles=cccccccccrfc
- `nesting coffee table with chrome legs dark brown legs mid century ergonomic`  
  gold: cat=nesting_coffee_table filters={'leg_material': 'chrome', 'leg_color': 'dark_brown', 'style': 'mid_century'} roles=cccffffffffr  
  pred: cat=coffee_tables filters={'style': 'mid_century'} roles=cccfffccfccf
- `lift top coffee table red brass legs acrylic legs`  
  gold: cat=lift_top_coffee_table filters={'leg_color': 'brass', 'leg_material': 'acrylic', 'color': 'red'} roles=ccccfffff  
  pred: cat=lift_top_coffee_table filters={} roles=rrccccfcc
- `office chair black metal legs brass legs for studio`  
  gold: cat=office_chair filters={'color': 'black', 'leg_material': 'metal', 'leg_color': 'brass'} roles=ccfffffrr  
  pred: cat=office_chair filters={'color': 'black'} roles=ccccccccc
