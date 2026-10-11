# Search intent parsing: `embedded@kev:kev-latest` via /v1/systemone

1000 queries, one request per query (23.6 questions on average).

## Accuracy

| Metric | **tuned decoding** (group-max category, filter p≥0.95, masked) | tuned, unmasked filters | untuned (plain argmax) |
|---|---|---|---|
| Category exact node acc | 0.874 | 0.874 | 0.870 |
| Category L1 acc (given gold depth ≥ 1) | 0.935 | 0.935 | 0.955 |
| Category L2 acc (given gold depth ≥ 2) | 0.956 | 0.956 | 0.961 |
| Category L3 acc (given gold depth ≥ 3) | 0.863 | 0.863 | 0.869 |
| Category hierarchical F1 | 0.921 | 0.921 | 0.922 |
| 'No category' P / R (tp/fp/fn) | 0.59 / 0.88 (42/29/6) | 0.59 / 0.88 (42/29/6) | 0.75 / 0.56 (27/9/21) |
| Filters micro P / R / F1 | 0.890 / 0.754 / 0.817 | 0.887 / 0.754 / 0.815 | 0.558 / 0.968 / 0.708 |
| Filter set exact match | 0.608 | 0.605 | 0.335 |
| Word-role accuracy | 0.579 | 0.579 | 0.823 |
| Word-role macro F1 | 0.582 | 0.582 | 0.810 |
| Residual words P / R / F1 | 0.789 / 0.636 / 0.705 | 0.789 / 0.636 / 0.705 | 0.679 / 0.749 / 0.713 |
| Residual set exact match | 0.532 | 0.532 | 0.482 |
| **Full query exact match** | **0.307** | **0.305** | **0.142** |

### Per-attribute filter scores (masked)

| attribute | support | P | R | F1 |
|---|---|---|---|---|
| feature | 220 | 0.994 | 0.791 | 0.881 |
| color | 211 | 0.836 | 0.773 | 0.803 |
| material | 185 | 0.938 | 0.486 | 0.641 |
| style | 182 | 1.000 | 0.791 | 0.883 |
| room | 147 | 0.875 | 0.762 | 0.815 |
| leg_color | 142 | 0.817 | 0.880 | 0.847 |
| leg_material | 132 | 0.654 | 0.674 | 0.664 |
| bed_size | 103 | 1.000 | 0.922 | 0.960 |
| width | 83 | 0.951 | 0.699 | 0.806 |
| shape | 66 | 1.000 | 0.818 | 0.900 |
| firmness | 36 | 1.000 | 0.833 | 0.909 |
| seat_count | 13 | 0.600 | 0.923 | 0.727 |

### Word roles (gold → predicted)

| gold \ pred | category | filter | residual |
|---|---|---|---|
| category | 1668 | 10 | 76 |
| filter | 1622 | 704 | 152 |
| residual | 385 | 104 | 855 |

### Top category confusions (gold → predicted)

- seating → None: 14
- gaming_desk → tables: 14
- chesterfield → sofas: 9
- nesting_coffee_table → coffee_tables: 8
- l_shaped_desk → tables: 7
- desks → None: 6
- nightstands → side_tables: 4
- desk → tables: 4
- memory_foam_mattress → mattress: 4
- hybrid_mattress → mattress: 3
- platform_bed → bed_frames: 3
- seating → office_chair: 2
- step_stool → stools: 2
- sectional_sofa → sofas: 2
- tables → dining_tables: 2

## Latency (sequential, warm model)

| mean | p50 | p90 | p95 | p99 | max | queries/s | ms per question | mean input tokens |
|---|---|---|---|---|---|---|---|---|
| 11.682s | 11.596s | 12.878s | 13.207s | 14.202s | 45.210s | 0.09 | 495.5 | 3164 |

p50 latency by query length: 1-3 words: 11.116s (n=185), 4-6 words: 11.535s (n=505), 7-9 words: 12.323s (n=258), 10-12 words: 12.988s (n=47), 13-15 words: 13.394s (n=5)

Truncated queries (more than the word-question budget): 0

## 20 worst examples

- `beige furniture with wooden legs traditional style pet friendly sturdy`  
  gold: cat=None filters={'color': 'beige', 'leg_material': 'wood', 'style': 'traditional'} roles=frffffrrrr  
  pred: cat=None filters={'leg_material': 'wood', 'leg_color': 'natural', 'style': 'traditional'} roles=ccrccccccc
- `trestle table scandinavian style white legs under 30 inch 69 cm for small spaces ergonomic`  
  gold: cat=trestle_table filters={'width': 'under_30in', 'style': 'scandinavian', 'leg_color': 'white'} roles=ccfffffffffrrrr  
  pred: cat=trestle_table filters={'color': 'white', 'leg_color': 'white', 'style': 'scandinavian'} roles=ccccccccffcrccc
- `l shaped desk japandi beige square luxury for small spaces`  
  gold: cat=l_shaped_desk filters={'style': 'japandi', 'color': 'beige', 'shape': 'square'} roles=cccfffrrrr  
  pred: cat=tables filters={} roles=cccccccccc
- `drop leaf table brass legs walnut legs foldable in stock pet friendly`  
  gold: cat=drop_leaf_table filters={'leg_color': 'brass', 'leg_material': 'walnut', 'feature': 'foldable'} roles=cccfffffrrrr  
  pred: cat=drop_leaf_table filters={} roles=rcccccccrrcc
- `cheap step stool with natural wood legs and metal legs for office`  
  gold: cat=step_stool filters={'leg_color': 'natural', 'leg_material': 'metal', 'room': 'office'} roles=rccfffffffff  
  pred: cat=step_stool filters={'room': 'office'} roles=rccccccccccc
- `office furniture for kids with hairpin legs boucle`  
  gold: cat=desks filters={'room': 'kids_room', 'leg_material': 'hairpin', 'material': 'boucle'} roles=ccffffff  
  pred: cat=None filters={} roles=rcrcrfcc
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
- `latex mattress twin xl extra firm cooling heavy duty for tall people`  
  gold: cat=latex_mattress filters={'bed_size': 'twin_xl', 'firmness': 'extra_firm', 'feature': 'cooling'} roles=ccfffffrrrrr  
  pred: cat=latex_mattress filters={'bed_size': 'twin_xl', 'firmness': 'extra_firm', 'feature': 'cooling'} roles=cccccccccrcc
- `seating with chrome legs art deco for studio`  
  gold: cat=seating filters={'leg_material': 'chrome', 'style': 'art_deco'} roles=cfffffrr  
  pred: cat=None filters={'style': 'art_deco'} roles=cfcccccc
- `wall shelf with acrylic legs and natural wood legs for tall people under 500 dollars`  
  gold: cat=wall_shelf filters={'leg_material': 'acrylic', 'leg_color': 'natural'} roles=ccfffffffrrrrrr  
  pred: cat=wall_shelf filters={'leg_material': 'acrylic'} roles=ccccccccfrcrcrr
- `gaming chair living room brass legs tufted made in usa for studio`  
  gold: cat=gaming_chair filters={'room': 'living_room', 'leg_color': 'brass', 'feature': 'tufted'} roles=ccfffffrrrrr  
  pred: cat=gaming_chair filters={'leg_material': 'metal', 'feature': 'tufted', 'room': 'living_room'} roles=cccccccrrcrc
- `furniture with engineered wood and acrylic legs like west elm heavy duty`  
  gold: cat=None filters={'material': 'engineered_wood', 'leg_material': 'acrylic'} roles=rrffrffrrrrr  
  pred: cat=None filters={'material': 'engineered_wood', 'leg_material': 'acrylic'} roles=cccccccrrrcc
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
- `dining chair with japandi style and brass legs luxury`  
  gold: cat=dining_chair filters={'leg_color': 'brass', 'style': 'japandi'} roles=ccffffffr  
  pred: cat=dining_chair filters={'leg_material': 'metal', 'room': 'dining_room'} roles=ccrccccfr
