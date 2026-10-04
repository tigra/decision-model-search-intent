# Search intent parsing: `router@ollama:nimble` via /v1/systemone

1000 queries, one request per query (24.6 questions on average).

## Accuracy

| Metric | **tuned decoding** (group-max category, filter p≥0.95, masked) | tuned, unmasked filters | untuned (plain argmax) |
|---|---|---|---|
| Category exact node acc | 0.836 | 0.836 | 0.626 |
| Category L1 acc (given gold depth ≥ 1) | 0.867 | 0.867 | 0.661 |
| Category L2 acc (given gold depth ≥ 2) | 0.893 | 0.893 | 0.637 |
| Category L3 acc (given gold depth ≥ 3) | 0.857 | 0.857 | 0.531 |
| Category hierarchical F1 | 0.845 | 0.845 | 0.652 |
| 'No category' P / R (tp/fp/fn) | 0.25 / 0.48 (23/69/25) | 0.25 / 0.48 (23/69/25) | 0.31 / 0.69 (33/74/15) |
| Filters micro P / R / F1 | 0.913 / 0.872 / 0.892 | 0.909 / 0.911 / 0.910 | 0.720 / 0.885 / 0.794 |
| Filter set exact match | 0.742 | 0.761 | 0.531 |
| Word-role accuracy | 0.839 | 0.839 | 0.787 |
| Word-role macro F1 | 0.833 | 0.833 | 0.782 |
| Residual words P / R / F1 | 0.681 / 0.786 / 0.730 | 0.681 / 0.786 / 0.730 | 0.607 / 0.817 / 0.697 |
| Residual set exact match | 0.524 | 0.524 | 0.448 |
| **Full query exact match** | **0.373** | **0.369** | **0.197** |

### Per-attribute filter scores (masked)

| attribute | support | P | R | F1 |
|---|---|---|---|---|
| feature | 220 | 0.995 | 0.968 | 0.982 |
| color | 211 | 0.931 | 0.962 | 0.946 |
| material | 185 | 0.973 | 0.773 | 0.861 |
| style | 182 | 0.994 | 0.984 | 0.989 |
| room | 147 | 0.846 | 0.932 | 0.887 |
| leg_color | 142 | 0.865 | 0.951 | 0.906 |
| leg_material | 132 | 0.681 | 0.856 | 0.758 |
| bed_size | 103 | 1.000 | 0.883 | 0.938 |
| width | 83 | 1.000 | 0.277 | 0.434 |
| shape | 66 | 0.917 | 0.667 | 0.772 |
| firmness | 36 | 1.000 | 0.861 | 0.925 |
| seat_count | 13 | 0.812 | 1.000 | 0.897 |

### Word roles (gold → predicted)

| gold \ pred | category | filter | residual |
|---|---|---|---|
| category | 1619 | 73 | 62 |
| filter | 39 | 2006 | 433 |
| residual | 75 | 213 | 1056 |

### Top category confusions (gold → predicted)

- pub_table → None: 18
- gaming_desk → None: 16
- wall_shelf → None: 14
- box_springs → None: 10
- None → storage: 9
- futon → None: 9
- mattresses → futon: 7
- curio_cabinet → futon: 6
- None → tables: 6
- desks → box_springs: 6
- mattress → futon: 5
- None → box_springs: 5
- beds → sectional_sofa: 4
- None → seating: 3
- tables → kids_beds: 3

## Latency (sequential, warm model)

| mean | p50 | p90 | p95 | p99 | max | queries/s | ms per question | mean input tokens |
|---|---|---|---|---|---|---|---|---|
| 15.980s | 15.855s | 17.629s | 18.025s | 18.996s | 19.637s | 0.06 | 650.2 | 132801 |

p50 latency by query length: 1-3 words: 14.700s (n=185), 4-6 words: 15.496s (n=505), 7-9 words: 16.344s (n=258), 10-12 words: 17.159s (n=47), 13-15 words: 18.398s (n=5)

Truncated queries (more than the word-question budget): 0

## 20 worst examples

- `office furniture for kids with hairpin legs boucle`  
  gold: cat=desks filters={'room': 'kids_room', 'leg_material': 'hairpin', 'material': 'boucle'} roles=ccffffff  
  pred: cat=box_springs filters={} roles=rcrrrffc
- `two seater sofa 42 in under 500 dollars`  
  gold: cat=loveseat filters={'width': '30_to_48in'} roles=cccffrrr  
  pred: cat=sofas filters={'seat_count': 'seats_2'} roles=ffcfrffr
- `wall shelf with acrylic legs and natural wood legs for tall people under 500 dollars`  
  gold: cat=wall_shelf filters={'leg_material': 'acrylic', 'leg_color': 'natural'} roles=ccfffffffrrrrrr  
  pred: cat=None filters={'leg_color': 'natural'} roles=ccrffrcffrcrffr
- `furniture square with acrylic legs charcoal easy to assemble reviews`  
  gold: cat=None filters={'shape': 'square', 'leg_material': 'acrylic', 'color': 'charcoal'} roles=rfffffrrrr  
  pred: cat=box_springs filters={} roles=cfrffffrrr
- `office furniture with walnut legs modern made in usa for studio`  
  gold: cat=desks filters={'leg_material': 'walnut', 'style': 'modern'} roles=ccffffrrrrr  
  pred: cat=box_springs filters={} roles=ccrfffrrfrf
- `futon couch with 4 seats oak stackable`  
  gold: cat=futon filters={'feature': 'stackable', 'seat_count': 'seats_4', 'material': 'oak'} roles=ccfffff  
  pred: cat=None filters={'leg_material': 'oak', 'seat_count': 'seats_4', 'feature': 'stackable'} roles=crrffrf
- `office furniture with teal color and acrylic legs oval`  
  gold: cat=desks filters={'color': 'teal', 'leg_material': 'acrylic', 'shape': 'oval'} roles=ccfffffff  
  pred: cat=box_springs filters={} roles=ccrffrfff
- `kids bed with drawers beige king`  
  gold: cat=kids_beds filters={'feature': 'with_drawers', 'color': 'beige', 'bed_size': 'king'} roles=ccffff  
  pred: cat=futon filters={'color': 'beige', 'feature': 'with_drawers', 'room': 'kids_room'} roles=rcrrff
- `dining table round pine for tall people entryway`  
  gold: cat=dining_tables filters={'material': 'pine', 'room': 'entryway', 'shape': 'round'} roles=ccffrrrf  
  pred: cat=box_springs filters={} roles=ccffrfff
- `dresser with drawers brass legs 114 cm under 500 dollars`  
  gold: cat=dressers filters={'leg_color': 'brass', 'width': '30_to_48in', 'feature': 'with_drawers'} roles=cffffffrrr  
  pred: cat=dressers filters={'leg_material': 'metal', 'leg_color': 'brass', 'feature': 'with_drawers'} roles=crrfffffff
- `furniture with natural wood legs oak legs hexagonal deals for tall people`  
  gold: cat=None filters={'leg_color': 'natural', 'leg_material': 'oak', 'shape': 'hexagonal'} roles=rfffffffrrrr  
  pred: cat=None filters={'leg_material': 'oak', 'shape': 'hexagonal'} roles=crcrfrrfrrrr
- `drop leaf table brass legs walnut legs foldable in stock pet friendly`  
  gold: cat=drop_leaf_table filters={'leg_color': 'brass', 'leg_material': 'walnut', 'feature': 'foldable'} roles=cccfffffrrrr  
  pred: cat=drop_leaf_table filters={'feature': 'foldable'} roles=ffcfffffrrff
- `furniture with chrome legs glam round`  
  gold: cat=None filters={'leg_material': 'chrome', 'style': 'glam', 'shape': 'round'} roles=rfffff  
  pred: cat=storage filters={'leg_material': 'chrome', 'leg_color': 'silver', 'style': 'glam'} roles=crffff
- `floating shelf for bathroom 69 inch glam easy to assemble`  
  gold: cat=wall_shelf filters={'room': 'bathroom', 'width': '48_to_72in', 'style': 'glam'} roles=ccfffffrrr  
  pred: cat=None filters={'style': 'glam', 'room': 'bathroom'} roles=fcrfffffrr
- `office furniture rectangular acacia bedroom reviews`  
  gold: cat=desks filters={'room': 'bedroom', 'shape': 'rectangular', 'material': 'acacia'} roles=ccfffr  
  pred: cat=box_springs filters={} roles=rcfffr
- `floating shelf with natural wood legs under 30 in ergonomic easy to assemble`  
  gold: cat=wall_shelf filters={'width': 'under_30in', 'leg_material': 'wood', 'leg_color': 'natural'} roles=ccrffffffrrrr  
  pred: cat=None filters={'leg_color': 'natural', 'width': 'under_30in'} roles=rcrfffffrfrrr
- `furniture best luxury leather oval green`  
  gold: cat=None filters={'material': 'leather', 'shape': 'oval', 'color': 'green'} roles=rrrfff  
  pred: cat=box_springs filters={} roles=crrfff
- `floating shelf art deco white legs for studio`  
  gold: cat=wall_shelf filters={'leg_color': 'white', 'style': 'art_deco'} roles=ccffffrr  
  pred: cat=None filters={'color': 'white', 'leg_material': 'oak', 'leg_color': 'white', 'style': 'art_deco'} roles=fcffffrf
- `furniture with engineered wood and acrylic legs like west elm heavy duty`  
  gold: cat=None filters={'material': 'engineered_wood', 'leg_material': 'acrylic'} roles=rrffrffrrrrr  
  pred: cat=box_springs filters={} roles=crffrffrrrcr
- `trestle table scandinavian style white legs under 30 inch 69 cm for small spaces ergonomic`  
  gold: cat=trestle_table filters={'width': 'under_30in', 'style': 'scandinavian', 'leg_color': 'white'} roles=ccfffffffffrrrr  
  pred: cat=trestle_table filters={'color': 'white', 'leg_color': 'white', 'style': 'scandinavian'} roles=ccfrrffffffrfrf
