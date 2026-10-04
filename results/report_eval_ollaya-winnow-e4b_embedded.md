# Search intent parsing: `embedded@ollaya:winnow:e4b` via /v1/systemone

1000 queries, one request per query (23.6 questions on average).

## Accuracy

| Metric | **tuned decoding** (group-max category, filter p≥0.95, masked) | tuned, unmasked filters | untuned (plain argmax) |
|---|---|---|---|
| Category exact node acc | 0.966 | 0.966 | 0.962 |
| Category L1 acc (given gold depth ≥ 1) | 0.985 | 0.985 | 0.980 |
| Category L2 acc (given gold depth ≥ 2) | 0.988 | 0.988 | 0.988 |
| Category L3 acc (given gold depth ≥ 3) | 0.965 | 0.965 | 0.965 |
| Category hierarchical F1 | 0.978 | 0.978 | 0.974 |
| 'No category' P / R (tp/fp/fn) | 0.96 / 0.92 (44/2/4) | 0.96 / 0.92 (44/2/4) | 0.87 / 0.94 (45/7/3) |
| Filters micro P / R / F1 | 0.817 / 0.864 / 0.840 | 0.796 / 0.867 / 0.830 | 0.800 / 0.876 / 0.836 |
| Filter set exact match | 0.619 | 0.584 | 0.616 |
| Word-role accuracy | 0.699 | 0.699 | 0.752 |
| Word-role macro F1 | 0.585 | 0.585 | 0.632 |
| Residual words P / R / F1 | 1.000 / 0.100 / 0.183 | 1.000 / 0.100 / 0.183 | 0.986 / 0.102 / 0.185 |
| Residual set exact match | 0.419 | 0.419 | 0.421 |
| **Full query exact match** | **0.245** | **0.235** | **0.246** |

### Per-attribute filter scores (masked)

| attribute | support | P | R | F1 |
|---|---|---|---|---|
| feature | 220 | 0.908 | 0.809 | 0.856 |
| color | 211 | 0.795 | 0.972 | 0.874 |
| material | 185 | 0.969 | 0.670 | 0.792 |
| style | 182 | 0.984 | 0.984 | 0.984 |
| room | 147 | 0.444 | 0.728 | 0.552 |
| leg_color | 142 | 0.879 | 0.972 | 0.923 |
| leg_material | 132 | 0.772 | 0.947 | 0.850 |
| bed_size | 103 | 0.970 | 0.942 | 0.956 |
| width | 83 | 0.827 | 0.747 | 0.785 |
| shape | 66 | 0.910 | 0.924 | 0.917 |
| firmness | 36 | 0.962 | 0.694 | 0.806 |
| seat_count | 13 | 0.800 | 0.923 | 0.857 |

### Word roles (gold → predicted)

| gold \ pred | category | filter | residual |
|---|---|---|---|
| category | 1742 | 12 | 0 |
| filter | 457 | 2021 | 0 |
| residual | 199 | 1010 | 135 |

### Top category confusions (gold → predicted)

- curio_cabinet → cabinets: 5
- None → tables: 4
- beds → mattresses: 2
- desks → None: 2
- sleigh_bed → mattresses: 2
- upholstered_bed → mattresses: 2
- desks → desk: 2
- seating → console_tables: 1
- step_stool → dining_tables: 1
- loveseat → sofas: 1
- sectional_sofa → sofas: 1
- cube_organizer → bookcases: 1
- sectional_sofa → console_tables: 1
- sleigh_bed → sleeper_sofa: 1
- chairs → dining_chair: 1

## Latency (sequential, warm model)

| mean | p50 | p90 | p95 | p99 | max | queries/s | ms per question | mean input tokens |
|---|---|---|---|---|---|---|---|---|
| 6.252s | 6.238s | 6.652s | 6.770s | 7.012s | 7.742s | 0.16 | 265.2 | 9099 |

p50 latency by query length: 1-3 words: 5.990s (n=185), 4-6 words: 6.215s (n=505), 7-9 words: 6.538s (n=258), 10-12 words: 6.860s (n=47), 13-15 words: 7.273s (n=5)

Truncated queries (more than the word-question budget): 0

## 20 worst examples

- `dining table with glass doors wood legs under 300 for my cat`  
  gold: cat=dining_tables filters={'leg_material': 'wood', 'feature': 'glass_doors'} roles=ccfffffrrrrr  
  pred: cat=dining_tables filters={'leg_material': 'wood', 'feature': 'glass_doors', 'room': 'dining_room'} roles=ccccccffffrf
- `furniture with engineered wood and acrylic legs like west elm heavy duty`  
  gold: cat=None filters={'material': 'engineered_wood', 'leg_material': 'acrylic'} roles=rrffrffrrrrr  
  pred: cat=None filters={'leg_material': 'acrylic'} roles=cffffffffcff
- `china cabinet with glass doors for small apartment luxury`  
  gold: cat=curio_cabinet filters={'feature': 'glass_doors'} roles=ccfffrrrr  
  pred: cat=cabinets filters={'feature': 'glass_doors', 'room': 'living_room'} roles=ccffcffff
- `wall shelf with acrylic legs and natural wood legs for tall people under 500 dollars`  
  gold: cat=wall_shelf filters={'leg_material': 'acrylic', 'leg_color': 'natural'} roles=ccfffffffrrrrrr  
  pred: cat=wall_shelf filters={'leg_color': 'natural'} roles=ccffcffffffffff
- `drafting table velvet art deco for my cat under 300`  
  gold: cat=drafting_tables filters={'material': 'velvet', 'style': 'art_deco'} roles=ccfffrrrrr  
  pred: cat=drafting_tables filters={'material': 'velvet', 'style': 'art_deco', 'width': 'under_30in'} roles=cccccfrcff
- `adjustable base full foldable heavy duty comfortable`  
  gold: cat=adjustable_bases filters={'feature': 'foldable', 'bed_size': 'full'} roles=ccffrrr  
  pred: cat=adjustable_bases filters={} roles=fcccfcf
- `gaming chair living room brass legs tufted made in usa for studio`  
  gold: cat=gaming_chair filters={'room': 'living_room', 'leg_color': 'brass', 'feature': 'tufted'} roles=ccfffffrrrrr  
  pred: cat=gaming_chair filters={'leg_color': 'brass', 'room': 'living_room'} roles=cccfffcffffc
- `murphy bed tufted for small spaces heavy duty`  
  gold: cat=murphy_beds filters={'feature': 'tufted'} roles=ccfrrrrr  
  pred: cat=murphy_beds filters={'room': 'bedroom'} roles=cccfffff
- `furniture square with acrylic legs charcoal easy to assemble reviews`  
  gold: cat=None filters={'shape': 'square', 'leg_material': 'acrylic', 'color': 'charcoal'} roles=rfffffrrrr  
  pred: cat=tables filters={'color': 'charcoal', 'leg_material': 'acrylic', 'shape': 'square'} roles=ccffffffff
- `bed frame twin xl made in usa pet friendly`  
  gold: cat=bed_frames filters={'bed_size': 'twin_xl'} roles=ccffrrrrr  
  pred: cat=bed_frames filters={'bed_size': 'twin_xl', 'room': 'bedroom'} roles=ccccfffff
- `accent cabinet entryway linen near me for small apartmet`  
  gold: cat=accent_cabinet filters={'room': 'entryway', 'material': 'linen'} roles=ccffrrrrr  
  pred: cat=accent_cabinet filters={'material': 'linen', 'room': 'entryway'} roles=ccccfffff
- `latex mattress twin xl extra firm cooling heavy duty for tall people`  
  gold: cat=latex_mattress filters={'bed_size': 'twin_xl', 'firmness': 'extra_firm', 'feature': 'cooling'} roles=ccfffffrrrrr  
  pred: cat=latex_mattress filters={'bed_size': 'twin_xl', 'firmness': 'extra_firm', 'feature': 'cooling'} roles=ccccfffccfff
- `bed tufted solid wood bedroom 2024`  
  gold: cat=beds filters={'feature': 'tufted', 'room': 'bedroom', 'material': 'solid_wood'} roles=cffffr  
  pred: cat=mattresses filters={} roles=ccffff
- `bed frame full bedroom in stock for small apartment`  
  gold: cat=bed_frames filters={'bed_size': 'full', 'room': 'bedroom'} roles=ccffrrrrr  
  pred: cat=bed_frames filters={'bed_size': 'full', 'room': 'bedroom'} roles=ccccfffff
- `dresser with wood legs foldable fast delivery for tall people`  
  gold: cat=dressers filters={'leg_material': 'wood', 'feature': 'foldable'} roles=cffffrrrrr  
  pred: cat=dressers filters={'leg_material': 'wood', 'feature': 'foldable'} roles=ccfcffffff
- `drop leaf table brass legs walnut legs foldable in stock pet friendly`  
  gold: cat=drop_leaf_table filters={'leg_color': 'brass', 'leg_material': 'walnut', 'feature': 'foldable'} roles=cccfffffrrrr  
  pred: cat=drop_leaf_table filters={'leg_material': 'walnut', 'leg_color': 'brass', 'feature': 'foldable'} roles=cccfcccfffff
- `trestle table bamboo square like pottery barn free shipping`  
  gold: cat=trestle_table filters={'material': 'bamboo', 'shape': 'square'} roles=ccffrrrrr  
  pred: cat=trestle_table filters={'material': 'bamboo', 'shape': 'square'} roles=ccccfccff
- `drafting table square gold legs heavy duty best`  
  gold: cat=drafting_tables filters={'shape': 'square', 'leg_color': 'gold'} roles=ccfffrrr  
  pred: cat=drafting_tables filters={'color': 'grey', 'leg_color': 'gold', 'shape': 'square', 'room': 'office'} roles=cccfcfff
- `sleigh bed california king walnut foldable affordable near me`  
  gold: cat=sleigh_bed filters={'bed_size': 'california_king', 'material': 'walnut', 'feature': 'foldable'} roles=ccffffrrr  
  pred: cat=sleigh_bed filters={'color': 'brown', 'material': 'walnut', 'bed_size': 'king', 'feature': 'foldable', 'room': 'bedroom'} roles=ccfffffff
- `ladder shelf with walnut and chrome legs farmhouse style ikea`  
  gold: cat=ladder_shelf filters={'material': 'walnut', 'leg_material': 'chrome', 'style': 'farmhouse'} roles=ccrfrfffrr  
  pred: cat=ladder_shelf filters={'color': 'brown', 'leg_material': 'chrome', 'leg_color': 'silver', 'style': 'farmhouse'} roles=ccfffffffc
