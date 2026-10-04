"""Synthetic furniture product-type ontology, filterable attributes and residual vocabulary.

Category tree: L1 -> L2 -> (optional) L3, organized by product type (not by room).
Each node: id -> (display name, [surface synonyms]).
"""
import json
import re
from pathlib import Path

# ---------------------------------------------------------------- categories
# Nested spec: {l1_id: (name, syns, {l2_id: (name, syns, {l3_id: (name, syns)})})}
TREE = {
    "seating": ("Seating", ["seating", "seats"], {
        "sofas": ("Sofas", ["sofa", "couch", "settee"], {
            "sectional_sofa": ("Sectional sofa", ["sectional", "l shaped couch", "corner sofa", "sectional couch"]),
            "loveseat": ("Loveseat", ["loveseat", "love seat", "two seater sofa"]),
            "sleeper_sofa": ("Sleeper sofa", ["sleeper sofa", "sofa bed", "pull out couch"]),
            "chesterfield": ("Chesterfield sofa", ["chesterfield", "chesterfield sofa"]),
            "futon": ("Futon", ["futon", "futon couch"]),
        }),
        "chairs": ("Chairs", ["chair", "chairs"], {
            "accent_chair": ("Accent chair", ["accent chair", "armchair", "arm chair"]),
            "dining_chair": ("Dining chair", ["dining chair", "kitchen chair"]),
            "office_chair": ("Office chair", ["office chair", "desk chair", "task chair"]),
            "gaming_chair": ("Gaming chair", ["gaming chair", "gamer chair"]),
            "rocking_chair": ("Rocking chair", ["rocking chair", "rocker"]),
            "lounge_chair": ("Lounge chair", ["lounge chair", "lounger"]),
        }),
        "stools": ("Stools", ["stool", "stools"], {
            "bar_stool": ("Bar stool", ["bar stool", "barstool"]),
            "counter_stool": ("Counter stool", ["counter stool", "counter height stool"]),
            "step_stool": ("Step stool", ["step stool"]),
        }),
        "benches": ("Benches", ["bench", "benches"], {}),
        "ottomans": ("Ottomans & poufs", ["ottoman", "pouf", "footstool", "foot rest"], {}),
        "recliners": ("Recliners", ["recliner", "recliner chair", "lazy boy"], {}),
        "bean_bags": ("Bean bags", ["bean bag", "beanbag chair"], {}),
        "chaise_lounges": ("Chaise lounges", ["chaise", "chaise lounge", "daybed chaise"], {}),
    }),
    "tables": ("Tables", ["table", "tables"], {
        "coffee_tables": ("Coffee tables", ["coffee table", "cocktail table", "center table"], {
            "lift_top_coffee_table": ("Lift-top coffee table", ["lift top coffee table", "lift-top coffee table"]),
            "nesting_coffee_table": ("Nesting coffee tables", ["nesting coffee table", "nesting tables"]),
        }),
        "dining_tables": ("Dining tables", ["dining table", "kitchen table", "dinner table"], {
            "drop_leaf_table": ("Drop-leaf table", ["drop leaf table", "drop-leaf table", "gateleg table"]),
            "pub_table": ("Pub table", ["pub table", "bar table", "bistro table"]),
            "trestle_table": ("Trestle table", ["trestle table"]),
        }),
        "side_tables": ("Side tables", ["side table", "end table", "accent table", "sofa side table"], {}),
        "console_tables": ("Console tables", ["console table", "entryway table", "sofa table", "hallway table"], {}),
        "vanity_tables": ("Vanity tables", ["vanity table", "makeup vanity", "dressing table"], {}),
        "kitchen_islands": ("Kitchen islands", ["kitchen island", "kitchen cart", "island cart"], {}),
    }),
    "beds": ("Beds", ["bed", "beds"], {
        "bed_frames": ("Bed frames", ["bed frame", "bedframe"], {
            "platform_bed": ("Platform bed", ["platform bed", "low profile bed"]),
            "canopy_bed": ("Canopy bed", ["canopy bed", "four poster bed", "four-poster bed"]),
            "sleigh_bed": ("Sleigh bed", ["sleigh bed"]),
            "upholstered_bed": ("Upholstered bed", ["upholstered bed", "padded bed"]),
        }),
        "kids_beds": ("Kids beds", ["kids bed", "children bed", "toddler bed"], {
            "bunk_bed": ("Bunk bed", ["bunk bed", "bunkbed", "bunk beds"]),
            "loft_bed": ("Loft bed", ["loft bed", "high sleeper"]),
            "trundle_bed": ("Trundle bed", ["trundle bed", "trundle"]),
        }),
        "daybeds": ("Daybeds", ["daybed", "day bed"], {}),
        "murphy_beds": ("Murphy beds", ["murphy bed", "wall bed", "fold down bed"], {}),
        "headboards": ("Headboards", ["headboard", "head board"], {}),
        "cribs": ("Cribs", ["crib", "baby crib", "cot"], {}),
    }),
    "mattresses": ("Mattresses & bed bases", ["sleep products", "mattresses and bases", "mattress and foundation"], {
        "mattress": ("Mattresses", ["mattress", "mattresses"], {
            "memory_foam_mattress": ("Memory foam mattress", ["memory foam mattress", "foam mattress"]),
            "hybrid_mattress": ("Hybrid mattress", ["hybrid mattress"]),
            "innerspring_mattress": ("Innerspring mattress", ["innerspring mattress", "spring mattress", "pocket spring mattress"]),
            "latex_mattress": ("Latex mattress", ["latex mattress"]),
        }),
        "box_springs": ("Box springs", ["box spring", "boxspring", "foundation"], {}),
        "adjustable_bases": ("Adjustable bases", ["adjustable base", "adjustable bed base", "power base"], {}),
        "mattress_toppers": ("Mattress toppers", ["mattress topper", "mattress pad", "topper"], {}),
    }),
    "storage": ("Storage furniture", ["storage furniture", "storage units", "storage unit"], {
        "dressers": ("Dressers & chests", ["dresser", "chest of drawers", "drawer chest", "tallboy"], {}),
        "nightstands": ("Nightstands", ["nightstand", "night stand", "bedside table", "night table"], {}),
        "bookcases": ("Bookcases & shelves", ["bookcase", "bookshelf", "shelving unit", "book shelf"], {
            "ladder_shelf": ("Ladder shelf", ["ladder shelf", "leaning shelf", "ladder bookcase"]),
            "cube_organizer": ("Cube organizer", ["cube organizer", "cube shelf", "cubby"]),
            "wall_shelf": ("Wall shelf", ["wall shelf", "floating shelf", "wall mounted shelf"]),
        }),
        "cabinets": ("Cabinets", ["cabinet", "cupboard"], {
            "accent_cabinet": ("Accent cabinet", ["accent cabinet", "accent chest"]),
            "curio_cabinet": ("Curio cabinet", ["curio cabinet", "display cabinet", "china cabinet", "vitrine"]),
            "pantry_cabinet": ("Pantry cabinet", ["pantry cabinet", "kitchen pantry", "larder"]),
        }),
        "wardrobes": ("Wardrobes & armoires", ["wardrobe", "armoire", "closet cabinet"], {}),
        "tv_stands": ("TV stands", ["tv stand", "media console", "entertainment center", "tv unit"], {}),
        "sideboards": ("Sideboards & buffets", ["sideboard", "buffet", "credenza"], {}),
        "shoe_storage": ("Shoe storage", ["shoe rack", "shoe cabinet", "shoe organizer", "shoe bench"], {}),
    }),
    "desks": ("Desks & workstations", ["office furniture", "workstation", "desks and workstations"], {
        "desk": ("Desks", ["desk", "desks"], {
            "standing_desk": ("Standing desk", ["standing desk", "stand up desk", "sit stand desk"]),
            "l_shaped_desk": ("L-shaped desk", ["l shaped desk", "l-shaped desk", "corner desk"]),
            "writing_desk": ("Writing desk", ["writing desk", "secretary desk"]),
            "computer_desk": ("Computer desk", ["computer desk", "pc desk"]),
            "gaming_desk": ("Gaming desk", ["gaming desk"]),
        }),
        "filing_cabinets": ("Filing cabinets", ["filing cabinet", "file cabinet", "file drawer"], {}),
        "desk_hutches": ("Desk hutches", ["desk hutch", "hutch"], {}),
        "drafting_tables": ("Drafting tables", ["drafting table", "drawing table", "craft table"], {}),
    }),
}

# ---------------------------------------------------------------- attributes
# attr -> (description, {value_id: [surface forms]})
ATTRIBUTES = {
    "color": ("Main color of the item", {
        "black": ["black"], "white": ["white"], "grey": ["grey", "gray"], "beige": ["beige", "sand"],
        "brown": ["brown", "chocolate"], "blue": ["blue", "light blue"], "navy": ["navy", "navy blue"],
        "green": ["green", "sage", "olive green"], "emerald": ["emerald", "emerald green"],
        "red": ["red", "burgundy"], "pink": ["pink", "blush"], "yellow": ["yellow", "mustard"],
        "orange": ["orange", "rust", "terracotta"], "cream": ["cream", "ivory", "off white"],
        "charcoal": ["charcoal", "dark grey"], "teal": ["teal", "turquoise"],
        "purple": ["purple", "lavender"],
    }),
    "material": ("Main material or upholstery of the item (not the legs)", {
        "leather": ["leather", "genuine leather"], "faux_leather": ["faux leather", "vegan leather", "pu leather"],
        "velvet": ["velvet"], "linen": ["linen"], "boucle": ["boucle", "bouclé"], "fabric": ["fabric", "upholstered fabric"],
        "microfiber": ["microfiber", "microfibre"], "solid_wood": ["solid wood", "wooden", "wood"],
        "oak": ["oak"], "walnut": ["walnut"], "pine": ["pine"], "acacia": ["acacia"], "mango_wood": ["mango wood"],
        "bamboo": ["bamboo"], "metal": ["metal", "steel", "iron"], "glass": ["glass", "tempered glass"],
        "marble": ["marble", "marble top"], "rattan": ["rattan", "wicker", "cane"], "plastic": ["plastic", "acrylic"],
        "engineered_wood": ["mdf", "particle board", "engineered wood"],
    }),
    "leg_material": ("Material of the legs or base only", {
        "wood": ["wood legs", "wooden legs"], "oak": ["oak legs"], "walnut": ["walnut legs"],
        "metal": ["metal legs", "steel legs", "iron legs"], "chrome": ["chrome legs", "chrome base"],
        "acrylic": ["acrylic legs", "lucite legs"], "hairpin": ["hairpin legs"],
    }),
    "leg_color": ("Color of the legs or base only", {
        "black": ["black legs", "black base"], "white": ["white legs"], "gold": ["gold legs", "gold base"],
        "brass": ["brass legs"], "silver": ["silver legs", "silver base"], "natural": ["natural wood legs", "light wood legs"],
        "dark_brown": ["dark legs", "dark brown legs", "espresso legs"],
    }),
    "style": ("Design style", {
        "modern": ["modern"], "contemporary": ["contemporary"], "mid_century": ["mid century", "mid-century", "mcm", "mid century modern"],
        "farmhouse": ["farmhouse"], "industrial": ["industrial"], "scandinavian": ["scandinavian", "scandi", "nordic"],
        "traditional": ["traditional", "classic"], "rustic": ["rustic"], "bohemian": ["boho", "bohemian"],
        "minimalist": ["minimalist", "minimal"], "glam": ["glam", "hollywood glam"], "coastal": ["coastal", "beach style"],
        "japandi": ["japandi"], "vintage": ["vintage", "retro"], "art_deco": ["art deco"],
    }),
    "shape": ("Shape of the top or body", {
        "round": ["round", "circular"], "square": ["square"], "rectangular": ["rectangular", "rectangle"],
        "oval": ["oval"], "curved": ["curved", "rounded"], "hexagonal": ["hexagon", "hexagonal"],
    }),
    "bed_size": ("Bed or mattress size", {
        "twin": ["twin", "single"], "twin_xl": ["twin xl"], "full": ["full", "double"], "queen": ["queen", "queen size"],
        "king": ["king", "king size"], "california_king": ["california king", "cal king"],
    }),
    "width": ("Width / length of the item as a size bucket (number with inch, in, \", ft or cm)", {
        "under_30in": [], "30_to_48in": [], "48_to_72in": [], "72_to_90in": [], "over_90in": [],
    }),
    "seat_count": ("Number of people it seats", {
        "seats_2": ["2 seater", "seats 2", "for 2"], "seats_3": ["3 seater", "seats 3", "three seat"],
        "seats_4": ["4 seater", "seats 4", "for 4", "4 person"], "seats_6": ["6 seater", "seats 6", "for 6", "6 person"],
        "seats_8": ["8 seater", "seats 8", "for 8", "8 person"],
    }),
    "firmness": ("Mattress firmness", {
        "soft": ["soft", "plush"], "medium": ["medium", "medium firm"], "firm": ["firm"], "extra_firm": ["extra firm"],
    }),
    "feature": ("A functional feature", {
        "with_storage": ["with storage", "storage"], "with_drawers": ["with drawers", "drawers"],
        "reclining": ["reclining", "power reclining"], "swivel": ["swivel"], "foldable": ["foldable", "folding"],
        "extendable": ["extendable", "expandable", "extending"], "adjustable_height": ["adjustable height", "height adjustable"],
        "stackable": ["stackable", "stacking"], "usb_charging": ["with usb", "usb charging", "with charging station"],
        "glass_doors": ["with glass doors", "glass doors"], "tufted": ["tufted", "button tufted"],
        "cooling": ["cooling", "cooling gel"], "wheels": ["on wheels", "with casters", "rolling"],
    }),
    "room": ("Room or place of use", {
        "outdoor": ["outdoor", "patio", "garden"], "kids_room": ["kids", "for kids", "nursery"],
        "office": ["home office", "office"], "bedroom": ["bedroom"], "living_room": ["living room"],
        "dining_room": ["dining room"], "entryway": ["entryway", "hallway", "entry"], "bathroom": ["bathroom"],
    }),
}

WIDTH_BUCKETS = {  # value -> (min_inch_inclusive, max_inch_exclusive)
    "under_30in": (0, 30), "30_to_48in": (30, 48), "48_to_72in": (48, 72),
    "72_to_90in": (72, 90), "over_90in": (90, 1000),
}

# Which attributes apply to which L1 subtree (overrides at L2 via APPLICABLE_L2).
APPLICABLE_L1 = {
    "seating": ["color", "material", "leg_material", "leg_color", "style", "feature", "room"],
    "tables": ["color", "material", "leg_material", "leg_color", "style", "shape", "width", "feature", "room"],
    "beds": ["color", "material", "style", "bed_size", "feature", "room"],
    "mattresses": ["bed_size", "firmness", "feature"],
    "storage": ["color", "material", "leg_material", "leg_color", "style", "width", "feature", "room"],
    "desks": ["color", "material", "leg_material", "leg_color", "style", "width", "shape", "feature", "room"],
}
APPLICABLE_L2 = {
    "sofas": APPLICABLE_L1["seating"] + ["width", "seat_count", "shape"],
    "benches": APPLICABLE_L1["seating"] + ["width"],
    "ottomans": APPLICABLE_L1["seating"] + ["shape"],
    "dining_tables": APPLICABLE_L1["tables"] + ["seat_count"],
    "headboards": ["color", "material", "style", "bed_size", "feature"],
    "cribs": ["color", "material", "style", "feature"],
    "box_springs": ["bed_size"],
    "adjustable_bases": ["bed_size", "feature"],
    "mattress_toppers": ["bed_size", "firmness", "feature"],
}
# Feature values that make sense per L1 (keeps generated queries realistic).
FEATURES_L1 = {
    "seating": ["with_storage", "reclining", "swivel", "foldable", "stackable", "usb_charging", "tufted", "wheels"],
    "tables": ["with_storage", "with_drawers", "foldable", "extendable", "adjustable_height", "wheels", "glass_doors"],
    "beds": ["with_storage", "with_drawers", "tufted", "usb_charging", "foldable"],
    "mattresses": ["cooling", "foldable"],
    "storage": ["with_drawers", "glass_doors", "wheels", "foldable"],
    "desks": ["with_drawers", "with_storage", "adjustable_height", "usb_charging", "foldable", "wheels"],
}
ROOMS_L1 = {
    "seating": ["outdoor", "kids_room", "office", "bedroom", "living_room", "dining_room", "entryway"],
    "tables": ["outdoor", "kids_room", "living_room", "dining_room", "entryway", "office"],
    "beds": ["kids_room", "bedroom"],
    "mattresses": [],
    "storage": ["kids_room", "bedroom", "living_room", "dining_room", "entryway", "bathroom", "office"],
    "desks": ["kids_room", "office", "bedroom"],
}

# ---------------------------------------------------------------- residual vocab
RESIDUAL_PHRASES = [
    "cheap", "affordable", "best", "highly rated", "on sale", "sale", "deals", "clearance", "free shipping",
    "near me", "for my mom", "gift", "2024", "2025", "reviews", "ikea", "like west elm", "like pottery barn",
    "for small apartment", "for small spaces", "comfortable", "sturdy", "easy to assemble", "heavy duty",
    "under $300", "under 500 dollars", "luxury", "budget", "fast delivery", "in stock", "used",
    "for my cat", "pet friendly", "for tall people", "ergonomic", "quality", "brand new", "handmade",
    "made in usa", "for studio", "bundle", "set",
]


# ---------------------------------------------------------------- flattened views
class Node:
    __slots__ = ("id", "name", "synonyms", "level", "parent", "children")

    def __init__(self, id, name, synonyms, level, parent):
        self.id, self.name, self.synonyms, self.level, self.parent = id, name, synonyms, level, parent
        self.children = []

    def path(self):
        out, n = [], self
        while n is not None:
            out.append(n.id)
            n = NODES.get(n.parent) if n.parent else None
        return out[::-1]


NODES = {}


def _build():
    for l1, (n1, s1, kids1) in TREE.items():
        NODES[l1] = Node(l1, n1, s1, 1, None)
        for l2, (n2, s2, kids2) in kids1.items():
            NODES[l2] = Node(l2, n2, s2, 2, l1)
            NODES[l1].children.append(l2)
            for l3, (n3, s3) in kids2.items():
                NODES[l3] = Node(l3, n3, s3, 3, l2)
                NODES[l2].children.append(l3)


_build()
L1_IDS = list(TREE)


def applicable_attrs(node_id):
    """Attributes applicable to a category node (None -> all)."""
    if node_id is None:
        return list(ATTRIBUTES)
    path = NODES[node_id].path()
    if len(path) >= 2 and path[1] in APPLICABLE_L2:
        return APPLICABLE_L2[path[1]]
    return APPLICABLE_L1[path[0]]


def _words(s):
    return re.findall(r"[a-z0-9$]+", s.lower())


def _check_collisions():
    attr_words = set()
    for _, (_, vals) in ATTRIBUTES.items():
        for forms in vals.values():
            for f in forms:
                attr_words.update(_words(f))
    cat_words = set()
    for n in NODES.values():
        for f in n.synonyms:
            cat_words.update(_words(f))
    function_words = {"for", "with", "my", "to", "in", "of", "on", "under", "like", "made", "set"}
    bad = []
    for p in RESIDUAL_PHRASES:
        for w in _words(p):
            if w in function_words:
                continue
            if w in attr_words or w in cat_words:
                bad.append((p, w))
    assert not bad, "residual/attribute or category vocabulary collision: %s" % bad
    seen = {}
    for nid, n in NODES.items():
        for syn in n.synonyms:
            assert syn not in seen, "synonym %r shared by %s and %s" % (syn, seen[syn], nid)
            seen[syn] = nid
    for n in NODES.values():
        for c in n.children:
            assert len(NODES[c].children) <= 25
    for a, (_, vals) in ATTRIBUTES.items():
        assert len(vals) <= 25, a


_check_collisions()


def dump(path):
    out = {
        "categories": {nid: {"name": n.name, "synonyms": n.synonyms, "level": n.level,
                             "parent": n.parent, "children": n.children} for nid, n in NODES.items()},
        "attributes": {a: {"description": d, "values": v} for a, (d, v) in ATTRIBUTES.items()},
        "width_buckets_inch": WIDTH_BUCKETS,
        "applicable_l1": APPLICABLE_L1, "applicable_l2": APPLICABLE_L2,
        "residual_phrases": RESIDUAL_PHRASES,
    }
    Path(path).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    dump(Path(__file__).resolve().parents[2] / "data" / "schema.json")
    from collections import Counter
    print(Counter(n.level for n in NODES.values()), "nodes;", len(ATTRIBUTES), "attributes")
