NAMES = [
    ("Alice", "she", "her"), ("Bob", "he", "his"), ("Carol", "she", "her"),
    ("David", "he", "his"), ("Emma", "she", "her"), ("Frank", "he", "his"),
    ("Grace", "she", "her"), ("Henry", "he", "his"), ("Isla", "she", "her"),
    ("Jack", "he", "his"), ("Kate", "she", "her"), ("Leo", "he", "his"),
    ("Mia", "she", "her"), ("Noah", "he", "his"), ("Olivia", "she", "her"),
    ("Peter", "he", "his"), ("Quinn", "she", "her"), ("Ryan", "he", "his"),
    ("Sarah", "she", "her"), ("Tom", "he", "his"), ("Uma", "she", "her"),
    ("Victor", "he", "his"), ("Wendy", "she", "her"), ("Xander", "he", "his"),
    ("Yara", "she", "her"), ("Zach", "he", "his"), ("Bella", "she", "her"),
    ("Carlos", "he", "his"), ("Diana", "she", "her"), ("Ethan", "he", "his"),
    ("Fiona", "she", "her"), ("George", "he", "his"), ("Hannah", "she", "her"),
    ("Ivan", "he", "his"), ("Julia", "she", "her"), ("Kevin", "he", "his"),
    ("Luna", "she", "her"), ("Marcus", "he", "his"), ("Nina", "she", "her"),
    ("Oscar", "he", "his"), ("Paula", "she", "her"), ("Ray", "he", "his"),
    ("Sophie", "she", "her"), ("Tyler", "he", "his"), ("Vera", "she", "her"),
    ("Will", "he", "his"), ("Zoe", "she", "her"), ("Amy", "she", "her"),
    ("Ben", "he", "his"), ("Clara", "she", "her"), ("Dan", "he", "his"),
    ("Eva", "she", "her"), ("Finn", "he", "his"), ("Gina", "she", "her"),
    ("Hugo", "he", "his"), ("Iris", "she", "her"), ("Jake", "he", "his"),
    ("Lily", "she", "her"), ("Max", "he", "his"), ("Nora", "she", "her"),
    ("Owen", "he", "his"), ("Rosa", "she", "her"), ("Sam", "he", "his"),
]

BAKEABLES = [
    "cookies", "muffins", "cupcakes", "brownies", "pies", "cakes",
    "scones", "pastries", "bagels", "bread rolls", "croissants", "donuts",
]

CONSUMABLES = [
    "apples", "oranges", "bananas", "sandwiches", "grapes", "peaches",
    "pears", "candies", "tacos", "pancakes", "strawberries", "pizzas",
]

COLLECTIBLES = [
    "stickers", "cards", "marbles", "stamps", "coins", "badges",
    "postcards", "ribbons", "buttons", "pins", "patches", "tickets",
]

SCHOOL_SUPPLIES = [
    "pencils", "notebooks", "erasers", "crayons", "markers", "pens",
    "rulers", "folders", "books", "worksheets", "papers", "calculators",
]

NATURE_ITEMS = [
    "flowers", "seeds", "leaves", "rocks", "shells", "pinecones",
    "acorns", "feathers", "plants", "mushrooms", "berries", "pebbles",
]

ANIMALS = [
    "chickens", "rabbits", "fish", "ducks", "cats", "dogs",
    "hamsters", "turtles", "horses", "goats", "pigs", "sheep",
]

MANUFACTURED = [
    "bottles", "parts", "widgets", "bolts", "screws", "batteries",
    "light bulbs", "clips", "tubes", "plates", "bricks", "tiles",
]

CLOTHING = [
    "shirts", "socks", "hats", "scarves", "gloves", "shoes",
    "ties", "jackets", "belts", "caps", "sweaters", "coats",
]

ALL_ITEMS = (
    BAKEABLES + CONSUMABLES + COLLECTIBLES + SCHOOL_SUPPLIES
    + NATURE_ITEMS + MANUFACTURED + CLOTHING
)

CONTAINERS = [
    "bags", "boxes", "baskets", "crates", "jars", "packs",
    "trays", "buckets", "cartons", "bins",
]

GROUPS = [
    "friends", "students", "children", "people", "teams",
    "families", "classmates", "workers",
]


def singularize(word):
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ves"):
        return word[:-3] + "f"
    if word.endswith("ses") or word.endswith("shes") or word.endswith("ches"):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word
