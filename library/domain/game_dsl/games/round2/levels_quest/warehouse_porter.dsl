# Warehouse Porter: push each crate onto a marked slot, one tap per step. Fill the slots to move to the next hall.
Window {
    size(8, 12)
    background("27272a")
    music("bg_6")
    gravity(0)
    Bar("placed") {
        initial(0)
        OnValue(1) { Actions { sound("win2") nextLevel() } }
        OnValue(3) { Actions { sound("win2") nextLevel() } }
        OnValue(6) { Actions { sound("win") gameOver("-- THE WAREHOUSE IS TIDY --") } }
    }
    Bar("moves") { initial(0) }

    Level {
        dpad("porter", "step")
        Entity { id("slot1") startPosition(400, 700) color("166534") Type { rectangle(100, 100) } }
        Entity { id("crate1") startPosition(400, 600) color("a16207") pushable(true) Type { rectangle(100, 100) }
            Events { OnCollision("slot1") { Actions { sound("thud") burst(10, "4ade80") updateBar("placed", 1) deleteEntity() } } } }
        Entity { id("porter") startPosition(400, 500) color("38bdf8") Type { circle(40) }
            Movement { GridMove(100, 0.3) { direction(0, 0) } }
            Events { OnGridStep { Actions { updateBar("moves", 1) } } } }
        Entity { id("pillar1") startPosition(200, 700) color("52525b") solid(true) Type { rectangle(100, 100) } }
    }

    Level {
        dpad("porter", "step")
        Entity { id("slot1") startPosition(200, 900) color("166534") Type { rectangle(100, 100) } }
        Entity { id("slot2") startPosition(500, 900) color("166534") Type { rectangle(100, 100) } }
        Entity { id("crate1") startPosition(200, 700) color("a16207") pushable(true) Type { rectangle(100, 100) }
            Events { OnCollision("slot1") { Actions { sound("thud") burst(10, "4ade80") updateBar("placed", 1) deleteEntity() } } } }
        Entity { id("crate2") startPosition(500, 700) color("ca8a04") pushable(true) Type { rectangle(100, 100) }
            Events { OnCollision("slot2") { Actions { sound("thud") burst(10, "4ade80") updateBar("placed", 1) deleteEntity() } } } }
        Entity { id("porter") startPosition(400, 500) color("38bdf8") Type { circle(40) }
            Movement { GridMove(100, 0.3) { direction(0, 0) } }
            Events { OnGridStep { Actions { updateBar("moves", 1) } } } }
        Entity { id("pillar1") startPosition(300, 800) color("52525b") solid(true) Type { rectangle(100, 100) } }
    }

    Level {
        dpad("porter", "step")
        Entity { id("slot1") startPosition(100, 1000) color("166534") Type { rectangle(100, 100) } }
        Entity { id("slot2") startPosition(400, 1000) color("166534") Type { rectangle(100, 100) } }
        Entity { id("slot3") startPosition(700, 1000) color("166534") Type { rectangle(100, 100) } }
        Entity { id("crate1") startPosition(100, 800) color("a16207") pushable(true) Type { rectangle(100, 100) }
            Events { OnCollision("slot1") { Actions { sound("thud") burst(10, "4ade80") updateBar("placed", 1) deleteEntity() } } } }
        Entity { id("crate2") startPosition(400, 800) color("ca8a04") pushable(true) Type { rectangle(100, 100) }
            Events { OnCollision("slot2") { Actions { sound("thud") burst(10, "4ade80") updateBar("placed", 1) deleteEntity() } } } }
        Entity { id("crate3") startPosition(700, 800) color("b45309") pushable(true) Type { rectangle(100, 100) }
            Events { OnCollision("slot3") { Actions { sound("thud") burst(10, "4ade80") updateBar("placed", 1) deleteEntity() } } } }
        Entity { id("porter") startPosition(400, 500) color("38bdf8") Type { circle(40) }
            Movement { GridMove(100, 0.3) { direction(0, 0) } }
            Events { OnGridStep { Actions { updateBar("moves", 1) } } } }
        Entity { id("pillar1") startPosition(200, 700) color("52525b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("pillar2") startPosition(500, 700) color("52525b") solid(true) Type { rectangle(100, 100) } }
    }
}
