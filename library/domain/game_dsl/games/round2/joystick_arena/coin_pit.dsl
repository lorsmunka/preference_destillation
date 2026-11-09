# Drag the miner around the pit, grab the coins, stay clear of the beetles.
Window {
    size(10, 10)
    background("1c1917")
    music("bg_1")
    gravity(0)
    lives(3, "-- BEETLE FOOD --", "lose")
    winAt("coins", 8, "-- POCKETS FULL --", "win")
    wallBox(0, 0, 1000, 1000, "57534e", 20)

    Entity { id("miner") startPosition(480, 480) color("fbbf24") Type { circle(26) }
        Movement { speed(6) Control { movementControl(0) } } }

    Entity { id("wall") startPosition(300, 300) color("78716c") solid(true) Type { rectangle(20, 400) } }
    Entity { id("wall2") startPosition(680, 300) color("78716c") solid(true) Type { rectangle(20, 400) } }

    Entity { id("coin1") startPosition(100, 100) color("facc15") Type { circle(16) }
        Events { OnCollision("miner") { Actions { sound("coin") updateBar("coins", 1) burst(6, "facc15") deleteEntity() } } } }
    Entity { id("coin2") startPosition(860, 100) color("facc15") Type { circle(16) }
        Events { OnCollision("miner") { Actions { sound("coin") updateBar("coins", 1) burst(6, "facc15") deleteEntity() } } } }
    Entity { id("coin3") startPosition(100, 860) color("facc15") Type { circle(16) }
        Events { OnCollision("miner") { Actions { sound("coin") updateBar("coins", 1) burst(6, "facc15") deleteEntity() } } } }
    Entity { id("coin4") startPosition(860, 860) color("facc15") Type { circle(16) }
        Events { OnCollision("miner") { Actions { sound("coin") updateBar("coins", 1) burst(6, "facc15") deleteEntity() } } } }
    Entity { id("coin5") startPosition(490, 120) color("fde68a") Type { circle(16) }
        Events { OnCollision("miner") { Actions { sound("coin") updateBar("coins", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("coin6") startPosition(490, 850) color("fde68a") Type { circle(16) }
        Events { OnCollision("miner") { Actions { sound("coin") updateBar("coins", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("coin7") startPosition(180, 480) color("fde68a") Type { circle(16) }
        Events { OnCollision("miner") { Actions { sound("coin") updateBar("coins", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("coin8") startPosition(790, 480) color("fde68a") Type { circle(16) }
        Events { OnCollision("miner") { Actions { sound("coin") updateBar("coins", 1) burst(6, "fde68a") deleteEntity() } } } }

    Entity { id("beetle1") startPosition(100, 200) color("dc2626") Type { polygon(6, 24) }
        Movement { speed(2.2) Path { waypoint(100, 200) waypoint(860, 200) } }
        Events { OnCollision("miner") { Actions { sound("hurt") updateBar("lives", -1) shake(4, 0.3) burst(8, "dc2626") setPosition("miner", 480, 480, 0) } } } }
    Entity { id("beetle2") startPosition(860, 760) color("ea580c") Type { polygon(6, 24) }
        Movement { speed(2.8) Path { waypoint(860, 760) waypoint(100, 760) } }
        Events { OnCollision("miner") { Actions { sound("hurt") updateBar("lives", -1) shake(4, 0.3) burst(8, "ea580c") setPosition("miner", 480, 480, 0) } } } }
    Entity { id("beetle3") startPosition(500, 100) color("be123c") Type { polygon(6, 24) }
        Movement { speed(2) Path { waypoint(500, 100) waypoint(500, 880) } }
        Events { OnCollision("miner") { Actions { sound("hurt") updateBar("lives", -1) shake(4, 0.3) burst(8, "be123c") setPosition("miner", 480, 480, 0) } } } }
}
