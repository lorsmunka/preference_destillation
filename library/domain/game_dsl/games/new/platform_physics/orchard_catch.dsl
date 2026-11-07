Window {
    size(7, 11)
    background("fff7e6")
    music("bg_nature")
    gravity(500)

    Bar("score") { initial(0) }
    winAt("score", 50, "-- BASKET FULL --", "win")
    lives(3, "-- STUNG THREE TIMES --", "lose")

    paddle("basket", 290, 40, 120, 30)

    Entity { id("grass") startPosition(0, 0) color("65a30d")
        Type { rectangle(700, 30) } }

    # light cherries drift down, heavy melons drop like stones
    Entity { id("cherry") template(true) startPosition(0, 0) color("dc2626") gravity(260)
        Type { text("🍒", 34) }
        Movement { speed(0.5) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("basket") { Actions { sound("coin") updateBar("score", 1) burst(8, "dc2626") deleteEntity() } }
            OnCollision("grass") { Actions { sound("pop") burst(6, "7f1d1d") deleteEntity() } } } }

    Entity { id("pear") template(true) startPosition(0, 0) color("a3e635")
        Type { text("🍐", 38) }
        Movement { speed(0.5) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("basket") { Actions { sound("coin") updateBar("score", 2) burst(10, "a3e635") deleteEntity() } }
            OnCollision("grass") { Actions { sound("pop") burst(6, "4d7c0f") deleteEntity() } } } }

    Entity { id("melon") template(true) startPosition(0, 0) color("16a34a") gravity(950)
        Type { text("🍉", 46) }
        Movement { speed(0.3) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("basket") { Actions { sound("bell") updateBar("score", 3) burst(16, "f43f5e") shake(3, 0.2) deleteEntity() } }
            OnCollision("grass") { Actions { sound("thud") burst(14, "f43f5e") shake(4, 0.2) deleteEntity() } } } }

    # wasps fly in zigzags and ignore gravity
    Entity { id("wasp") template(true) startPosition(0, 0) color("facc15") gravity(0)
        Type { text("🐝", 36) }
        Movement { speed(1.8) AutoMove { direction(0.7, -1) } }
        Events {
            OnCollision("basket") { Actions {
                sound("hurt") updateBar("lives", -1)
                flash("basket", "ff0000", 0.3) shake(6, 0.3) burst(12, "facc15")
                deleteEntity() } }
            OnCollision("grass") { Actions { sound("buzz") deleteEntity() } }
            OnCollision { Actions { bounceBack(0) } } } }

    Wave(20, 1.1) { Actions { randomAction {
        Actions { spawn("cherry", random(30, 640), 1040) }
        Actions { spawn("pear", random(30, 640), 1040) } } } }
    Wave(20, 0.9) { Actions { randomAction {
        Actions { spawn("cherry", random(30, 640), 1040) }
        Actions { spawn("pear", random(30, 640), 1040) }
        Actions { spawn("melon", random(30, 640), 1040) }
        Actions { spawn("wasp", random(30, 640), 1040) } } } }
    Wave(24, 0.7) { Actions { randomAction {
        Actions { spawn("cherry", random(30, 640), 1040) }
        Actions { spawn("pear", random(30, 640), 1040) }
        Actions { spawn("melon", random(30, 640), 1040) }
        Actions { spawn("wasp", random(30, 640), 1040) }
        Actions { spawn("wasp", random(30, 640), 1040) } } } }
    Wave(1) { Actions { sound("lose") shake(5, 0.4) gameOver("-- SUNSET: THE HARVEST IS OVER --") } }
}
