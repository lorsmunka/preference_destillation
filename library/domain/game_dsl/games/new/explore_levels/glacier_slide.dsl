# Glacier Slide: the penguin slides until something stops it. Every swipe costs a move;
# fish give two back and each igloo gives four. Cracked ice and the seal reset the lake.
Window {
    size(8, 12)
    background("e0f2fe")
    music("bg_6")
    gravity(0)

    Bar("moves") {
        initial(8)
        OnValue(-1) { Actions { sound("lose") gameOver("-- OUT OF MOVES --") } }
    }
    Bar("fish") { initial(0) }

    Level {
        Entity { id("title") startPosition(20, 40) color("0c4a6e") Type { text("LAKE 1/3", 34) } }
        Entity { id("retry") startPosition(560, 30) color("0369a1") Type { text("RETRY", 34) }
            Events { OnTapped("retry") { Actions { sound("click") restart() } } } }
        Entity { id("shore") startPosition(0, 100) color("f8fafc") solid(true) Type { rectangle(800, 100) } }

        Entity { id("pen") startPosition(10, 210) color("1e293b") Type { circle(40) } var("sliding", 0)
            Movement { GridMove(100, 0.08) { direction(0, 0) } }
            Events {
                OnSwipe("left") { Actions { whenVar("sliding", "==", 0) { Actions { changeVar("sliding", 1) updateBar("moves", -1) setDirection("pen", -1, 0) sound("click") } } } }
                OnSwipe("right") { Actions { whenVar("sliding", "==", 0) { Actions { changeVar("sliding", 1) updateBar("moves", -1) setDirection("pen", 1, 0) sound("click") } } } }
                OnSwipe("up") { Actions { whenVar("sliding", "==", 0) { Actions { changeVar("sliding", 1) updateBar("moves", -1) setDirection("pen", 0, 1) sound("click") } } } }
                OnSwipe("down") { Actions { whenVar("sliding", "==", 0) { Actions { changeVar("sliding", 1) updateBar("moves", -1) setDirection("pen", 0, -1) sound("click") } } } }
                OnGridBlocked { Actions { setDirection("pen", 0, 0) changeVar("sliding", -1) sound("bump") } }
                OnCollision("igloo") { Actions { sound("win2") burst(20, "38bdf8") updateBar("moves", 4) nextLevel() } }
            } }

        Entity { id("rockA") startPosition(400, 200) color("64748b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rockB") startPosition(300, 900) color("64748b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rockC") startPosition(100, 600) color("64748b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rockD") startPosition(600, 500) color("64748b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rockE") startPosition(500, 1000) color("64748b") solid(true) Type { rectangle(100, 100) } }

        Entity { id("fish1") startPosition(225, 625) color("f97316") Type { text("🐟", 40) }
            Events { OnCollision("pen") { Actions { sound("coin") updateBar("moves", 2) updateBar("fish", 1) burst(10, "f97316") deleteEntity() } } } }
        Entity { id("igloo") startPosition(400, 800) color("ffffff") Type { polygon(8, 50) } }
    }

    Level {
        Entity { id("title") startPosition(20, 40) color("0c4a6e") Type { text("LAKE 2/3", 34) } }
        Entity { id("retry") startPosition(560, 30) color("0369a1") Type { text("RETRY", 34) }
            Events { OnTapped("retry") { Actions { sound("click") restart() } } } }
        Entity { id("shore") startPosition(0, 100) color("f8fafc") solid(true) Type { rectangle(800, 100) } }

        Entity { id("pen") startPosition(10, 1110) color("1e293b") Type { circle(40) } var("sliding", 0)
            Movement { GridMove(100, 0.08) { direction(0, 0) } }
            Events {
                OnSwipe("left") { Actions { whenVar("sliding", "==", 0) { Actions { changeVar("sliding", 1) updateBar("moves", -1) setDirection("pen", -1, 0) sound("click") } } } }
                OnSwipe("right") { Actions { whenVar("sliding", "==", 0) { Actions { changeVar("sliding", 1) updateBar("moves", -1) setDirection("pen", 1, 0) sound("click") } } } }
                OnSwipe("up") { Actions { whenVar("sliding", "==", 0) { Actions { changeVar("sliding", 1) updateBar("moves", -1) setDirection("pen", 0, 1) sound("click") } } } }
                OnSwipe("down") { Actions { whenVar("sliding", "==", 0) { Actions { changeVar("sliding", 1) updateBar("moves", -1) setDirection("pen", 0, -1) sound("click") } } } }
                OnGridBlocked { Actions { setDirection("pen", 0, 0) changeVar("sliding", -1) sound("bump") } }
                OnSignal("thaw") { Actions { flash("fde047", 0.4) } }
                OnCollision("igloo") { Actions { sound("win2") burst(20, "38bdf8") updateBar("moves", 4) nextLevel() } }
            } }

        Entity { id("rockA") startPosition(0, 400) color("64748b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rockB") startPosition(500, 500) color("64748b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rockC") startPosition(400, 800) color("64748b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rockD") startPosition(600, 200) color("64748b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rockE") startPosition(200, 900) color("64748b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rockF") startPosition(600, 1000) color("64748b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rockG") startPosition(100, 300) color("64748b") solid(true) Type { rectangle(100, 100) } }

        Entity { id("iceGate") startPosition(700, 300) color("7dd3fc") Type { rectangle(100, 100) } door("thaw", "glass") }
        Entity { id("goldFish") startPosition(225, 525) color("facc15") Type { text("🐠", 44) } key("thaw", "pen", "sparkle") }
        Entity { id("fish1") startPosition(325, 1025) color("f97316") Type { text("🐟", 40) }
            Events { OnCollision("pen") { Actions { sound("coin") updateBar("moves", 2) updateBar("fish", 1) burst(10, "f97316") deleteEntity() } } } }
        Entity { id("igloo") startPosition(700, 200) color("ffffff") Type { polygon(8, 50) } }
    }

    Level {
        Entity { id("title") startPosition(20, 40) color("0c4a6e") Type { text("LAKE 3/3", 34) } }
        Entity { id("retry") startPosition(560, 30) color("0369a1") Type { text("RETRY", 34) }
            Events { OnTapped("retry") { Actions { sound("click") restart() } } } }
        Entity { id("shore") startPosition(0, 100) color("f8fafc") solid(true) Type { rectangle(800, 100) } }

        Entity { id("pen") startPosition(10, 210) color("1e293b") Type { circle(40) } var("sliding", 0)
            Movement { GridMove(100, 0.08) { direction(0, 0) } }
            Events {
                OnSwipe("left") { Actions { whenVar("sliding", "==", 0) { Actions { changeVar("sliding", 1) updateBar("moves", -1) setDirection("pen", -1, 0) sound("click") } } } }
                OnSwipe("right") { Actions { whenVar("sliding", "==", 0) { Actions { changeVar("sliding", 1) updateBar("moves", -1) setDirection("pen", 1, 0) sound("click") } } } }
                OnSwipe("up") { Actions { whenVar("sliding", "==", 0) { Actions { changeVar("sliding", 1) updateBar("moves", -1) setDirection("pen", 0, 1) sound("click") } } } }
                OnSwipe("down") { Actions { whenVar("sliding", "==", 0) { Actions { changeVar("sliding", 1) updateBar("moves", -1) setDirection("pen", 0, -1) sound("click") } } } }
                OnGridBlocked { Actions { setDirection("pen", 0, 0) changeVar("sliding", -1) sound("bump") } }
                OnCollision("seal") { Actions { sound("hurt") shake(6, 0.3) restart() } }
                OnCollision("igloo") { Actions { sound("win") shake(5, 0.4) burst(30, "38bdf8") gameOver("-- HOME BEFORE THE THAW --") } }
            } }

        Entity { id("rockA") startPosition(300, 200) color("64748b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rockB") startPosition(200, 700) color("64748b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rockC") startPosition(600, 600) color("64748b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rockD") startPosition(500, 1100) color("64748b") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rockE") startPosition(200, 1000) color("64748b") solid(true) Type { rectangle(100, 100) } }

        Entity { id("crackA") startPosition(0, 800) color("93c5fd") Type { cross(100, 100, 20) }
            Events { OnCollision("pen") { Actions { sound("glass") burst(16, "bfdbfe") restart() } } } }
        Entity { id("crackB") startPosition(500, 400) color("93c5fd") Type { cross(100, 100, 20) }
            Events { OnCollision("pen") { Actions { sound("glass") burst(16, "bfdbfe") restart() } } } }
        Entity { id("crackC") startPosition(100, 600) color("93c5fd") Type { cross(100, 100, 20) }
            Events { OnCollision("pen") { Actions { sound("glass") burst(16, "bfdbfe") restart() } } } }

        Entity { id("seal") color("78716c") Type { rectangle(80, 60) }
            Movement { speed(2.2) Path { waypoint(10, 820) waypoint(710, 820) } } }
        Entity { id("fish1") startPosition(625, 325) color("f97316") Type { text("🐟", 40) }
            Events { OnCollision("pen") { Actions { sound("coin") updateBar("moves", 2) updateBar("fish", 1) burst(10, "f97316") deleteEntity() } } } }
        Entity { id("igloo") startPosition(300, 1100) color("ffffff") Type { polygon(8, 50) } }
    }
}
