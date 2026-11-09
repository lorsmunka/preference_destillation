# Slime Arena: slimes crawl in from the edges toward your crystal. Tap them to pop them; pop enough to clear each arena.
Window {
    size(8, 12)
    background("14532d")
    music("bg_6")
    Bar("crystal") {
        initial(6)
        OnValue(0) { Actions { sound("lose") gameOver("-- THE CRYSTAL SHATTERED --") } }
    }
    Bar("pops") {
        initial(0)
        OnValue(8) { Actions { sound("win2") nextLevel() } }
        OnValue(20) { Actions { sound("win2") nextLevel() } }
        OnValue(36) { Actions { sound("win") gameOver("-- THE ARENA IS CLEAR --") } }
    }

    Level {
        Entity { id("core") startPosition(360, 560) color("22d3ee") Type { diamond(80, 100) } }
        Entity { id("slime") template(true) startPosition(0, 0) color("84cc16") Type { circle(34) }
            Movement { speed(1.2) chase("core", 0.3) }
            Events {
                OnTapped { Actions { sound("pop") updateBar("pops", 1) burst(10, "bef264") deleteEntity() } }
                OnCollision("core") { Actions { sound("hurt") updateBar("crystal", -1) shake(4, 0.2) deleteEntity() } }
            } }
        Entity { id("gate") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnElapsedTime(2.2) { repeatCount(-1) Actions { randomAction {
                Actions { spawn("slime", 40, random(100, 1100)) }
                Actions { spawn("slime", 720, random(100, 1100)) }
            } } } } }
    }

    Level {
        Entity { id("core") startPosition(360, 560) color("22d3ee") Type { diamond(80, 100) } }
        Entity { id("slime") template(true) startPosition(0, 0) color("84cc16") Type { circle(34) }
            Movement { speed(1.6) chase("core", 0.3) }
            Events {
                OnTapped { Actions { sound("pop") updateBar("pops", 1) burst(10, "bef264") deleteEntity() } }
                OnCollision("core") { Actions { sound("hurt") updateBar("crystal", -1) shake(4, 0.2) deleteEntity() } }
            } }
        Entity { id("brute") template(true) startPosition(0, 0) color("65a30d") Type { circle(54) }
            var("hp", 3)
            Movement { speed(0.9) chase("core", 0.3) }
            Events {
                OnTapped { Actions { changeVar("hp", -1) flash("ffffff", 0.15)
                    whenVar("hp", "<=", 0) { Actions { sound("pop") updateBar("pops", 3) burst(16, "d9f99d") deleteEntity() } } } }
                OnCollision("core") { Actions { sound("hurt") updateBar("crystal", -1) shake(6, 0.3) deleteEntity() } }
            } }
        Entity { id("gate") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events {
                OnElapsedTime(1.8) { repeatCount(-1) Actions { randomAction {
                    Actions { spawn("slime", 40, random(100, 1100)) }
                    Actions { spawn("slime", 720, random(100, 1100)) }
                    Actions { spawn("slime", random(60, 700), 1140) }
                } } }
                OnElapsedTime(7) { repeatCount(-1) Actions { spawn("brute", random(60, 700), 1100) } }
            } }
    }

    Level {
        Entity { id("core") startPosition(360, 560) color("22d3ee") Type { diamond(80, 100) } }
        Entity { id("slime") template(true) startPosition(0, 0) color("84cc16") Type { circle(34) }
            Movement { speed(2) chase("core", 0.3) }
            Events {
                OnTapped { Actions { sound("pop") updateBar("pops", 1) burst(10, "bef264") deleteEntity() } }
                OnCollision("core") { Actions { sound("hurt") updateBar("crystal", -1) shake(4, 0.2) deleteEntity() } }
            } }
        Entity { id("brute") template(true) startPosition(0, 0) color("65a30d") Type { circle(54) }
            var("hp", 3)
            Movement { speed(1.2) chase("core", 0.3) }
            Events {
                OnTapped { Actions { changeVar("hp", -1) flash("ffffff", 0.15)
                    whenVar("hp", "<=", 0) { Actions { sound("pop") updateBar("pops", 3) burst(16, "d9f99d") deleteEntity() } } } }
                OnCollision("core") { Actions { sound("hurt") updateBar("crystal", -1) shake(6, 0.3) deleteEntity() } }
            } }
        Entity { id("splitter") template(true) startPosition(0, 0) color("fde047") Type { polygon(5, 36) }
            Movement { speed(1.5) chase("core", 0.3) }
            Events {
                OnTapped { Actions { sound("pop") updateBar("pops", 2) burst(12, "fef08a")
                    spawnHere("slime") spawnHere("slime") deleteEntity() } }
                OnCollision("core") { Actions { sound("hurt") updateBar("crystal", -1) deleteEntity() } }
            } }
        Entity { id("gate") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events {
                OnElapsedTime(1.4) { repeatCount(-1) Actions { randomAction {
                    Actions { spawn("slime", 40, random(100, 1100)) }
                    Actions { spawn("slime", 720, random(100, 1100)) }
                    Actions { spawn("slime", random(60, 700), 1140) }
                } } }
                OnElapsedTime(6) { repeatCount(-1) Actions { spawn("brute", random(60, 700), 1100) } }
                OnElapsedTime(9) { repeatCount(-1) Actions { spawn("splitter", random(60, 700), 100) } }
            } }
    }
}
