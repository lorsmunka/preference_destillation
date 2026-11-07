# Ice cave. The marble slides until something stops it. It can only swallow the gem whose power
# matches its own (1, then 2, 4, 8): any other gem stops it like a rock. Its colour shows the
# gem it can take next. Stopping on a crack drops it back to the entrance.
Window {
    size(7, 10)
    background("0c1a2b")
    music("bg_5")
    gravity(0)

    lives(3, "-- LOST IN THE ICE --", "lose")

    Level {
        Entity { id("roof") startPosition(0, 900) color("1e3a5f") Type { rectangle(700, 100) } solid(true) }
        Entity { id("ground") startPosition(0, 0) color("1e3a5f") Type { rectangle(700, 100) } solid(true) }
        Entity { id("title") startPosition(20, 930) color("bae6fd") Type { text("CAVE 1/3", 36) } }
        Entity { id("hint") startPosition(20, 35) color("7dd3fc") Type { text("swipe to slide", 30) } }

        Entity { id("r1") startPosition(0, 800) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r2") startPosition(300, 800) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r3") startPosition(600, 800) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r4") startPosition(100, 700) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r5") startPosition(200, 400) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r6") startPosition(500, 400) color("475569") Type { rectangle(100, 100) } solid(true) }

        Entity { id("gem1") startPosition(520, 520) color("38bdf8") Type { diamond(60, 60) } var("power", 1) }
        Entity { id("gem2") startPosition(20, 620) color("a78bfa") Type { diamond(60, 60) } var("power", 2) }
        Entity { id("gem4") startPosition(120, 620) color("f472b6") Type { diamond(60, 60) } var("power", 4) }
        Entity { id("gem8") startPosition(420, 620) color("facc15") Type { diamond(60, 60) } var("power", 8) }

        Entity { id("marble") startPosition(10, 110) color("38bdf8") Type { circle(40) } var("power", 1)
            Movement { GridMove(100, 1) { direction(0, 0) } }
            Events {
                OnSwipe("left") { Actions { sound("blip") slide(-1, 0, "power", "gemTaken") } }
                OnSwipe("right") { Actions { sound("blip") slide(1, 0, "power", "gemTaken") } }
                OnSwipe("up") { Actions { sound("blip") slide(0, 1, "power", "gemTaken") } }
                OnSwipe("down") { Actions { sound("blip") slide(0, -1, "power", "gemTaken") } }
                OnSignal("gemTaken") { Actions { sound("sparkle") burst(14, "e0f2fe")
                    whenVar("power", "==", 2) { Actions { setColor("a78bfa") } }
                    whenVar("power", "==", 4) { Actions { setColor("f472b6") } }
                    whenVar("power", "==", 8) { Actions { setColor("facc15") } }
                    whenVar("power", "==", 16) { Actions { sound("win2") shake(4, 0.3) nextLevel() } } } } } }
    }

    Level {
        Entity { id("roof") startPosition(0, 900) color("1e3a5f") Type { rectangle(700, 100) } solid(true) }
        Entity { id("ground") startPosition(0, 0) color("1e3a5f") Type { rectangle(700, 100) } solid(true) }
        Entity { id("title") startPosition(20, 930) color("bae6fd") Type { text("CAVE 2/3", 36) } }
        Entity { id("hint") startPosition(20, 35) color("7dd3fc") Type { text("mind the cracks", 30) } }

        Entity { id("r1") startPosition(200, 800) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r2") startPosition(300, 600) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r3") startPosition(600, 600) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r4") startPosition(0, 500) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r5") startPosition(500, 400) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r6") startPosition(100, 300) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r7") startPosition(300, 200) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r8") startPosition(600, 100) color("475569") Type { rectangle(100, 100) } solid(true) }

        Entity { id("crack1") startPosition(320, 520) color("94a3b8") Type { ring(30, 18) }
            Events { OnCollision("marble") { Actions { sound("hurt") shake(5, 0.3) burst(12, "94a3b8")
                updateBar("lives", -1) setPosition("marble", 10, 110, 0) } } } }

        Entity { id("gem1") startPosition(220, 420) color("38bdf8") Type { diamond(60, 60) } var("power", 1) }
        Entity { id("gem2") startPosition(620, 320) color("a78bfa") Type { diamond(60, 60) } var("power", 2) }
        Entity { id("gem4") startPosition(20, 720) color("f472b6") Type { diamond(60, 60) } var("power", 4) }
        Entity { id("gem8") startPosition(520, 820) color("facc15") Type { diamond(60, 60) } var("power", 8) }

        Entity { id("marble") startPosition(10, 110) color("38bdf8") Type { circle(40) } var("power", 1)
            Movement { GridMove(100, 1) { direction(0, 0) } }
            Events {
                OnSwipe("left") { Actions { sound("blip") slide(-1, 0, "power", "gemTaken") } }
                OnSwipe("right") { Actions { sound("blip") slide(1, 0, "power", "gemTaken") } }
                OnSwipe("up") { Actions { sound("blip") slide(0, 1, "power", "gemTaken") } }
                OnSwipe("down") { Actions { sound("blip") slide(0, -1, "power", "gemTaken") } }
                OnSignal("gemTaken") { Actions { sound("sparkle") burst(14, "e0f2fe")
                    whenVar("power", "==", 2) { Actions { setColor("a78bfa") } }
                    whenVar("power", "==", 4) { Actions { setColor("f472b6") } }
                    whenVar("power", "==", 8) { Actions { setColor("facc15") } }
                    whenVar("power", "==", 16) { Actions { sound("win2") shake(4, 0.3) nextLevel() } } } } } }
    }

    Level {
        Entity { id("roof") startPosition(0, 900) color("1e3a5f") Type { rectangle(700, 100) } solid(true) }
        Entity { id("ground") startPosition(0, 0) color("1e3a5f") Type { rectangle(700, 100) } solid(true) }
        Entity { id("title") startPosition(20, 930) color("bae6fd") Type { text("CAVE 3/3", 36) } }
        Entity { id("hint") startPosition(20, 35) color("7dd3fc") Type { text("the deep hall", 30) } }

        Entity { id("r1") startPosition(100, 800) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r2") startPosition(0, 700) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r3") startPosition(400, 700) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r4") startPosition(100, 600) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r5") startPosition(500, 600) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r6") startPosition(100, 300) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r7") startPosition(0, 200) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("r8") startPosition(500, 100) color("475569") Type { rectangle(200, 100) } solid(true) }

        Entity { id("crack1") startPosition(420, 820) color("94a3b8") Type { ring(30, 18) }
            Events { OnCollision("marble") { Actions { sound("hurt") shake(5, 0.3) burst(12, "94a3b8")
                updateBar("lives", -1) setPosition("marble", 10, 110, 0) } } } }
        Entity { id("crack2") startPosition(520, 820) color("94a3b8") Type { ring(30, 18) }
            Events { OnCollision("marble") { Actions { sound("hurt") shake(5, 0.3) burst(12, "94a3b8")
                updateBar("lives", -1) setPosition("marble", 10, 110, 0) } } } }
        Entity { id("crack3") startPosition(120, 520) color("94a3b8") Type { ring(30, 18) }
            Events { OnCollision("marble") { Actions { sound("hurt") shake(5, 0.3) burst(12, "94a3b8")
                updateBar("lives", -1) setPosition("marble", 10, 110, 0) } } } }
        Entity { id("crack4") startPosition(420, 320) color("94a3b8") Type { ring(30, 18) }
            Events { OnCollision("marble") { Actions { sound("hurt") shake(5, 0.3) burst(12, "94a3b8")
                updateBar("lives", -1) setPosition("marble", 10, 110, 0) } } } }

        Entity { id("gem1") startPosition(620, 220) color("38bdf8") Type { diamond(60, 60) } var("power", 1) }
        Entity { id("gem2") startPosition(520, 320) color("a78bfa") Type { diamond(60, 60) } var("power", 2) }
        Entity { id("gem4") startPosition(320, 220) color("f472b6") Type { diamond(60, 60) } var("power", 4) }
        Entity { id("gem8") startPosition(220, 820) color("facc15") Type { diamond(60, 60) } var("power", 8) }

        Entity { id("marble") startPosition(10, 110) color("38bdf8") Type { circle(40) } var("power", 1)
            Movement { GridMove(100, 1) { direction(0, 0) } }
            Events {
                OnSwipe("left") { Actions { sound("blip") slide(-1, 0, "power", "gemTaken") } }
                OnSwipe("right") { Actions { sound("blip") slide(1, 0, "power", "gemTaken") } }
                OnSwipe("up") { Actions { sound("blip") slide(0, 1, "power", "gemTaken") } }
                OnSwipe("down") { Actions { sound("blip") slide(0, -1, "power", "gemTaken") } }
                OnSignal("gemTaken") { Actions { sound("sparkle") burst(14, "e0f2fe")
                    whenVar("power", "==", 2) { Actions { setColor("a78bfa") } }
                    whenVar("power", "==", 4) { Actions { setColor("f472b6") } }
                    whenVar("power", "==", 8) { Actions { setColor("facc15") } }
                    whenVar("power", "==", 16) { Actions { sound("win") shake(6, 0.5) burst(30, "facc15")
                        gameOver("-- THE CRYSTAL HALL IS YOURS --") } } } } } }
    }
}
