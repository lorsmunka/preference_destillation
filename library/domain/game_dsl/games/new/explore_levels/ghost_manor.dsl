# Ghost Manor: three floors. Tap the portraits to look behind them -- one hides the key to
# the stairs, the others let a ghost out. On the top floor, light every candle to find the way out.
Window {
    size(8, 12)
    worldSize(12, 14)
    cameraFollow("hero")
    background("1c1917")
    music("bg_3")
    gravity(0)

    lives(4, "-- THE MANOR KEEPS YOU --", "lose")
    Bar("candles") {
        initial(0)
        OnValue(4) { Actions { broadcast("allCandles") } }
    }

    Level {
        Entity { id("note") startPosition(40, 250) color("d6d3d1") Type { text("FLOOR 1 - one portrait hides the key", 26) }
            Events { OnSignal("brass") { Actions { setText("note", "The key opens the gate to the stairs") } } } }
        Entity { id("hero") startPosition(100, 100) color("fef08a") Type { rectangle(40, 50) }
            Movement { speed(5) Control { movementControl(0) } }
            Events {
                OnSignal("brass") { Actions { sound("pickup") burst(12, "eab308") } }
                OnCollision("stairs") { Actions { sound("win2") burst(16, "fef08a") nextLevel() } } } }

        Entity { id("wallA") startPosition(0, 500) color("44403c") solid(true) Type { rectangle(900, 40) } }
        Entity { id("wallB") startPosition(1050, 500) color("44403c") solid(true) Type { rectangle(150, 40) } }
        Entity { id("wallC") startPosition(0, 950) color("44403c") solid(true) Type { rectangle(150, 40) } }
        Entity { id("wallD") startPosition(300, 950) color("44403c") solid(true) Type { rectangle(900, 40) } }
        Entity { id("gate") startPosition(150, 950) color("a16207") Type { rectangle(150, 40) } door("brass", "clang") }

        Entity { id("ghost") template(true) color("b3f5f5f4") Type { circle(26) }
            Movement { speed(1.3) chase("hero") }
            Events { OnCollision("hero") { Actions { sound("hurt") shake(6, 0.3) flash("hero", "a5b4fc", 0.4) updateBar("lives", -1) burst(14, "e7e5e4") deleteEntity() } } } }

        Entity { id("pA") startPosition(320, 400) color("b45309") Type { rectangle(70, 90) } var("seen", 0)
            Events { OnTapped("pA") { Actions { whenVar("seen", "==", 0) { Actions { changeVar("seen", 1) setColor("57534e") sound("death") spawn("ghost", 340, 300) } } } } } }
        Entity { id("pB") startPosition(760, 400) color("b45309") Type { rectangle(70, 90) } var("seen", 0)
            Events { OnTapped("pB") { Actions { whenVar("seen", "==", 0) { Actions { changeVar("seen", 1) setColor("57534e") sound("death") spawn("ghost", 780, 300) } } } } } }
        Entity { id("pC") startPosition(200, 850) color("b45309") Type { rectangle(70, 90) } var("seen", 0)
            Events { OnTapped("pC") { Actions { whenVar("seen", "==", 0) { Actions { changeVar("seen", 1) setColor("57534e") sound("sparkle") setEnable("brassKey", true) } } } } } }
        Entity { id("pD") startPosition(700, 850) color("b45309") Type { rectangle(70, 90) } var("seen", 0)
            Events { OnTapped("pD") { Actions { whenVar("seen", "==", 0) { Actions { changeVar("seen", 1) setColor("57534e") sound("death") spawn("ghost", 720, 700) } } } } } }

        Entity { id("brassKey") startPosition(220, 760) color("eab308") Type { star(4, 22, 9) } key("brass", "hero")
            Events { OnElapsedTime { Actions { setEnable(false) } } } }
        Entity { id("stairs") startPosition(950, 1200) color("78716c") Type { rectangle(140, 120) } }
    }

    Level {
        Entity { id("note") startPosition(660, 380) color("d6d3d1") Type { text("FLOOR 2 - mind the maid", 26) }
            Events { OnSignal("silver") { Actions { setText("note", "Silver key found - the stairs are west") } } } }
        Entity { id("hero") startPosition(1050, 100) color("fef08a") Type { rectangle(40, 50) }
            Movement { speed(5) Control { movementControl(0) } }
            Events {
                OnSignal("silver") { Actions { sound("pickup") burst(12, "e5e7eb") } }
                OnCollision("stairs") { Actions { sound("win2") burst(16, "fef08a") nextLevel() } } } }

        Entity { id("wallA") startPosition(0, 500) color("44403c") solid(true) Type { rectangle(150, 40) } }
        Entity { id("wallB") startPosition(300, 500) color("44403c") solid(true) Type { rectangle(900, 40) } }
        Entity { id("wallC") startPosition(600, 540) color("44403c") solid(true) Type { rectangle(40, 300) } }
        Entity { id("wallD") startPosition(0, 950) color("44403c") solid(true) Type { rectangle(900, 40) } }
        Entity { id("wallE") startPosition(1050, 950) color("44403c") solid(true) Type { rectangle(150, 40) } }
        Entity { id("gate") startPosition(900, 950) color("94a3b8") Type { rectangle(150, 40) } door("silver", "clang") }

        Entity { id("ghost") template(true) color("b3f5f5f4") Type { circle(26) }
            Movement { speed(1.5) chase("hero") }
            Events { OnCollision("hero") { Actions { sound("hurt") shake(6, 0.3) flash("hero", "a5b4fc", 0.4) updateBar("lives", -1) burst(14, "e7e5e4") deleteEntity() } } } }

        Entity { id("pA") startPosition(500, 400) color("b45309") Type { rectangle(70, 90) } var("seen", 0)
            Events { OnTapped("pA") { Actions { whenVar("seen", "==", 0) { Actions { changeVar("seen", 1) setColor("57534e") sound("death") spawn("ghost", 520, 300) } } } } } }
        Entity { id("pB") startPosition(100, 850) color("b45309") Type { rectangle(70, 90) } var("seen", 0)
            Events { OnTapped("pB") { Actions { whenVar("seen", "==", 0) { Actions { changeVar("seen", 1) setColor("57534e") sound("death") spawn("ghost", 120, 700) } } } } } }
        Entity { id("pC") startPosition(450, 850) color("b45309") Type { rectangle(70, 90) } var("seen", 0)
            Events { OnTapped("pC") { Actions { whenVar("seen", "==", 0) { Actions { changeVar("seen", 1) setColor("57534e") sound("death") spawn("ghost", 470, 700) } } } } } }
        Entity { id("pD") startPosition(1000, 850) color("b45309") Type { rectangle(70, 90) } var("seen", 0)
            Events { OnTapped("pD") { Actions { whenVar("seen", "==", 0) { Actions { changeVar("seen", 1) setColor("57534e") sound("sparkle") setEnable("silverKey", true) } } } } } }

        Entity { id("silverKey") startPosition(1020, 640) color("e5e7eb") Type { star(4, 22, 9) } key("silver", "hero")
            Events { OnElapsedTime { Actions { setEnable(false) } } } }
        Entity { id("maid") color("c4b5fd") Type { circle(28) }
            Movement { speed(2.2) Path { waypoint(80, 1100) waypoint(1100, 1100) waypoint(1100, 1300) waypoint(80, 1300) } }
            Events { OnCollision("hero") { Actions { sound("hurt") shake(6, 0.3) flash("hero", "a5b4fc", 0.4) updateBar("lives", -1) setPosition("hero", 960, 1040, 0) } } } }
        Entity { id("stairs") startPosition(60, 1200) color("78716c") Type { rectangle(140, 120) } }
    }

    Level {
        Entity { id("note") startPosition(60, 1300) color("d6d3d1") Type { text("FLOOR 3 - light every candle", 26) }
            Events { OnSignal("allCandles") { Actions { setText("note", "A draught... the bookcase moved!") } } } }
        Entity { id("hero") startPosition(100, 1200) color("fef08a") Type { rectangle(40, 50) }
            Movement { speed(5) Control { movementControl(0) } }
            Events { OnSignal("allCandles") { Actions { sound("bell") shake(5, 0.5) burst(20, "fb923c") } } } }

        Entity { id("wallA") startPosition(0, 900) color("44403c") solid(true) Type { rectangle(1000, 40) } }
        Entity { id("wallB") startPosition(400, 400) color("44403c") solid(true) Type { rectangle(800, 40) } }
        Entity { id("wallC") startPosition(400, 0) color("44403c") solid(true) Type { rectangle(40, 260) } }
        Entity { id("bookcase") startPosition(400, 260) color("7c2d12") solid(true) Type { rectangle(40, 140) }
            Events { OnSignal("allCandles") { Actions { sound("thud") burst(16, "a16207") setEnable(false) } } } }
        Entity { id("draught") startPosition(300, 300) color("fde68a") Type { triangle(40, "right") }
            Events { OnElapsedTime { Actions { setVisible(false) } } OnSignal("allCandles") { Actions { setVisible(true) } } } }

        Entity { id("candle1") startPosition(1100, 1300) color("57534e") Type { rectangle(20, 50) } var("lit", 0)
            Events { OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions { changeVar("lit", 1) setColor("fb923c") sound("sparkle") burst(10, "fb923c") updateBar("candles", 1) } } } } } }
        Entity { id("candle2") startPosition(1100, 960) color("57534e") Type { rectangle(20, 50) } var("lit", 0)
            Events { OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions { changeVar("lit", 1) setColor("fb923c") sound("sparkle") burst(10, "fb923c") updateBar("candles", 1) } } } } } }
        Entity { id("candle3") startPosition(80, 500) color("57534e") Type { rectangle(20, 50) } var("lit", 0)
            Events { OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions { changeVar("lit", 1) setColor("fb923c") sound("sparkle") burst(10, "fb923c") updateBar("candles", 1) } } } } } }
        Entity { id("candle4") startPosition(1100, 480) color("57534e") Type { rectangle(20, 50) } var("lit", 0)
            Events { OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions { changeVar("lit", 1) setColor("fb923c") sound("sparkle") burst(10, "fb923c") updateBar("candles", 1) } } } } } }

        Entity { id("ghost") template(true) color("b3f5f5f4") Type { circle(26) }
            Movement { speed(1.6) chase("hero") }
            Events { OnCollision("hero") { Actions { sound("hurt") shake(6, 0.3) flash("hero", "a5b4fc", 0.4) updateBar("lives", -1) burst(14, "e7e5e4") deleteEntity() } } } }
        Entity { id("clock") startPosition(600, 1250) color("a8a29e") Type { polygon(8, 40) }
            Events { OnElapsedTime(9) { repeatCount(-1) Actions { sound("bell") flash("e7e5e4", 0.5) spawn("ghost", 620, 1150) } } } }
        Entity { id("butler") color("a78bfa") Type { circle(28) }
            Movement { speed(2) Path { waypoint(100, 650) waypoint(1100, 650) dwell(0.7) } }
            Events { OnCollision("hero") { Actions { sound("hurt") shake(6, 0.3) flash("hero", "a5b4fc", 0.4) updateBar("lives", -1) setPosition("hero", 100, 1200, 0) } } } }

        goal("frontDoor", 1000, 60, 160, 160, "-- YOU ESCAPED THE MANOR --")
    }
}
