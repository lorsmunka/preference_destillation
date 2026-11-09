# Orchard Defender: tap the crows before they eat the apples. Score climbs through three orchards.
Window {
    size(8, 12)
    background("365314")
    music("bg_nature")
    lives(4, "-- THE ORCHARD IS BARE --", "lose")
    Bar("score") {
        initial(0)
        OnValue(8) { Actions { sound("win2") nextLevel() } }
        OnValue(20) { Actions { sound("win2") nextLevel() } }
        OnValue(36) { Actions { sound("win") gameOver("-- THE ORCHARD IS SAFE --") } }
    }

    Level {
        Entity { id("crow") template(true) startPosition(0, 0) color("1e1b4b") Type { triangle(44, "down") }
            Movement { speed(2) AutoMove { direction(0, -1) } }
            Events {
                OnTapped { Actions { sound("pop") updateBar("score", 1) burst(8, "a78bfa") deleteEntity() } }
                OnCollision { Actions { sound("hurt") updateBar("lives", -1) deleteEntity() } }
            } }
        Entity { id("ground") startPosition(0, 0) color("a16207") Type { rectangle(800, 20) } }
        Entity { id("spawner") startPosition(400, 600) color("00000000") Type { circle(1) }
            Events { OnElapsedTime(1.6) { repeatCount(-1) Actions { spawn("crow", random(60, 700), 1160) } } } }
    }

    Level {
        Entity { id("crow") template(true) startPosition(0, 0) color("1e1b4b") Type { triangle(44, "down") }
            Movement { speed(2.8) AutoMove { direction(0, -1) } }
            Events {
                OnTapped { Actions { sound("pop") updateBar("score", 1) burst(8, "a78bfa") deleteEntity() } }
                OnCollision { Actions { sound("hurt") updateBar("lives", -1) deleteEntity() } }
            } }
        Entity { id("swoop") template(true) startPosition(0, 0) color("7f1d1d") Type { triangle(36, "down") }
            Movement { speed(2) AutoMove { direction(1.5, -1) } }
            Events {
                OnTapped { Actions { sound("pop") updateBar("score", 2) burst(10, "f87171") deleteEntity() } }
                OnCollision { Actions { deleteEntity() } }
            } }
        Entity { id("ground") startPosition(0, 0) color("a16207") Type { rectangle(800, 20) } }
        Entity { id("spawner") startPosition(400, 600) color("00000000") Type { circle(1) }
            Events {
                OnElapsedTime(1.2) { repeatCount(-1) Actions { spawn("crow", random(60, 700), 1160) } }
                OnElapsedTime(3.5) { repeatCount(-1) Actions { spawn("swoop", random(20, 200), 1160) } }
            } }
    }

    Level {
        Entity { id("crow") template(true) startPosition(0, 0) color("1e1b4b") Type { triangle(44, "down") }
            Movement { speed(3.4) AutoMove { direction(0, -1) } }
            Events {
                OnTapped { Actions { sound("pop") updateBar("score", 1) burst(8, "a78bfa") deleteEntity() } }
                OnCollision { Actions { sound("hurt") updateBar("lives", -1) deleteEntity() } }
            } }
        Entity { id("swoop") template(true) startPosition(0, 0) color("7f1d1d") Type { triangle(36, "down") }
            Movement { speed(2.6) AutoMove { direction(-1.5, -1) } }
            Events {
                OnTapped { Actions { sound("pop") updateBar("score", 2) burst(10, "f87171") deleteEntity() } }
                OnCollision { Actions { deleteEntity() } }
            } }
        Entity { id("raven") template(true) startPosition(0, 0) color("0f172a") Type { circle(34) }
            var("hp", 2)
            Movement { speed(1.6) AutoMove { direction(0, -1) } }
            Events {
                OnTapped { Actions { changeVar("hp", -1) flash("ffffff", 0.15) whenVar("hp", "<=", 0) { Actions { sound("pop") updateBar("score", 3) burst(14, "e2e8f0") deleteEntity() } } } }
                OnCollision { Actions { sound("hurt") updateBar("lives", -1) deleteEntity() } }
            } }
        Entity { id("ground") startPosition(0, 0) color("a16207") Type { rectangle(800, 20) } }
        Entity { id("spawner") startPosition(400, 600) color("00000000") Type { circle(1) }
            Events {
                OnElapsedTime(0.9) { repeatCount(-1) Actions { spawn("crow", random(60, 700), 1160) } }
                OnElapsedTime(2.8) { repeatCount(-1) Actions { spawn("swoop", random(600, 780), 1160) } }
                OnElapsedTime(6) { repeatCount(-1) Actions { spawn("raven", random(100, 650), 1160) } }
            } }
    }
}
