# Space Gates: slide the ship through three sectors of falling rocks. Survive each sector's timer to jump to the next.
Window {
    size(8, 12)
    background("020617")
    music("bg_5")
    lives(3, "-- SHIP LOST --", "lose")

    Level {
        paddle("ship", 360, 80, 90, 36)
        Entity { id("rock") template(true) startPosition(0, 0) color("94a3b8") Type { polygon(6, 30) }
            Movement { speed(3) AutoMove { direction(0, -1) } }
            Events {
                OnCollision("ship") { Actions { sound("hurt") updateBar("lives", -1) shake(5, 0.3) deleteEntity() } }
                OnCollision { Actions { deleteEntity() } }
            } }
        Entity { id("clock") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnElapsedTime(20) { Actions { sound("win2") nextLevel() } } } }
        Wave(20, 1.1) { Actions { spawn("rock", random(60, 740), 1180) } }
    }

    Level {
        paddle("ship", 360, 80, 90, 36)
        Entity { id("rock") template(true) startPosition(0, 0) color("94a3b8") Type { polygon(6, 30) }
            Movement { speed(3.5) AutoMove { direction(0, -1) } }
            Events {
                OnCollision("ship") { Actions { sound("hurt") updateBar("lives", -1) shake(5, 0.3) deleteEntity() } }
                OnCollision { Actions { deleteEntity() } }
            } }
        Entity { id("comet") template(true) startPosition(0, 0) color("f97316") Type { star(5, 24, 10) }
            Movement { speed(2.5) AutoMove { direction(1, -1) } }
            Events {
                OnCollision("ship") { Actions { sound("hurt") updateBar("lives", -1) deleteEntity() } }
                OnCollision { Actions { deleteEntity() } }
            } }
        Entity { id("clock") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnElapsedTime(25) { Actions { sound("win2") nextLevel() } } } }
        Wave(25, 0.9) { Actions { spawn("rock", random(60, 740), 1180) } }
        Wave(25, 4) { Actions { spawn("comet", random(20, 300), 1180) } }
    }

    Level {
        paddle("ship", 360, 80, 90, 36)
        Entity { id("rock") template(true) startPosition(0, 0) color("94a3b8") Type { polygon(6, 30) }
            Movement { speed(4) AutoMove { direction(0, -1) } }
            Events {
                OnCollision("ship") { Actions { sound("hurt") updateBar("lives", -1) shake(5, 0.3) deleteEntity() } }
                OnCollision { Actions { deleteEntity() } }
            } }
        Entity { id("comet") template(true) startPosition(0, 0) color("f97316") Type { star(5, 24, 10) }
            Movement { speed(3) AutoMove { direction(-1, -1) } }
            Events {
                OnCollision("ship") { Actions { sound("hurt") updateBar("lives", -1) deleteEntity() } }
                OnCollision { Actions { deleteEntity() } }
            } }
        Entity { id("clock") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnElapsedTime(30) { Actions { sound("win") gameOver("-- YOU REACHED THE FAR GATE --") } } } }
        Wave(30, 0.7) { Actions { spawn("rock", random(60, 740), 1180) } }
        Wave(30, 3) { Actions { spawn("comet", random(500, 780), 1180) } }
    }
}
