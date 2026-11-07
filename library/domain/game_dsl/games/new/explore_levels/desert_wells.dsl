Window {
    size(8, 9)
    worldSize(40, 9)
    cameraFollow("hero")
    background("fcd34d")
    music("bg_4")
    gravity(0)

    Bar("water") {
        initial(14)
        OnValue(5) { Actions { broadcast("thirst") } }
        OnValue(0) { Actions { sound("lose") gameOver("-- DIED OF THIRST --") } }
    }

    Entity { id("tip") startPosition(1850, 820) color("78350f") Type { text("Halfway. Wells lie off the track.", 26) }
        Events { OnSignal("thirst") { Actions { setText("tip", "Thirsty! Find a well") setColor("b91c1c") } } } }

    Entity { id("hero") startPosition(400, 400) color("ffffff") Type { text("🐪", 56) } facing("left")
        Movement { speed(0) Control { movementControl(0) } }
        Events {
            OnElapsedTime(0.8) { repeatCount(-1) Actions { updateBar("water", -1) } }
            OnSignal("thirst") { Actions { sound("beep") flash("ef4444", 0.6) } } } }
    joystick("hero", 80, 3)

    Entity { id("storm") startPosition(0, 0) color("b4a16207") Type { rectangle(220, 900) }
        Movement { speed(1.1) AutoMove { direction(1, 0) } }
        hostile("hero", "-- SWALLOWED BY THE SANDSTORM --", "lose")
        Events { OnElapsedTime(0.3) { repeatCount(-1) Actions { burst(4, "d97706") } } } }

    Entity { id("dune1") startPosition(1200, 220) color("d97706") solid(true) Type { rectangle(320, 420) } }
    Entity { id("dune2") startPosition(2000, 0) color("d97706") solid(true) Type { rectangle(260, 520) } }
    Entity { id("dune3") startPosition(2900, 380) color("d97706") solid(true) Type { rectangle(320, 520) } }

    Entity { id("well1") startPosition(880, 760) color("0ea5e9") Type { ring(36, 22) } var("full", 1)
        Events { OnCollision("hero") { Actions { whenVar("full", "==", 1) { Actions { changeVar("full", -1) sound("bubble") updateBar("water", 10) burst(12, "38bdf8") setColor("78716c") } } } } } }
    Entity { id("well2") startPosition(1700, 40) color("0ea5e9") Type { ring(36, 22) } var("full", 1)
        Events { OnCollision("hero") { Actions { whenVar("full", "==", 1) { Actions { changeVar("full", -1) sound("bubble") updateBar("water", 10) burst(12, "38bdf8") setColor("78716c") } } } } } }
    Entity { id("well3") startPosition(2600, 780) color("0ea5e9") Type { ring(36, 22) } var("full", 1)
        Events { OnCollision("hero") { Actions { whenVar("full", "==", 1) { Actions { changeVar("full", -1) sound("bubble") updateBar("water", 10) burst(12, "38bdf8") setColor("78716c") } } } } } }

    Entity { id("scorpion1") color("7c2d12") Type { star(8, 22, 10) }
        Movement { speed(2.2) Path { waypoint(1600, 700) waypoint(1600, 150) dwell(0.4) } }
        Events { OnCollision("hero") { Actions { sound("hurt") shake(5, 0.3) flash("hero", "ef4444", 0.3) loopCount(3) updateBar("water", -1) } } } }
    Entity { id("scorpion2") color("7c2d12") Type { star(8, 22, 10) }
        Movement { speed(2.6) Path { waypoint(2500, 100) waypoint(3400, 700) } }
        Events { OnCollision("hero") { Actions { sound("hurt") shake(5, 0.3) flash("hero", "ef4444", 0.3) loopCount(3) updateBar("water", -1) } } } }

    goal("oasis", 3780, 330, 180, 240, "-- THE CARAVAN REACHED THE OASIS --")
}
