Window {
    size(9, 12)
    background("3a5a1c")
    music("bg_nature")
    gravity(0)

    Bar("tagged") {
        initial(0)
        OnValue(4) { Actions { sound("win") shake(5, 0.4) burst("it", 30, "f472b6") gameOver("-- EVERYONE IS TAGGED --") } }
    }
    Bar("time") {
        initial(45)
        OnValue(0) { Actions { sound("lose") gameOver("-- TIME'S UP --") } }
    }

    wallBox(0, 0, 900, 1200, "a16207", 20)

    # You are "it": tag all four kids before the clock runs out.
    Entity {
        id("it")
        startPosition(430, 40)
        color("f472b6")
        Type { circle(24) }
        Movement {
            speed(4.4)
            Control { movementControl(0) }
        }
        Events {
            OnSignal("tag") { Actions { flash("fbcfe8", 0.2) } }
        }
    }

    Entity { id("clock") startPosition(860, 1160) color("00000000") Type { circle(1) }
        Events { OnElapsedTime(1) { repeatCount(-1) Actions { updateBar("time", -1) } } } }

    Entity { id("tree1") startPosition(240, 520) color("14532d") solid(true) Type { circle(45) } }
    Entity { id("tree2") startPosition(600, 520) color("14532d") solid(true) Type { circle(45) } }
    Entity { id("tree3") startPosition(420, 780) color("14532d") solid(true) Type { circle(45) } }
    Entity { id("tree4") startPosition(130, 760) color("14532d") solid(true) Type { circle(40) } }
    Entity { id("tree5") startPosition(700, 760) color("14532d") solid(true) Type { circle(40) } }

    # The kids run their own loops and catch their breath at every corner.
    # Each tag makes the rest run faster.
    Entity { id("kidA") startPosition(60, 120) color("38bdf8") Type { circle(20) }
        Movement { speed(2.6) Path { waypoint(60, 120) waypoint(360, 120) waypoint(360, 400) waypoint(60, 400) dwell(0.6) } }
        Events {
            OnCollision("it") { Actions { sound("pop") updateBar("tagged", 1) burst(14, "38bdf8") broadcast("tag") setEnable(false) } }
            OnSignal("tag") { Actions {
                whenBar("tagged", "==", 1) { Actions { updateSpeed(3.1) } }
                whenBar("tagged", ">=", 2) { Actions { updateSpeed(3.6) } } } } } }
    Entity { id("kidB") startPosition(500, 150) color("facc15") Type { circle(20) }
        Movement { speed(2.8) Path { waypoint(500, 150) waypoint(820, 300) waypoint(520, 440) dwell(0.5) } }
        Events {
            OnCollision("it") { Actions { sound("pop") updateBar("tagged", 1) burst(14, "facc15") broadcast("tag") setEnable(false) } }
            OnSignal("tag") { Actions {
                whenBar("tagged", "==", 1) { Actions { updateSpeed(3.3) } }
                whenBar("tagged", ">=", 2) { Actions { updateSpeed(3.8) } } } } } }
    Entity { id("kidC") startPosition(60, 640) color("a78bfa") Type { circle(20) }
        Movement { speed(3.2) Path { waypoint(60, 640) waypoint(820, 640) dwell(1.2) } }
        Events {
            OnCollision("it") { Actions { sound("pop") updateBar("tagged", 1) burst(14, "a78bfa") broadcast("tag") setEnable(false) } }
            OnSignal("tag") { Actions {
                whenBar("tagged", "==", 1) { Actions { updateSpeed(3.7) } }
                whenBar("tagged", ">=", 2) { Actions { updateSpeed(4.2) } } } } } }
    Entity { id("kidD") startPosition(120, 880) color("fb923c") Type { circle(20) }
        Movement { speed(3) Path { waypoint(120, 880) waypoint(760, 1120) waypoint(760, 880) waypoint(120, 1120) dwell(0.4) } }
        Events {
            OnCollision("it") { Actions { sound("pop") updateBar("tagged", 1) burst(14, "fb923c") broadcast("tag") setEnable(false) } }
            OnSignal("tag") { Actions {
                whenBar("tagged", "==", 1) { Actions { updateSpeed(3.5) } }
                whenBar("tagged", ">=", 2) { Actions { updateSpeed(4) } } } } } }

    # The farmer's goose hunts whoever is "it": each peck costs five seconds.
    Entity {
        id("goose")
        startPosition(60, 1100)
        color("f8fafc")
        Type { triangle(40, "left") }
        Movement {
            speed(2)
            chase("it")
        }
        Events {
            OnCollision("it") {
                Actions {
                    sound("hurt")
                    flash("it", "ff0000", 0.3)
                    shake(5, 0.3)
                    burst(12, "f8fafc")
                    setPosition("goose", 60, 1100, 0)
                    loopCount(5)
                    updateBar("time", -1)
                }
            }
        }
    }
}
