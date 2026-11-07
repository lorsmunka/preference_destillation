# Hot-air balloon festival: pump the burner faster than the envelope cools.
Window {
    size(6, 10)
    background("7dd3fc")
    music("bg_nature")

    Bar("heat") {
        initial(5)
        OnValue(0) { Actions { sound("lose") shake(6, 0.4) gameOver("-- THE FLAME WENT OUT --") } }
        OnValue(9) { Actions { setScale("balloon", 1) } }
        OnValue(10) { Actions { sound("blip") setScale("balloon", 1.2) } }
        OnValue(19) { Actions { setScale("balloon", 1.2) } }
        OnValue(20) { Actions { sound("blip") setScale("balloon", 1.4) flash("balloon", "fde68a", 0.3) } }
    }
    winAt("heat", 30, "-- LIFT OFF --", "win")

    Entity { id("balloon") startPosition(210, 520) color("ef4444") Type { circle(90) } }
    Entity { id("basket") startPosition(265, 400) color("92400e") Type { rectangle(70, 50) } }

    Entity { id("flame") startPosition(285, 455) color("f97316") Type { triangle(30, "up") }
        Events {
            OnElapsedTime(0.5) { repeatCount(-1) Actions { updateBar("heat", -1) setOpacity(random(0.4, 1)) } }
        } }

    Entity { id("pump") startPosition(180, 60) color("dc2626") Type { rectangle(240, 110) }
        Events {
            OnTapped("pump") { Actions {
                sound("click")
                updateBar("heat", 1)
                flash("flame", "fde047", 0.15)
                burst("flame", 5, "fb923c")
            } }
        } }
    Entity { id("pumpLabel") startPosition(205, 95) color("ffffff") Type { text("BURNER", 40) } }

    Entity { id("cloud") startPosition(20, 860) color("f8fafc") Type { rectangle(150, 50) }
        Movement { speed(1) Path { waypoint(20, 860) waypoint(430, 860) } }
        Events {
            OnElapsedTime(random(2.5, 4.5)) { repeatCount(-1) Actions {
                randomAction {
                    Actions { sound("buzz") setColor("64748b") flash("balloon", "94a3b8", 0.3) loopCount(2) updateBar("heat", -1) }
                    Actions { setColor("cbd5e1") updateBar("heat", -1) }
                    Actions { setColor("f8fafc") }
                }
            } }
        } }

    Entity { id("clock") startPosition(30, 930) color("0f172a") Type { text("30", 48) }
        Events {
            Sequence {
                Phase(1, 5) { Actions { setText("clock", "25") } }
                Phase(1, 5) { Actions { setText("clock", "20") } }
                Phase(1, 5) { Actions { setText("clock", "15") } }
                Phase(1, 5) { Actions { setText("clock", "10") setColor("b91c1c") } }
                Phase(1, 5) { Actions { setText("clock", "5") sound("beep") } }
                Phase(1, 5) { Actions { sound("lose") shake(5, 0.3) gameOver("-- THE FESTIVAL LEFT WITHOUT YOU --") } }
            }
        } }
}
