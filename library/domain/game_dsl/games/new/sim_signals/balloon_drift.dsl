# Balloon Drift: the air in the envelope cools and the balloon sinks one level every few seconds.
# Swipe up to fire the burner (costs fuel), swipe down to vent, dodge the gulls and grab fuel canisters.
Window {
    size(7, 12)
    background("7dd3fc")
    music("bg_2")
    gravity(0)

    Bar("altitude") { initial(5)
        OnValue(0) { Actions { sound("death") shake(10, 0.6) burst("balloon", 30, "f97316") gameOver("-- THE BASKET HIT THE GROUND --") } }
        OnValue(1) { Actions { setPosition("balloon", 140, 160, 0.4) sound("beep") } }
        OnValue(2) { Actions { setPosition("balloon", 140, 260, 0.4) } }
        OnValue(3) { Actions { setPosition("balloon", 140, 360, 0.4) } }
        OnValue(4) { Actions { setPosition("balloon", 140, 460, 0.4) } }
        OnValue(5) { Actions { setPosition("balloon", 140, 560, 0.4) } }
        OnValue(6) { Actions { setPosition("balloon", 140, 660, 0.4) } }
        OnValue(7) { Actions { setPosition("balloon", 140, 760, 0.4) } }
        OnValue(8) { Actions { setPosition("balloon", 140, 860, 0.4) } }
        OnValue(9) { Actions { setPosition("balloon", 140, 960, 0.4) } }
        OnValue(10) { Actions { setPosition("balloon", 140, 1060, 0.4) } } }
    Bar("fuel") { initial(24) }
    Bar("distance") { initial(0)
        OnValue(25) { Actions { broadcast("dusk") } }
        OnValue(50) { Actions { sound("win") burst("balloon", 30, "fde047") gameOver("-- LANDED SAFE ACROSS THE VALLEY --") } } }

    Entity { id("sky") startPosition(0, 0) color("38bdf8") Type { rectangle(700, 1200) }
        Events { OnSignal("dusk") { Actions { setColor("c2410c") } } } }
    Entity { id("sun") startPosition(520, 1020) color("fde047") Type { circle(60) }
        Events { OnSignal("dusk") { Actions { setOpacity(0.5) setPosition("sun", 520, 120, 20) } } } }
    Entity { id("gauge") startPosition(20, 20) color("0f172a") Type { text("FUEL OK", 32) }
        Events { OnElapsedTime(0.5) { repeatCount(-1) Actions {
            whenBar("fuel", "<=", 6) { Actions { setText("gauge", "FUEL LOW") setColor("b91c1c") } }
            whenBar("fuel", ">", 6) { Actions { setText("gauge", "FUEL OK") setColor("0f172a") } } } } } }

    Entity { id("balloon") startPosition(140, 560) color("f97316") Type { circle(55) }
        Events {
            OnElapsedTime { Actions { ramp("distance", 0, 50, 70) } }
            OnElapsedTime(2.4) { repeatCount(-1) Actions { updateBar("altitude", -1) } }
            OnSwipe("up") { Actions { whenBar("fuel", ">", 0) { Actions { whenBar("altitude", "<", 10) { Actions {
                updateBar("fuel", -1) updateBar("altitude", 1) sound("jump") burst(8, "fbbf24")
                setScale("balloon", 1.12) delay(0.25) { Actions { setScale("balloon", 1) } } } } } } } }
            OnSwipe("down") { Actions { whenBar("altitude", ">", 1) { Actions { updateBar("altitude", -1) sound("pop") burst(5, "e2e8f0") } } } }
            OnCollision("gull") { Actions { sound("thud") updateBar("altitude", -1) flash("ef4444", 0.25) shake(5, 0.3) deleteCollider() } }
            OnCollision("canister") { Actions { sound("pickup") updateBar("fuel", 4) burst(10, "22c55e") deleteCollider() } }
        } }

    Entity { id("wind") startPosition(0, 0) color("00000000") Type { circle(1) }
        Events {
            Sequence {
                Phase(6, 3) { Actions { spawn("gull", 640, random(150, 1100)) } }
                Phase(-1, 1.8) { Actions { spawn("gull", 640, random(150, 1100))
                    randomAction { Actions { } Actions { } Actions { spawn("gull", 640, random(150, 1100)) } } } }
            }
            OnElapsedTime(6.5) { repeatCount(-1) Actions { spawn("canister", 640, random(200, 1050)) } }
        } }

    Entity { id("gull") template(true) color("f8fafc") Type { triangle(40, "left") }
        Movement { speed(2.5) AutoMove { direction(-1, 0) } }
        Events { OnCollision { Actions { deleteEntity() } } } }
    Entity { id("canister") template(true) color("16a34a") Type { rectangle(30, 44) }
        Movement { speed(1.6) AutoMove { direction(-1, 0) } }
        Events { OnCollision { Actions { deleteEntity() } } } }
}
