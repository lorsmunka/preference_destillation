Window {
    size(9, 9)
    background("1c1917")
    music("bg_3")
    gravity(0)

    lives(3, "-- CRUSHED --", "lose")
    Bar("parts") { initial(0) }
    winAt("parts", 60, "-- SHIFT COMPLETE --", "win")

    Entity { id("robot") startPosition(425, 30) color("38bdf8")
        Type { rectangle(44, 60) }
        Movement { speed(5) Control { movementControl(2) } } }

    # the belt drags the robot right; after 20 parts it reverses and runs faster
    Entity { id("belt") startPosition(0, 0) color("44403c")
        Type { rectangle(900, 30) }
        Events {
            OnElapsedTime(0.1) { repeatCount(-1) Actions {
                whenBar("parts", "<", 20) { Actions { setVelocity("robot", 70, 0) } }
                whenBar("parts", ">=", 20) { Actions { setVelocity("robot", -120, 0) } }
            } }
            OnElapsedTime(0.5) { repeatCount(-1) Actions { burst(2, "a8a29e") } }
        } }

    Entity { id("p1") startPosition(25, 700) color("78716c") Type { rectangle(140, 150) }
        Events {
            OnElapsedTime(random(2.5, 5)) { repeatCount(-1) Actions {
                setColor("ef4444") sound("beep")
                delay(0.8) { Actions { setPosition("p1", 25, 30, 0.15)
                    delay(0.15) { Actions { sound("thud") shake(4, 0.15) burst(10, "fbbf24") updateBar("parts", 1) } }
                    delay(0.6) { Actions { setPosition("p1", 25, 700, 0.5) setColor("78716c") } } } } } }
            OnCollision("robot") { Actions { sound("hurt") updateBar("lives", -1) flash("robot", "ff0000", 0.4) shake(9, 0.4) } }
        } }
    Entity { id("p2") startPosition(200, 700) color("78716c") Type { rectangle(140, 150) }
        Events {
            OnElapsedTime(random(2.5, 5)) { repeatCount(-1) Actions {
                setColor("ef4444") sound("beep")
                delay(0.8) { Actions { setPosition("p2", 200, 30, 0.15)
                    delay(0.15) { Actions { sound("thud") shake(4, 0.15) burst(10, "fbbf24") updateBar("parts", 1) } }
                    delay(0.6) { Actions { setPosition("p2", 200, 700, 0.5) setColor("78716c") } } } } } }
            OnCollision("robot") { Actions { sound("hurt") updateBar("lives", -1) flash("robot", "ff0000", 0.4) shake(9, 0.4) } }
        } }
    Entity { id("p3") startPosition(375, 700) color("78716c") Type { rectangle(140, 150) }
        Events {
            OnElapsedTime(random(2.5, 5)) { repeatCount(-1) Actions {
                setColor("ef4444") sound("beep")
                delay(0.8) { Actions { setPosition("p3", 375, 30, 0.15)
                    delay(0.15) { Actions { sound("thud") shake(4, 0.15) burst(10, "fbbf24") updateBar("parts", 1) } }
                    delay(0.6) { Actions { setPosition("p3", 375, 700, 0.5) setColor("78716c") } } } } } }
            OnCollision("robot") { Actions { sound("hurt") updateBar("lives", -1) flash("robot", "ff0000", 0.4) shake(9, 0.4) } }
        } }
    Entity { id("p4") startPosition(550, 700) color("78716c") Type { rectangle(140, 150) }
        Events {
            OnElapsedTime(random(2.5, 5)) { repeatCount(-1) Actions {
                setColor("ef4444") sound("beep")
                delay(0.8) { Actions { setPosition("p4", 550, 30, 0.15)
                    delay(0.15) { Actions { sound("thud") shake(4, 0.15) burst(10, "fbbf24") updateBar("parts", 1) } }
                    delay(0.6) { Actions { setPosition("p4", 550, 700, 0.5) setColor("78716c") } } } } } }
            OnCollision("robot") { Actions { sound("hurt") updateBar("lives", -1) flash("robot", "ff0000", 0.4) shake(9, 0.4) } }
        } }
    Entity { id("p5") startPosition(725, 700) color("78716c") Type { rectangle(140, 150) }
        Events {
            OnElapsedTime(random(2.5, 5)) { repeatCount(-1) Actions {
                setColor("ef4444") sound("beep")
                delay(0.8) { Actions { setPosition("p5", 725, 30, 0.15)
                    delay(0.15) { Actions { sound("thud") shake(4, 0.15) burst(10, "fbbf24") updateBar("parts", 1) } }
                    delay(0.6) { Actions { setPosition("p5", 725, 700, 0.5) setColor("78716c") } } } } } }
            OnCollision("robot") { Actions { sound("hurt") updateBar("lives", -1) flash("robot", "ff0000", 0.4) shake(9, 0.4) } }
        } }

    # sparks fly off the presses once the line is running hot
    Entity { id("forge") startPosition(0, 860) color("7c2d12") Type { rectangle(900, 40) }
        Events { OnElapsedTime(1.5) { repeatCount(-1) Actions {
            whenBar("parts", ">=", 30) { Actions { sound("glass") spawn("spark", random(20, 860), 840) } } } } } }
    Entity { id("spark") template(true) startPosition(0, 0) color("fb923c") Type { circle(10) }
        Movement { speed(4) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("robot") { Actions { sound("hurt") updateBar("lives", -1) flash("robot", "fb923c", 0.3) deleteEntity() } }
            OnCollision { Actions { burst(5, "fdba74") deleteEntity() } }
        } }
}
