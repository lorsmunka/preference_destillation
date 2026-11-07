Window {
    size(12, 7)
    background("0c4a6e")
    music("bg_3")
    gravity(0)
    lives(3, "-- SHARK SNACK --", "lose")
    Bar("dodged") { initial(0) }
    winAt("dodged", 30, "-- OUTSWAM THE PACK --", "win")

    Entity { id("minnow") startPosition(90, 330) color("fde047") Type { triangle(36, "right") }
        Movement { speed(5) Control { movementControl(1) } } }

    Entity { id("reef") startPosition(1150, 0) color("be185d") Type { rectangle(50, 700) }
        Events { OnElapsedTime(random(1.0, 1.8)) { repeatCount(-1) Actions {
            spawn("shark", 1080, random(10, 630))
            whenBar("dodged", ">=", 10) { Actions { randomAction {
                Actions { spawn("shark", 1080, random(10, 630)) }
                Actions { } } } } } } } }

    # a shark creeps out of the reef, turns red, then dashes straight across its lane
    Entity { id("shark") template(true) startPosition(0, 0) color("94a3b8") Type { triangle(60, "left") }
        Movement { speed(0.4) AutoMove { direction(-1, 0) } }
        Events {
            Sequence {
                Phase(1, 1.4) { Actions { setColor("ef4444") sound("beep") } }
                Phase(1, 0.4) { Actions {
                    sound("laser") updateSpeed(9)
                    whenBar("dodged", ">=", 15) { Actions { updateSpeed(12) } } } }
            }
            OnCollision("minnow") { Actions {
                sound("hurt") updateBar("lives", -1) flash("minnow", "ff0000", 0.4)
                shake(8, 0.35) burst(16, "fde047") deleteEntity() } }
            OnCollision { Actions { updateBar("dodged", 1) burst(6, "7dd3fc") deleteEntity() } }
        } }

    Entity { id("bubbles") startPosition(0, 0) color("00000000") Type { circle(1) }
        Events { OnElapsedTime(0.7) { repeatCount(-1) Actions { spawn("bubble", random(150, 1100), 0) } } } }
    Entity { id("bubble") template(true) startPosition(0, 0) color("667dd3fc") Type { circle(6) }
        Movement { speed(1.2) AutoMove { direction(0, 1) } }
        Events { OnCollision { Actions { deleteEntity() } } } }
}
