Window {
    size(10, 8)
    background("18181b")
    music("bg_5")
    gravity(0)

    lives(3, "-- DROWNED OUT --", "lose")
    Bar("volume") { initial(0) }
    winAt("volume", 10, "-- ENCORE! --", "win")

    Entity { id("singer") startPosition(480, 380) color("f472b6")
        Type { circle(22) }
        Movement { speed(5.5) Control { movementControl(0) } }
        Events { OnElapsedTime { Actions { ramp("volume", 0, 10, 60) } } } }

    # amp stacks: solid cover that blocks both the singer and the squeals
    Entity { id("amp1") startPosition(250, 230) color("3f3f46") solid(true) Type { rectangle(70, 130) } }
    Entity { id("amp2") startPosition(680, 440) color("3f3f46") solid(true) Type { rectangle(70, 130) } }
    Entity { id("amp3") startPosition(440, 640) color("3f3f46") solid(true) Type { rectangle(130, 50) } }

    Entity { id("spkA") startPosition(0, 0) color("52525b") Type { rectangle(80, 80) }
        Events { OnElapsedTime(random(3, 5)) { repeatCount(-1) Actions {
            flash("a3e635", 0.2) sound("buzz") spawnHere("squeal") } } } }
    Entity { id("spkB") startPosition(920, 720) color("52525b") Type { rectangle(80, 80) }
        Events { OnElapsedTime(random(3, 5)) { repeatCount(-1) Actions {
            whenBar("volume", ">=", 2) { Actions { flash("a3e635", 0.2) sound("buzz") spawnHere("squeal") } } } } } }
    Entity { id("spkC") startPosition(920, 0) color("52525b") Type { rectangle(80, 80) }
        Events { OnElapsedTime(random(3, 5)) { repeatCount(-1) Actions {
            whenBar("volume", ">=", 4) { Actions { flash("a3e635", 0.2) sound("buzz") spawnHere("squeal") } } } } } }
    Entity { id("spkD") startPosition(0, 720) color("52525b") Type { rectangle(80, 80) }
        Events { OnElapsedTime(random(3, 5)) { repeatCount(-1) Actions {
            whenBar("volume", ">=", 6) { Actions { flash("a3e635", 0.2) sound("buzz") spawnHere("squeal") } } } } } }

    Entity {
        id("squeal")
        template(true)
        startPosition(0, 0)
        color("a3e635")
        Type { star(6, 18, 8) }
        Movement { speed(2.2) chase("singer") }
        Events {
            OnElapsedTime {
                Actions {
                    whenBar("volume", ">=", 5) { Actions { updateSpeed(3) } }
                    whenBar("volume", ">=", 8) { Actions { updateSpeed(3.6) } }
                }
            }
            OnCollision("singer") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("singer", "ff0000", 0.3)
                    shake(6, 0.3)
                    burst(14, "a3e635")
                    deleteEntity()
                }
            }
            OnElapsedTime(5.5) {
                Actions {
                    sound("pop")
                    burst(8, "d9f99d")
                    deleteEntity()
                }
            }
        }
    }
}
