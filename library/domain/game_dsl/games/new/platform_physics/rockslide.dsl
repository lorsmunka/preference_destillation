Window {
    size(10, 7)
    background("fed7aa")
    music("bg_5")
    gravity(700)

    lives(3, "-- BURIED BY THE SLIDE --", "lose")
    Bar("dodged") { initial(0) }
    winAt("dodged", 25, "-- SAFE AT THE PASS --", "win2")

    Entity {
        id("trail")
        startPosition(0, 0)
        color("78350f")
        solid(true)
        Type { rectangle(1000, 40) }
    }

    Entity {
        id("ranger")
        startPosition(470, 40)
        color("ffffff")
        facing("left")
        Type { text("🏃", 46) }
        Movement { speed(5) Control { movementControl(2) } }
    }

    # the whole sky is the jump button; one jump per 1.6 s (a full hop)
    Entity {
        id("sky")
        startPosition(0, 330)
        color("00000000")
        Type { rectangle(1000, 370) }
        var("ready", 1)
        Events {
            OnTapped("sky") {
                Actions {
                    whenVar("ready", "==", 1) {
                        Actions {
                            sound("jump")
                            setVelocity("ranger", 0, 560)
                            burst("ranger", 6, "fef3c7")
                            changeVar("ready", -1)
                            delay(1.6) { Actions { changeVar("ready", 1) } }
                        }
                    }
                }
            }
        }
    }
    Entity { id("hint") startPosition(330, 640) color("9a3412")
        Type { text("drag to run - tap the sky to jump", 24) } }

    Entity {
        id("boulder")
        template(true)
        startPosition(0, 0)
        color("57534e")
        Type { circle(30) }
        Movement { speed(2.4) AutoMove { direction(-1, 0) } }
        Events {
            OnCollision("trail") { Actions { sound("thud") burst(6, "a8a29e") bounceBack(0) } }
            OnCollision("ranger") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("ranger", "ff0000", 0.3)
                    shake(8, 0.4)
                    burst(16, "57534e")
                    deleteEntity()
                }
            }
            OnCollision { Actions { updateBar("dodged", 1) deleteEntity() } }
        }
    }

    Entity {
        id("scree")
        template(true)
        startPosition(0, 0)
        color("a16207")
        Type { circle(16) }
        Movement { speed(3.2) AutoMove { direction(1, 0) } }
        Events {
            OnCollision("trail") { Actions { sound("bump") bounceBack(0) } }
            OnCollision("ranger") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("ranger", "ff0000", 0.3)
                    shake(5, 0.3)
                    burst(10, "a16207")
                    deleteEntity()
                }
            }
            OnCollision { Actions { updateBar("dodged", 1) deleteEntity() } }
        }
    }

    Entity {
        id("peak")
        startPosition(860, 560)
        color("7c2d12")
        Type { triangle(70, "up") }
        Events {
            Sequence {
                Phase(8, 2.2) { Actions { flash("fbbf24", 0.2) spawn("boulder", 930, random(380, 560)) } }
                Phase(10, 1.6) { Actions { flash("fbbf24", 0.2) randomAction {
                    Actions { spawn("boulder", 930, random(380, 560)) }
                    Actions { spawn("scree", 10, random(150, 260)) } } } }
                Phase(-1, 1.1) { Actions { flash("fbbf24", 0.2) shake(2, 0.2) randomAction {
                    Actions { spawn("boulder", 930, random(380, 560)) }
                    Actions { spawn("scree", 10, random(150, 260)) }
                    Actions { spawn("scree", 10, random(150, 260)) } } } }
            }
        }
    }
}
