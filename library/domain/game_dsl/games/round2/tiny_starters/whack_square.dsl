Window {
    size(8, 10)
    background("3b0764")
    winAt("hits", 10, "-- WHACKED 10 --", "win")
    lives(3, "-- THEY ESCAPED --", "lose")
    Bar("hits") { initial(0) }

    Entity {
        id("mole")
        template(true)
        startPosition(0, 0)
        color("fbbf24")
        Type { rectangle(110, 110) }
        Events {
            OnTapped {
                Actions {
                    sound("pop")
                    updateBar("hits", 1)
                    burst(8, "fbbf24")
                    deleteEntity()
                }
            }
            OnElapsedTime(1.4) { Actions { updateBar("lives", -1) deleteEntity() } }
        }
    }

    Entity {
        id("bomb")
        template(true)
        startPosition(0, 0)
        color("1f2937")
        Type { circle(50) }
        Events {
            OnTapped { Actions { sound("hurt") updateBar("lives", -1) deleteEntity() } }
            OnElapsedTime(1.4) { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("popper")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1.0) {
                repeatCount(-1)
                Actions {
                    spawn("mole", random(40, 650), random(120, 800))
                    spawn("bomb", random(40, 650), random(120, 800))
                }
            }
        }
    }
}
