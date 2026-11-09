Window {
    size(6, 12)
    background("111827")
    music("bg_2")
    gravity(0)

    lives(3, "-- OUT OF CARS --", "lose")
    Bar("laps") { initial(0) }
    Bar("pace") {
        initial(0)
        OnValue(5) { Actions { broadcast("faster") } }
        OnValue(10) { Actions { broadcast("faster") } }
    }
    winAt("laps", 20, "-- CHAMPION --", "win")

    Entity { id("roadA") startPosition(200, 0) color("374151")
        Type { rectangle(8, 1200) } }
    Entity { id("roadB") startPosition(400, 0) color("374151")
        Type { rectangle(8, 1200) } }

    Entity {
        id("car")
        startPosition(240, 90)
        color("38bdf8")
        Type { rectangle(60, 110) }
        var("lane", 1)
        Events {
            OnSwipe("left") {
                Actions {
                    whenVar("lane", "==", 1) { Actions {
                        changeVar("lane", -1)
                        setPosition("car", 40, 90, 0.12)
                    } }
                    whenVar("lane", "==", 2) { Actions {
                        changeVar("lane", -1)
                        setPosition("car", 240, 90, 0.12)
                    } }
                }
            }
            OnSwipe("right") {
                Actions {
                    whenVar("lane", "==", 1) { Actions {
                        changeVar("lane", 1)
                        setPosition("car", 440, 90, 0.12)
                    } }
                    whenVar("lane", "==", 0) { Actions {
                        changeVar("lane", 1)
                        setPosition("car", 240, 90, 0.12)
                    } }
                }
            }
            OnCollision("van") {
                Actions {
                    sound("hurt")
                    shake(7, 0.35)
                    flash("ff0000", 0.3)
                    updateBar("lives", -1)
                }
            }
        }
    }

    Entity {
        id("van")
        template(true)
        startPosition(0, 0)
        color("facc15")
        Type { rectangle(70, 120) }
        Movement { speed(4) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("car") { Actions { deleteEntity() } }
            OnCollision {
                Actions {
                    updateBar("laps", 1)
                    updateBar("pace", 1)
                    deleteEntity()
                }
            }
            OnSignal("faster") { Actions { updateSpeed(6) } }
        }
    }

    Entity {
        id("director")
        startPosition(300, 20)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1.2) {
                repeatCount(-1)
                Actions {
                    randomAction {
                        Actions { spawn("van", 35, 1050) }
                        Actions { spawn("van", 235, 1050) }
                        Actions { spawn("van", 435, 1050) }
                    }
                }
            }
        }
    }
}
