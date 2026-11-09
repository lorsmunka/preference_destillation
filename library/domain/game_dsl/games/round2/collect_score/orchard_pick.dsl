# A tree swings along the top and drops apples. Catch the red ones, dodge the rotten.
Window {
    size(8, 12)
    background("365314")
    music("bg_nature")
    gravity(0)

    Bar("apples") { initial(0) }
    winAt("apples", 18, "-- BASKET FULL --", "win2")
    Bar("lives") {
        initial(3)
        OnValue(0) { Actions { sound("lose") gameOver("-- TOO MANY WORMS --") } }
    }

    Entity {
        id("basket")
        startPosition(340, 40)
        color("b45309")
        Type { rectangle(120, 50) }
        Movement { speed(6.5) Control { movementControl(2) } }
    }

    Entity {
        id("tree")
        startPosition(60, 1050)
        color("166534")
        Type { circle(60) }
        Movement { speed(2.5) Path { waypoint(60, 1050) waypoint(620, 1050) } }
        Events {
            OnSignal("treeFaster") { Actions { updateSpeed(4) } }
            OnElapsedTime(1.2) {
                repeatCount(-1)
                Actions {
                    randomAction {
                        Actions { spawnHere("apple") }
                        Actions { spawnHere("apple") }
                        Actions { spawnHere("worm") }
                    }
                }
            }
        }
    }

    Entity {
        id("apple")
        template(true)
        startPosition(0, 0)
        color("ef4444")
        Type { circle(20) }
        Movement { speed(2.8) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("basket") {
                Actions {
                    sound("pickup")
                    updateBar("apples", 1)
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("worm")
        template(true)
        startPosition(0, 0)
        color("84cc16")
        Type { circle(16) }
        Movement { speed(3.2) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("basket") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("basket", "ef4444", 0.3)
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("pace")
        startPosition(10, 10)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime { Actions { ramp("round", 1, 5, 40) } }
        }
    }
    Bar("round") {
        initial(1)
        OnValue(3) { Actions { sound("bell") broadcast("treeFaster") } }
    }
}
