Window {
    size(8, 12)
    background("1a1a2e")
    music("bg_3")
    gravity(0)
    dpad("muncher")

    Bar("pellets") {
        initial(0)
        OnValue(12) { Actions { sound("win2") gameOver("-- STUFFED --") } }
    }
    Bar("lives") {
        initial(2)
        OnValue(0) { Actions { sound("lose") gameOver("-- SPLAT --") } }
    }

    Entity {
        id("muncher")
        startPosition(300, 700)
        color("fde047")
        Type { circle(40) }
        Movement { GridMove(100, 0.28) { direction(1, 0) } }
        Events {
            OnGridBlocked { Actions { sound("bump") } }
        }
    }

    Entity {
        id("pellet")
        template(true)
        startPosition(0, 0)
        color("f9a8d4")
        Type { circle(14) }
        Events {
            OnCollision("muncher") {
                Actions {
                    sound("pop")
                    updateBar("pellets", 1)
                    spawnAtEmpty("pellet", 100, 0, 300, 8, 8)
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("ghost")
        startPosition(700, 1000)
        color("a78bfa")
        Type { rectangle(60, 60) }
        Movement { speed(1.6) chase("muncher") }
        Events {
            OnCollision("muncher") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    setPosition("ghost", 700, 1000, 0)
                    shake(5, 0.3)
                }
            }
            OnElapsedTime(20) { repeatCount(-1) Actions { updateSpeed(2.6) } }
        }
    }

    Entity {
        id("keeper")
        startPosition(10, 300)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime {
                Actions {
                    spawnAtEmpty("pellet", 100, 0, 300, 8, 8)
                    spawnAtEmpty("pellet", 100, 0, 300, 8, 8)
                    spawnAtEmpty("pellet", 100, 0, 300, 8, 8)
                }
            }
        }
    }
}
