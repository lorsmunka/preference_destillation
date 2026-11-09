# Pick up coins on a grid. A magnet makes you big for a few seconds, a bomb costs time.
Window {
    size(8, 10)
    background("14213d")
    music("bg_nature")
    gravity(0)

    Bar("score") {
        initial(0)
        OnValue(15) {
            Actions {
                sound("win")
                gameOver("-- VAULT CLEARED --")
            }
        }
    }
    Bar("lives") {
        initial(2)
        OnValue(0) { Actions { sound("lose") gameOver("-- BLOWN UP --") } }
    }

    Entity {
        id("player")
        startPosition(380, 480)
        color("22d3ee")
        Type { circle(26) }
        Movement { speed(5) Control { movementControl(0) } }
        Events {
            OnSignal("magnetOn") {
                Actions {
                    setScale(2)
                    flash("fde047", 0.3)
                    delay(6) { Actions { setScale(1) } }
                }
            }
            OnElapsedTime(40) {
                Actions {
                    whenBar("score", "<", 15) { Actions { sound("lose") gameOver("-- TIME'S UP --") } }
                }
            }
        }
    }

    Entity {
        id("coin")
        template(true)
        startPosition(0, 0)
        color("fbbf24")
        Type { circle(16) }
        Events {
            OnCollision("player") {
                Actions {
                    sound("coin")
                    updateBar("score", 1)
                    burst(8, "fbbf24")
                    spawnAtEmpty("coin", 100, 100, 150, 6, 7)
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("magnet")
        template(true)
        startPosition(0, 0)
        color("f43f5e")
        Type { cross(40, 40, 14) }
        Events {
            OnCollision("player") {
                Actions {
                    sound("sparkle")
                    broadcast("magnetOn")
                    deleteEntity()
                }
            }
            OnElapsedTime(7) { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("bomb")
        template(true)
        startPosition(0, 0)
        color("94a3b8")
        Type { polygon(8, 24) }
        Events {
            OnCollision("player") {
                Actions {
                    sound("hurt")
                    shake(6, 0.3)
                    updateBar("lives", -1)
                    deleteEntity()
                }
            }
            OnElapsedTime(8) { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("director")
        startPosition(10, 10)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime { Actions {
                spawnAtEmpty("coin", 100, 100, 150, 6, 7)
                spawnAtEmpty("coin", 100, 100, 150, 6, 7)
                spawnAtEmpty("coin", 100, 100, 150, 6, 7)
            } }
            OnElapsedTime(5) {
                repeatCount(-1)
                Actions {
                    randomAction {
                        Actions { spawnAtEmpty("magnet", 100, 100, 150, 6, 7) }
                        Actions { spawnAtEmpty("bomb", 100, 100, 150, 6, 7) }
                        Actions { spawnAtEmpty("bomb", 100, 100, 150, 6, 7) }
                    }
                }
            }
        }
    }
}
