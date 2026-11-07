Window {
    size(8, 12)
    background("14532d")
    music("bg_nature")
    gravity(0)

    Bar("weeds") {
        initial(0)
        OnValue(14) {
            Actions {
                sound("lose")
                shake(6, 0.5)
                gameOver("-- THE WEEDS TOOK THE GARDEN --")
            }
        }
    }
    Bar("pulled") {
        initial(0)
    }
    winAt("pulled", 25, "-- PRIZE GARDEN --", "win")
    lives(3, "-- STUNG ONE TOO MANY TIMES --", "lose")

    dpad("gardener")

    # A weed left alone for 7 seconds drops seed into another empty bed.
    Entity {
        id("weed")
        template(true)
        startPosition(0, 0)
        color("84cc16")
        Type { star(6, 50, 20) }
        Events {
            OnCollision("gardener") {
                Actions {
                    sound("pop")
                    burst(10, "a3e635")
                    updateBar("weeds", -1)
                    updateBar("pulled", 1)
                    deleteEntity()
                }
            }
            OnElapsedTime(7) {
                repeatCount(-1)
                Actions {
                    setColor("3f6212")
                    spawnAtEmpty("weed", 100, 0, 400, 8, 8)
                    updateBar("weeds", 1)
                }
            }
        }
    }

    Entity {
        id("sower")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(12, 2.0) { Actions { spawnAtEmpty("weed", 100, 0, 400, 8, 8) updateBar("weeds", 1) } }
                Phase(-1, 1.3) { Actions { spawnAtEmpty("weed", 100, 0, 400, 8, 8) updateBar("weeds", 1) } }
            }
        }
    }

    Entity {
        id("wasp")
        startPosition(100, 600)
        color("facc15")
        Type { diamond(50, 50) }
        Movement {
            speed(1.5)
            Path {
                waypoint(100, 600)
                waypoint(650, 600)
                waypoint(650, 1000)
                waypoint(100, 1000)
            }
        }
    }

    Entity {
        id("gardener")
        startPosition(10, 410)
        color("fde047")
        Type { circle(40) }
        Movement { GridMove(100, 0.25) { direction(0, 0) } }
        Events {
            OnGridBlocked {
                Actions {
                    sound("bump")
                    setDirection("gardener", 0, 0)
                }
            }
            OnCollision("wasp") {
                Actions {
                    sound("hurt")
                    shake(5, 0.3)
                    flash("ef4444", 0.3)
                    burst(12, "facc15")
                    updateBar("lives", -1)
                    setPosition("gardener", 10, 410, 0)
                    setDirection("gardener", 0, 0)
                }
            }
        }
    }
}
