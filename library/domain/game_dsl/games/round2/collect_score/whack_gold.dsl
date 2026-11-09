# Tap the gold nuggets before they crumble. Rocks cost a life. The grid fills as time goes on.
Window {
    size(8, 10)
    background("3b2a1a")
    music("bg_3")
    gravity(0)

    Bar("gold") {
        initial(0)
        OnValue(20) { Actions { sound("win") gameOver("-- RICH! --") } }
    }
    lives(3, "-- THE MINE IS EMPTY --", "lose")

    Entity {
        id("nugget")
        template(true)
        startPosition(0, 0)
        color("fbbf24")
        Type { star(5, 38, 18) }
        Events {
            OnTapped {
                Actions {
                    sound("coin")
                    updateBar("gold", 1)
                    burst(10, "fde047")
                    deleteEntity()
                }
            }
            OnElapsedTime(2.2) {
                Actions {
                    sound("buzz")
                    updateBar("lives", -1)
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("rock")
        template(true)
        startPosition(0, 0)
        color("78716c")
        Type { polygon(7, 36) }
        Events {
            OnTapped {
                Actions {
                    sound("thud")
                    updateBar("lives", -1)
                    shake(5, 0.2)
                    deleteEntity()
                }
            }
            OnElapsedTime(2.5) { Actions { deleteEntity() } }
        }
    }

    Wave(12, 1.1) {
        Actions { spawnAtEmpty("nugget", 160, 80, 120, 5, 6) }
    }
    Wave(12, 0.8) {
        Actions {
            spawnAtEmpty("nugget", 160, 80, 120, 5, 6)
            spawnAtEmpty("rock", 160, 80, 120, 5, 6)
        }
    }
    Wave(30, 0.6) {
        Actions {
            randomAction {
                Actions { spawnAtEmpty("nugget", 160, 80, 120, 5, 6) }
                Actions { spawnAtEmpty("rock", 160, 80, 120, 5, 6) }
                Actions { spawnAtEmpty("nugget", 160, 80, 120, 5, 6) }
            }
        }
    }
}
