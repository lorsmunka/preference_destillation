Window {
    size(8, 12)
    background("0b132b")
    gravity(0)

    Bar {
        initial(0)
        OnValue(30) { Actions { sound("win") burst("director", 24, "fbbf24") shake(5, 0.4) gameOver("-- BUBBLES POPPED --") } }
        OnValue(-10) { Actions { sound("lose") burst("ceiling", 20, "ef4444") shake(7, 0.5) gameOver("-- TOO MANY ESCAPED --") } }
    }

    Entity {
        id("bubble_small")
        template(true)
        startPosition(0, 0)
        color("a5f3fc")
        Type { circle(18) }
        Movement { speed(2.55) AutoMove { direction(0, 1) } }
        Events {
            OnTapped {
                Actions {
                    sound("bubble")
                    burst(10, "a5f3fc")
                    deleteEntity()
                    updateBar(1)
                }
            }
            OnCollision("ceiling") {
                Actions {
                    sound("buzz")
                    flash("ceiling", "ff0000", 0.25)
                    shake(4, 0.2)
                    deleteEntity()
                    updateBar(-1)
                }
            }
        }
    }

    Entity {
        id("bubble_medium")
        template(true)
        startPosition(0, 0)
        color("5bc0eb")
        Type { circle(28) }
        Movement { speed(1.7) AutoMove { direction(0, 1) } }
        Events {
            OnTapped {
                Actions {
                    sound("bubble")
                    burst(14, "5bc0eb")
                    deleteEntity()
                    updateBar(1)
                }
            }
            OnCollision("ceiling") {
                Actions {
                    sound("buzz")
                    flash("ceiling", "ff0000", 0.25)
                    shake(4, 0.2)
                    deleteEntity()
                    updateBar(-1)
                }
            }
        }
    }

    Entity {
        id("bubble_large")
        template(true)
        startPosition(0, 0)
        color("0891b2")
        Type { circle(40) }
        Movement { speed(1.19) AutoMove { direction(0, 1) } }
        Events {
            OnTapped {
                Actions {
                    sound("bubble")
                    burst(18, "0891b2")
                    deleteEntity()
                    updateBar(1)
                }
            }
            OnCollision("ceiling") {
                Actions {
                    sound("buzz")
                    flash("ceiling", "ff0000", 0.25)
                    shake(4, 0.2)
                    deleteEntity()
                    updateBar(-1)
                }
            }
        }
    }

    Entity {
        id("ceiling")
        startPosition(0, 1180)
        color("1f2937")
        Type { rectangle(800, 20) }
    }

    Entity {
        id("director")
        startPosition(400, 40)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(20, 1.6) {
                    Actions {
                        spawn("bubble_small", random(40, 760), 40)
                        spawn("bubble_large", random(40, 760), 40)
                    }
                }
                Phase(25, 1.0) {
                    Actions {
                        spawn("bubble_small", random(40, 760), 40)
                        spawn("bubble_medium", random(40, 760), 40)
                        spawn("bubble_large", random(40, 760), 40)
                    }
                }
                Phase(60, 0.6) {
                    Actions {
                        spawn("bubble_small", random(40, 760), 40)
                        spawn("bubble_medium", random(40, 760), 40)
                        spawn("bubble_large", random(40, 760), 40)
                        shake(3, 0.15)
                    }
                }
            }
        }
    }
}
