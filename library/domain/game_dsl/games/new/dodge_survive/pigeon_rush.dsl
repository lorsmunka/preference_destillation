Window {
    size(8, 12)
    background("374151")
    music("bg_2")
    gravity(0)

    lives(3, "-- FLATTENED --", "lose")
    Bar("crumbs") { initial(0) }
    winAt("crumbs", 8, "-- FULL BELLY --", "win")

    # traffic gets denser wave by wave; the last wave adds buses and never ends
    Wave(12, 1.0) { Actions { broadcast("car") } }
    Wave(15, 0.7) { Actions { broadcast("car") } }
    Wave(20, 0.5) { Actions { broadcast("car") } }
    Wave(600, 0.45) { Actions { broadcast("car") broadcast("bus") } }

    Entity { id("curbLow") startPosition(0, 0) color("9ca3af") Type { rectangle(800, 140) } }
    Entity { id("curbTop") startPosition(0, 1060) color("9ca3af") Type { rectangle(800, 140) } }
    Entity { id("median") startPosition(0, 597) color("facc15") Type { rectangle(800, 6) } }

    Entity {
        id("pigeon")
        startPosition(375, 40)
        color("cbd5e1")
        Type { circle(25) }
        Movement {
            speed(4)
            Control { movementControl(1) }
        }
        Events {
            OnElapsedTime {
                Actions { spawn("crumbTop", 390, 1110) }
            }
        }
    }

    Entity {
        id("crumbTop")
        template(true)
        startPosition(0, 0)
        color("fcd34d")
        Type { circle(10) }
        Events {
            OnCollision("pigeon") {
                Actions {
                    sound("coin")
                    burst(10, "fcd34d")
                    updateBar("crumbs", 1)
                    spawn("crumbLow", 390, 60)
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("crumbLow")
        template(true)
        startPosition(0, 0)
        color("fcd34d")
        Type { circle(10) }
        Events {
            OnCollision("pigeon") {
                Actions {
                    sound("coin")
                    burst(10, "fcd34d")
                    updateBar("crumbs", 1)
                    spawn("crumbTop", 390, 1110)
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("carR")
        template(true)
        startPosition(0, 0)
        color("ef4444")
        Type { rectangle(110, 50) }
        Movement { speed(3) AutoMove { direction(1, 0) } }
        Events {
            OnElapsedTime { Actions { whenBar("crumbs", ">=", 4) { Actions { updateSpeed(4.2) } } } }
            OnCollision("pigeon") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("pigeon", "ff0000", 0.3)
                    shake(6, 0.3)
                    burst("pigeon", 12, "e5e7eb")
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("carL")
        template(true)
        startPosition(0, 0)
        color("38bdf8")
        Type { rectangle(110, 50) }
        Movement { speed(3) AutoMove { direction(-1, 0) } }
        Events {
            OnElapsedTime { Actions { whenBar("crumbs", ">=", 4) { Actions { updateSpeed(4.2) } } } }
            OnCollision("pigeon") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("pigeon", "ff0000", 0.3)
                    shake(6, 0.3)
                    burst("pigeon", 12, "e5e7eb")
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("bus")
        template(true)
        startPosition(0, 0)
        color("f59e0b")
        Type { rectangle(230, 60) }
        Movement { speed(2) AutoMove { direction(1, 0) } }
        Events {
            OnCollision("pigeon") {
                Actions {
                    sound("thud")
                    updateBar("lives", -1)
                    flash("pigeon", "ff0000", 0.4)
                    shake(9, 0.4)
                    burst("pigeon", 18, "f59e0b")
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("dispatch")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSignal("car") {
                Actions {
                    randomAction {
                        Actions { spawn("carR", 0, 165) }
                        Actions { spawn("carL", 690, 315) }
                        Actions { spawn("carR", 0, 465) }
                        Actions { spawn("carL", 690, 615) }
                        Actions { spawn("carR", 0, 765) }
                        Actions { spawn("carL", 690, 915) }
                    }
                }
            }
            OnSignal("bus") {
                Actions {
                    randomAction {
                        Actions { sound("buzz") spawn("bus", 0, 160) }
                        Actions { }
                        Actions { }
                        Actions { sound("buzz") spawn("bus", 0, 760) }
                        Actions { }
                        Actions { }
                    }
                }
            }
        }
    }
}
