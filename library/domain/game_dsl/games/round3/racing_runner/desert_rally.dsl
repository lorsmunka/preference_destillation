Window {
    size(8, 12)
    background("92400e")
    music("bg_6")
    gravity(0)

    lives(3, "-- ENGINE DEAD --", "lose")
    Bar("km") { initial(0) }
    winAt("km", 45, "-- FINISH LINE REACHED --", "win")
    Bar("fuel") {
        initial(10)
        OnValue(0) { Actions {
            sound("lose")
            gameOver("-- OUT OF FUEL --")
        } }
    }
    Bar("heat") {
        initial(0)
        OnValue(4) { Actions { broadcast("hotter") } }
        OnValue(8) { Actions { broadcast("hotter") } }
    }

    Entity { id("dune1") startPosition(80, 900) color("b45309")
        Type { ring(60, 40) }
        Movement { speed(1.2) AutoMove { direction(0, -1) } }
        Events { OnCollision { Actions { setPosition("dune1", 80, 1100, 0) } } } }
    Entity { id("dune2") startPosition(640, 500) color("b45309")
        Type { ring(50, 30) }
        Movement { speed(1.2) AutoMove { direction(0, -1) } }
        Events { OnCollision { Actions { setPosition("dune2", 640, 1100, 0) } } } }

    Entity {
        id("car")
        startPosition(370, 100)
        color("fef08a")
        Type { rectangle(64, 104) }
        Movement { speed(6) Control { movementControl(0) } }
        Events {
            OnSwipe("left") { Actions { setPosition("car", 90, 100, 0.15) } }
            OnSwipe("right") { Actions { setPosition("car", 640, 100, 0.15) } }
            OnCollision("tumbleweed") {
                Actions {
                    sound("hurt")
                    shake(7, 0.35)
                    flash("ff0000", 0.3)
                    updateBar("lives", -1)
                    deleteCollider()
                }
            }
            OnCollision("cactus") {
                Actions {
                    sound("thud")
                    shake(5, 0.25)
                    updateBar("fuel", -2)
                    deleteCollider()
                }
            }
            OnCollision("jerrycan") {
                Actions {
                    sound("pickup")
                    updateBar("fuel", 4)
                    burst(8, "22c55e")
                    deleteCollider()
                }
            }
            OnCollision("hunter") {
                Actions {
                    sound("clang")
                    shake(9, 0.45)
                    updateBar("lives", -1)
                    deleteCollider()
                }
            }
        }
    }

    Entity {
        id("tumbleweed")
        template(true)
        startPosition(0, 0)
        color("ca8a04")
        Type { polygon(7, 30) }
        Movement { speed(3) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("car") { Actions { deleteEntity() } }
            OnCollision {
                Actions {
                    updateBar("heat", 1)
                    deleteEntity()
                }
            }
            OnSignal("hotter") { Actions { updateSpeed(5) } }
        }
    }

    Entity {
        id("cactus")
        template(true)
        startPosition(0, 0)
        color("15803d")
        Type { cross(40, 90, 14) }
        Movement { speed(2.4) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("car") { Actions { deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("jerrycan")
        template(true)
        startPosition(0, 0)
        color("16a34a")
        Type { rectangle(34, 44) }
        Movement { speed(2.4) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("car") { Actions { deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("hunter")
        template(true)
        startPosition(0, 0)
        color("dc2626")
        Type { rectangle(60, 96) }
        Movement { speed(2.2) chase("car", 0.3) }
        Events {
            OnElapsedTime(7) { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("director")
        startPosition(400, 40)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(10, 1.5) {
                    Actions { spawn("tumbleweed", random(40, 700), 1080) }
                }
                Phase(15, 1.1) {
                    Actions {
                        spawn("tumbleweed", random(40, 700), 1080)
                        spawn("cactus", random(40, 700), 1080)
                    }
                }
                Phase(-1, 0.8) {
                    Actions {
                        spawn("tumbleweed", random(40, 700), 1080)
                        spawn("cactus", random(40, 700), 1080)
                    }
                }
            }
            OnElapsedTime(1) {
                repeatCount(-1)
                Actions {
                    updateBar("km", 1)
                    updateBar("fuel", -1)
                }
            }
        }
    }

    Wave(60, 4) { Actions { spawn("jerrycan", random(60, 700), 1080) } }
    Wave(60, 14) { Actions { spawn("hunter", random(60, 700), 1080) } }
}
