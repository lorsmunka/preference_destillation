Window {
    size(8, 11)
    background("1e1b4b")
    music("bg_4")
    lives(3, "-- THE BOSS LAUGHS --", "lose")
    winAt("boss", 0, "-- BOSS BROKEN --", "win")

    Bar("boss") { initial(10) }

    paddle("paddle", 320, 90, 160, 24)

    Entity {
        id("pit")
        startPosition(0, 0)
        color("00000000")
        Type { rectangle(800, 20) }
        Events {
            OnCollision("ball") {
                Actions {
                    updateBar("lives", -1)
                    sound("hurt")
                    shake(5, 0.3)
                    setPosition("ball", 380, 300, 0)
                }
            }
        }
    }

    Entity {
        id("ball")
        startPosition(380, 300)
        color("fde047")
        Type { circle(14) }
        Movement { speed(5) AutoMove { direction(1, 1) } }
        Events {
            OnCollision("paddle") { Actions { bounceBack() sound("blip") } }
            OnCollision("boss") {
                Actions {
                    bounceBack()
                    updateBar("boss", -1)
                    flash("boss", "ffffff", 0.15)
                    burst("boss", 10, "f472b6")
                    sound("pop")
                }
            }
            OnCollision("pit") { Actions { bounceBack(0) } }
            OnCollision { Actions { bounceBack() } }
        }
    }

    Entity {
        id("boss")
        startPosition(300, 850)
        color("ec4899")
        Type { ring(100, 60) }
        Movement { speed(1.6) Path { waypoint(60, 850) waypoint(500, 850) waypoint(300, 650) } }
        Events {
            Sequence {
                Phase(3, 3) { Actions { spawnHere("dart") } }
                Phase(-1, 1.7) {
                    Actions {
                        spawnHere("dart")
                        whenBar("boss", "<=", 5) { Actions { spawn("drone", 400, 700) } }
                    }
                }
            }
        }
    }

    Entity {
        id("dart")
        template(true)
        startPosition(0, 0)
        color("f9a8d4")
        Type { triangle(26, "down") }
        Movement { speed(3.4) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("paddle") {
                Actions { sound("clang") burst(6, "f9a8d4") deleteEntity() }
            }
            OnCollision("pit") {
                Actions { updateBar("lives", -1) sound("hurt") flash("ef4444", 0.2) deleteEntity() }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("drone")
        template(true)
        startPosition(0, 0)
        color("a78bfa")
        Type { circle(18) }
        Movement { speed(1.3) chase("paddle") }
        Events {
            OnCollision("paddle") {
                Actions { burst(8, "a78bfa") sound("bump") deleteEntity() }
            }
            OnCollision("ball") {
                Actions { burst(8, "a78bfa") updateBar("boss", -1) deleteEntity() }
            }
        }
    }
}
