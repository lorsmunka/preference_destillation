# Junkyard Titan: a scrap-metal giant guards the yard. Break both claws to crack its core open,
# then pound the core through two angrier phases. Drones drop spanners that speed up your gun.
Window {
    size(9, 14)
    background("1c1917")
    music("bg_6")
    gravity(0)

    lives(5, "-- SCRAPPED --", "lose")
    Bar("core") {
        initial(30)
        OnValue(20) { Actions { broadcast("phase2") } }
        OnValue(10) { Actions { broadcast("phase3") } }
        OnValue(0) {
            Actions {
                sound("win")
                shake(10, 0.8)
                burst("core", 50, "38bdf8")
                burst("titan", 40, "f97316")
                gameOver("-- THE TITAN IS SCRAP --")
            }
        }
    }
    Bar("scrap") {
        initial(0)
        OnValue(8) { Actions { broadcast("powerUp") } }
    }

    Entity {
        id("pilot")
        startPosition(410, 90)
        color("38bdf8")
        Type { polygon(5, 36) }
        var("guns", 0)
        var("shield", 0)
        Movement {
            speed(7)
            Control { movementControl(0) }
        }
        Events {
            OnElapsedTime(0.4) {
                repeatCount(-1)
                Actions {
                    spawnHere("bolt")
                    whenVar("guns", ">=", 1) { Actions { delay(0.2) { Actions { spawnHere("bolt") } } } }
                    whenVar("guns", ">=", 2) { Actions {
                        delay(0.1) { Actions { spawnHere("bolt") } }
                        delay(0.3) { Actions { spawnHere("bolt") } } } }
                }
            }
            OnSignal("ouch") {
                Actions {
                    whenVar("shield", "==", 0) {
                        Actions {
                            changeVar("shield", 1)
                            updateBar("lives", -1)
                            sound("hurt")
                            shake(7, 0.4)
                            flash("ef4444", 1.2)
                            setOpacity(0.5)
                            delay(1.2) { Actions { changeVar("shield", -1) setOpacity(1) } }
                        }
                    }
                }
            }
            OnSignal("powerUp") {
                Actions {
                    sound("sparkle")
                    flash("fde047", 0.4)
                    whenVar("guns", "<", 2) { Actions { changeVar("guns", 1) } }
                }
            }
        }
    }

    Entity {
        id("bolt")
        template(true)
        color("fde047")
        Type { rectangle(8, 22) }
        Movement { speed(10) AutoMove { direction(0, 1) } }
        Events { OnCollision { Actions { deleteEntity() } } }
    }

    # the core hangs under the titan's chest; it is shielded until both claws are gone
    Entity {
        id("core")
        startPosition(405, 1060)
        color("0ea5e9")
        Type { circle(45) }
        var("open", 0)
        Events {
            OnElapsedTime { Actions { setOpacity(0.4) } }
            OnCollision("bolt") {
                Actions {
                    deleteCollider()
                    whenVar("open", "==", 0) { Actions { sound("clang") } }
                    whenVar("open", "==", 1) { Actions { updateBar("core", -1) flash("ffffff", 0.08) burst(5, "7dd3fc") } }
                }
            }
            OnSignal("exposed") { Actions { changeVar("open", 1) setOpacity(1) sound("bell") flash("ffffff", 0.6) } }
            OnSignal("phase3") { Actions { setScale(1.3) setPosition("core", 391, 1046, 0) setColor("ef4444") } }
            OnCollision("pilot") { Actions { broadcast("ouch") } }
        }
    }

    Entity {
        id("titan")
        startPosition(200, 1150)
        color("78716c")
        Type { rectangle(500, 200) }
        var("arms", 2)
        var("rage", 0)
        Events {
            OnCollision("bolt") { Actions { deleteCollider() sound("clang") } }
            OnCollision("pilot") { Actions { broadcast("ouch") } }
            Sequence {
                Phase(2, 5) { Actions { sound("beep") spawn("drone", 150, 1090) spawn("drone", 700, 1090) } }
                Phase(-1, 6.5) {
                    Actions {
                        sound("beep")
                        randomAction {
                            Actions { spawn("drone", 150, 1090) spawn("drone", 700, 1090) }
                            Actions { spawn("drone", 420, 1000) }
                            Actions { spawn("drone", 0, 700) spawn("drone", 856, 700) }
                        }
                    }
                }
            }
            OnSignal("armDown") {
                Actions {
                    changeVar("arms", -1)
                    shake(8, 0.5)
                    whenVar("arms", "==", 0) { Actions { broadcast("exposed") sound("buzz") } }
                }
            }
            OnSignal("phase2") { Actions { changeVar("rage", 1) setColor("b45309") shake(8, 0.5) sound("buzz") } }
            OnSignal("phase3") { Actions { changeVar("rage", 1) setColor("7f1d1d") shake(10, 0.6) sound("buzz") } }
            OnElapsedTime(2.4) {
                repeatCount(-1)
                Actions {
                    whenVar("rage", ">=", 1) { Actions {
                        spawn("rivet", random(220, 680), 1130)
                        spawn("rivet", random(220, 680), 1130) } }
                }
            }
            OnElapsedTime(3.2) {
                repeatCount(-1)
                Actions {
                    whenVar("rage", ">=", 2) { Actions {
                        sound("laser")
                        spawn("missile", 260, 1110)
                        spawn("missile", 620, 1110) } }
                }
            }
            OnElapsedTime(7) {
                repeatCount(-1)
                Actions { whenVar("rage", ">=", 2) { Actions { spawn("medic", 0, 760) spawn("medic", 860, 760) } } }
            }
        }
    }

    Entity {
        id("clawL")
        startPosition(60, 900)
        color("a8a29e")
        Type { rectangle(110, 70) }
        var("hp", 10)
        Movement { speed(1.4) Path { waypoint(60, 900) waypoint(300, 760) } }
        Events {
            Sequence {
                Phase(4, 2.2) { Actions { spawnHere("rivet") } }
                Phase(-1, 1.6) { Actions { spawnHere("rivet") delay(0.25) { Actions { spawnHere("rivet") } } } }
            }
            OnCollision("bolt") {
                Actions {
                    deleteCollider()
                    changeVar("hp", -1)
                    flash("ffffff", 0.08)
                    whenVar("hp", "<=", 0) { Actions { broadcast("armDown") sound("glass") burst(30, "a8a29e") deleteEntity() } }
                }
            }
            OnCollision("pilot") { Actions { broadcast("ouch") } }
            OnSignal("phase2") { Actions { updateSpeed(2.2) } }
        }
    }

    Entity {
        id("clawR")
        startPosition(740, 900)
        color("a8a29e")
        Type { rectangle(110, 70) }
        var("hp", 10)
        Movement { speed(1.4) Path { waypoint(740, 900) waypoint(500, 760) } }
        Events {
            Sequence {
                Phase(4, 2.6) { Actions { spawnHere("rivet") } }
                Phase(-1, 1.8) { Actions { spawnHere("rivet") delay(0.25) { Actions { spawnHere("rivet") } } } }
            }
            OnCollision("bolt") {
                Actions {
                    deleteCollider()
                    changeVar("hp", -1)
                    flash("ffffff", 0.08)
                    whenVar("hp", "<=", 0) { Actions { broadcast("armDown") sound("glass") burst(30, "a8a29e") deleteEntity() } }
                }
            }
            OnCollision("pilot") { Actions { broadcast("ouch") } }
            OnSignal("phase2") { Actions { updateSpeed(2.2) } }
        }
    }

    Entity {
        id("rivet")
        template(true)
        color("fbbf24")
        Type { circle(10) }
        Movement { speed(3.6) AutoMove { direction(0, -1) } }
        Events {
            OnElapsedTime { Actions { setVelocity(random(-70, 70), -360) } }
            OnCollision("pilot") { Actions { broadcast("ouch") burst(8, "fbbf24") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("drone")
        template(true)
        color("facc15")
        Type { polygon(6, 22) }
        var("hp", 2)
        Movement { speed(1.3) chase("pilot") }
        Events {
            OnCollision("bolt") {
                Actions {
                    deleteCollider()
                    changeVar("hp", -1)
                    flash("ffffff", 0.08)
                    whenVar("hp", "<=", 0) {
                        Actions {
                            sound("pop")
                            burst(12, "facc15")
                            updateBar("scrap", 1)
                            randomAction { Actions { spawnHere("spanner") } Actions { } Actions { } }
                            deleteEntity()
                        }
                    }
                }
            }
            OnCollision("pilot") { Actions { broadcast("ouch") burst(12, "facc15") deleteEntity() } }
        }
    }

    Entity {
        id("spanner")
        template(true)
        color("22d3ee")
        Type { cross(30, 30, 10) }
        Movement { speed(1.2) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("pilot") { Actions { broadcast("powerUp") burst(10, "22d3ee") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("missile")
        template(true)
        color("ef4444")
        Type { triangle(24, "down") }
        Movement { speed(2) chase("pilot") }
        Events {
            OnCollision("bolt") { Actions { deleteCollider() sound("pop") burst(10, "ef4444") deleteEntity() } }
            OnCollision("pilot") { Actions { broadcast("ouch") burst(14, "ef4444") deleteEntity() } }
            OnElapsedTime(4.5) { Actions { burst(8, "fca5a5") deleteEntity() } }
        }
    }

    # in the last phase little repair bots crawl to the core and patch it unless you shoot them first
    Entity {
        id("medic")
        template(true)
        color("4ade80")
        Type { cross(34, 34, 12) }
        Movement { speed(0.9) chase("core") }
        Events {
            OnCollision("bolt") { Actions { deleteCollider() sound("pop") burst(10, "4ade80") deleteEntity() } }
            OnCollision("core") { Actions { updateBar("core", 2) sound("sparkle") flash("core", "4ade80", 0.4) deleteEntity() } }
        }
    }
}
