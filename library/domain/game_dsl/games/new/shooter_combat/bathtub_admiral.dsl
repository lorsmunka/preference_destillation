Window {
    size(10, 12)
    background("bfdbfe")
    music("bg_5")
    gravity(0)

    lives(4, "-- THE DUCK SANK --", "lose")
    Bar("score") {
        initial(0)
        OnValue(20) { Actions { broadcast("powerUp") } }
        OnValue(40) { Actions { broadcast("powerUp") } }
    }
    Bar("suds") {
        initial(0)
        OnValue(8) {
            Actions {
                sound("lose")
                shake(8, 0.6)
                gameOver("-- THE TUB OVERFLOWED --")
            }
        }
    }

    Wave(20, 2.2) {
        Actions {
            randomAction {
                Actions { spawn("bubble", random(20, 900), 1110) }
                Actions { spawn("bubble", random(20, 900), 1110) }
                Actions { spawn("sponge", random(0, 900), random(820, 1120)) }
            }
        }
    }
    Wave(25, 1.7) {
        Actions {
            randomAction {
                Actions { spawn("bubble", random(20, 900), 1110) }
                Actions { spawn("sponge", random(0, 900), random(820, 1120)) }
                Actions { spawn("shark", random(0, 940), 1140) }
            }
        }
    }
    Wave(25, 1.3) {
        Actions {
            randomAction {
                Actions { spawn("bubble", random(20, 440), 1110) spawn("bubble", random(480, 900), 1110) }
                Actions { spawn("shark", random(0, 940), 1140) }
                Actions { spawn("sponge", random(0, 900), random(820, 1120)) }
            }
        }
    }
    Wave(2) {
        Actions {
            sound("buzz")
            shake(8, 0.8)
            spawn("kraken", 420, 980)
            spawn("banner", 170, 620)
        }
    }

    Entity {
        id("duck")
        startPosition(470, 80)
        color("facc15")
        Type { circle(30) }
        var("power", 1)
        var("shield", 0)
        Movement {
            speed(7)
            Control { movementControl(0) }
        }
        Events {
            OnElapsedTime(0.5) {
                repeatCount(-1)
                Actions {
                    spawnHere("squirt")
                    whenVar("power", ">=", 2) {
                        Actions { delay(0.25) { Actions { spawnHere("squirt") } } }
                    }
                }
            }
            OnElapsedTime(2) {
                repeatCount(-1)
                Actions {
                    whenVar("power", ">=", 3) {
                        Actions { spawnHere("torpedo") sound("bubble") }
                    }
                }
            }
            OnSignal("powerUp") {
                Actions {
                    sound("sparkle")
                    flash("f472b6", 0.5)
                    whenVar("power", "<", 3) { Actions { changeVar("power", 1) } }
                }
            }
            OnSignal("splash") {
                Actions {
                    whenVar("shield", "==", 0) {
                        Actions {
                            changeVar("shield", 1)
                            updateBar("lives", -1)
                            sound("hurt")
                            shake(6, 0.3)
                            flash("ef4444", 1)
                            setOpacity(0.5)
                            delay(1) { Actions { changeVar("shield", -1) setOpacity(1) } }
                        }
                    }
                }
            }
        }
    }

    Entity {
        id("squirt")
        template(true)
        color("0284c7")
        Type { rectangle(8, 20) }
        Movement { speed(9) AutoMove { direction(0, 1) } }
        Events { OnCollision { Actions { deleteEntity() } } }
    }

    # the torpedo is the level-3 weapon: slow, but it ploughs through everything in its way
    Entity {
        id("torpedo")
        template(true)
        color("f97316")
        Type { circle(18) }
        Movement { speed(5) AutoMove { direction(0, 1) } }
        Events { OnCollision { Actions { deleteEntity() } } }
    }

    Entity {
        id("bubble")
        template(true)
        color("c0f0f9ff")
        Type { circle(40) }
        Movement { speed(0.8) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("squirt") {
                Actions {
                    deleteCollider()
                    sound("bubble")
                    burst(12, "e0f2fe")
                    updateBar("score", 1)
                    spawnHere("mini")
                    spawnHere("mini")
                    deleteEntity()
                }
            }
            OnCollision("torpedo") {
                Actions {
                    sound("bubble")
                    burst(12, "e0f2fe")
                    updateBar("score", 1)
                    deleteEntity()
                }
            }
            OnCollision("duck") { Actions { broadcast("splash") burst(12, "e0f2fe") deleteEntity() } }
            OnCollision { Actions { sound("buzz") updateBar("suds", 1) deleteEntity() } }
        }
    }

    Entity {
        id("mini")
        template(true)
        color("c0bae6fd")
        Type { circle(18) }
        Movement { speed(2) AutoMove { direction(1, 1) } }
        Events {
            OnElapsedTime { Actions { setVelocity(random(-220, 220), random(-200, 200)) } }
            OnCollision { Actions { bounceBack(0) } }
            OnCollision("squirt") { Actions { deleteCollider() sound("pop") burst(6, "bae6fd") updateBar("score", 1) deleteEntity() } }
            OnCollision("torpedo") { Actions { sound("pop") burst(6, "bae6fd") updateBar("score", 1) deleteEntity() } }
            OnCollision("duck") { Actions { broadcast("splash") deleteEntity() } }
            OnElapsedTime(7) { Actions { setOpacity(0.3) delay(0.6) { Actions { deleteEntity() } } } }
        }
    }

    Entity {
        id("sponge")
        template(true)
        color("fde68a")
        Type { rectangle(90, 40) }
        var("hp", 3)
        Movement { speed(1.4) AutoMove { direction(1, 0) } }
        Events {
            OnElapsedTime { Actions { randomAction { Actions { setVelocity(140, 0) } Actions { setVelocity(-140, 0) } } } }
            OnCollision { Actions { bounceBack(0) } }
            OnElapsedTime(random(2, 3.5)) { repeatCount(-1) Actions { spawnHere("soap") } }
            OnCollision("squirt") { Actions { deleteCollider() changeVar("hp", -1) flash("ffffff", 0.1) } }
            OnCollision("torpedo") { Actions { changeVar("hp", -3) } }
            OnElapsedTime(0) {
                repeatCount(-1)
                Actions {
                    whenVar("hp", "<=", 0) {
                        Actions {
                            sound("pop")
                            burst(16, "fde68a")
                            updateBar("score", 1)
                            randomAction { Actions { spawnHere("toy") } Actions { } }
                            deleteEntity()
                        }
                    }
                }
            }
        }
    }

    Entity {
        id("shark")
        template(true)
        color("64748b")
        Type { triangle(50, "down") }
        var("hp", 2)
        Movement { speed(1.1) chase("duck") }
        Events {
            OnCollision("squirt") { Actions { deleteCollider() changeVar("hp", -1) flash("ffffff", 0.1) sound("bump") } }
            OnCollision("torpedo") { Actions { changeVar("hp", -2) } }
            OnCollision("duck") { Actions { broadcast("splash") burst(12, "64748b") deleteEntity() } }
            OnElapsedTime(0) {
                repeatCount(-1)
                Actions {
                    whenVar("hp", "<=", 0) {
                        Actions { sound("pop") burst(14, "94a3b8") updateBar("score", 1) deleteEntity() }
                    }
                }
            }
        }
    }

    Entity {
        id("soap")
        template(true)
        color("f9a8d4")
        Type { circle(10) }
        Movement { speed(3.2) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("duck") { Actions { broadcast("splash") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("toy")
        template(true)
        color("f472b6")
        Type { star(5, 18, 8) }
        Movement { speed(1.5) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("duck") { Actions { broadcast("powerUp") burst(12, "f472b6") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("banner")
        template(true)
        color("4c1d95")
        Type { text("THE DRAIN KRAKEN RISES", 44) }
        Events { OnElapsedTime(2.5) { Actions { deleteEntity() } } }
    }

    Entity {
        id("kraken")
        template(true)
        color("7c3aed")
        Type { ring(80, 40) }
        var("hp", 40)
        var("angry", 0)
        Movement {
            speed(1.2)
            Path {
                waypoint(420, 980)
                waypoint(760, 980)
                waypoint(80, 980)
            }
        }
        Events {
            Sequence {
                Phase(3, 2.5) { Actions { sound("bubble") spawn("shark", 100, 1140) spawn("shark", 860, 1140) } }
                Phase(4, 1.8) { Actions { sound("blip") spawnHere("ink") spawnHere("ink") spawnHere("ink") } }
                Phase(-1, 1.4) {
                    Actions {
                        randomAction {
                            Actions { sound("blip") spawnHere("ink") spawnHere("ink") spawnHere("ink") }
                            Actions { spawnHere("bubble") }
                            Actions { sound("bubble") spawn("shark", random(0, 940), 1140) }
                        }
                    }
                }
            }
            OnCollision("squirt") { Actions { deleteCollider() changeVar("hp", -1) flash("ffffff", 0.08) } }
            OnCollision("torpedo") { Actions { changeVar("hp", -3) flash("f97316", 0.2) shake(4, 0.2) } }
            OnElapsedTime(0) {
                repeatCount(-1)
                Actions {
                    whenVar("hp", "<=", 20) {
                        Actions {
                            whenVar("angry", "==", 0) {
                                Actions {
                                    changeVar("angry", 1)
                                    sound("buzz")
                                    shake(8, 0.6)
                                    setColor("be123c")
                                    setScale(1.2)
                                    updateSpeed(2.2)
                                }
                            }
                        }
                    }
                    whenVar("hp", "<=", 0) {
                        Actions {
                            sound("win")
                            shake(10, 0.8)
                            burst(50, "7c3aed")
                            gameOver("-- ADMIRAL OF THE BATHTUB --")
                        }
                    }
                }
            }
        }
    }

    Entity {
        id("ink")
        template(true)
        color("4c1d95")
        Type { circle(14) }
        Movement { speed(3.4) AutoMove { direction(0, -1) } }
        Events {
            OnElapsedTime { Actions { setVelocity(random(-170, 170), -330) } }
            OnCollision("duck") { Actions { broadcast("splash") burst(8, "4c1d95") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }
}
