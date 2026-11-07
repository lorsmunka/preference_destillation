# Lighthouse Keeper: sea creatures swim at the lighthouse along the four compass lines.
# Tap an arrow to send a beam of light that way. Every beam burns oil, which slowly refills.
Window {
    size(9, 9)
    background("0c4a6e")
    music("bg_ambient")
    gravity(0)

    lives(3, "-- THE LIGHT WENT OUT --", "lose")
    Bar("oil") { initial(8) }
    Bar("banished") { initial(0) }

    Entity {
        id("lamp")
        startPosition(380, 380)
        color("60fde047")
        Type { ring(70, 56) }
        Events {
            OnElapsedTime(0.2) {
                repeatCount(-1)
                Actions {
                    whenBar("oil", ">=", 6) { Actions { setOpacity(1) } }
                    whenBar("oil", "<", 6) { Actions { setOpacity(0.55) } }
                    whenBar("oil", "<", 3) { Actions { setOpacity(0.2) } }
                }
            }
        }
    }

    Entity {
        id("tower")
        startPosition(410, 410)
        color("fef3c7")
        Type { rectangle(80, 80) }
        Events {
            OnElapsedTime(0.8) {
                repeatCount(-1)
                Actions { whenBar("oil", "<", 8) { Actions { updateBar("oil", 1) } } }
            }
            OnSignal("fireUp") { Actions { whenBar("oil", ">", 0) { Actions { spawnHere("beamU") updateBar("oil", -1) sound("laser") } } } }
            OnSignal("fireDown") { Actions { whenBar("oil", ">", 0) { Actions { spawnHere("beamD") updateBar("oil", -1) sound("laser") } } } }
            OnSignal("fireLeft") { Actions { whenBar("oil", ">", 0) { Actions { spawnHere("beamL") updateBar("oil", -1) sound("laser") } } } }
            OnSignal("fireRight") { Actions { whenBar("oil", ">", 0) { Actions { spawnHere("beamR") updateBar("oil", -1) sound("laser") } } } }
            OnCollision("squid") {
                Actions {
                    deleteCollider()
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("ef4444", 0.4)
                    shake(7, 0.4)
                    burst(14, "fef3c7")
                }
            }
            OnCollision("crab") {
                Actions {
                    deleteCollider()
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("ef4444", 0.4)
                    shake(7, 0.4)
                    burst(14, "fef3c7")
                }
            }
        }
    }

    Entity { id("upBtn") startPosition(420, 530) color("94a3b8") Type { triangle(60, "up") }
        Events { OnTapped("upBtn") { Actions { broadcast("fireUp") } } } }
    Entity { id("downBtn") startPosition(420, 310) color("94a3b8") Type { triangle(60, "down") }
        Events { OnTapped("downBtn") { Actions { broadcast("fireDown") } } } }
    Entity { id("leftBtn") startPosition(310, 420) color("94a3b8") Type { triangle(60, "left") }
        Events { OnTapped("leftBtn") { Actions { broadcast("fireLeft") } } } }
    Entity { id("rightBtn") startPosition(530, 420) color("94a3b8") Type { triangle(60, "right") }
        Events { OnTapped("rightBtn") { Actions { broadcast("fireRight") } } } }

    Entity { id("beamU") template(true) color("fde047") Type { rectangle(24, 70) }
        Movement { speed(9) AutoMove { direction(0, 1) } }
        Events { OnCollision { Actions { deleteEntity() } } } }
    Entity { id("beamD") template(true) color("fde047") Type { rectangle(24, 70) }
        Movement { speed(9) AutoMove { direction(0, -1) } }
        Events { OnCollision { Actions { deleteEntity() } } } }
    Entity { id("beamL") template(true) color("fde047") Type { rectangle(70, 24) }
        Movement { speed(9) AutoMove { direction(-1, 0) } }
        Events { OnCollision { Actions { deleteEntity() } } } }
    Entity { id("beamR") template(true) color("fde047") Type { rectangle(70, 24) }
        Movement { speed(9) AutoMove { direction(1, 0) } }
        Events { OnCollision { Actions { deleteEntity() } } } }

    Entity { id("squid") template(true) color("f472b6") Type { circle(26) }
        var("hp", 1)
        Movement { speed(0.9) chase("tower") }
        Events {
            OnCollision("beamU") { Actions { deleteCollider() changeVar("hp", -1) } }
            OnCollision("beamD") { Actions { deleteCollider() changeVar("hp", -1) } }
            OnCollision("beamL") { Actions { deleteCollider() changeVar("hp", -1) } }
            OnCollision("beamR") { Actions { deleteCollider() changeVar("hp", -1) } }
            OnElapsedTime(0) { repeatCount(-1) Actions { whenVar("hp", "<=", 0) { Actions {
                sound("pop") burst(12, "f9a8d4") updateBar("banished", 1) deleteEntity() } } } }
        } }

    Entity { id("crab") template(true) color("ea580c") Type { polygon(6, 30) }
        var("hp", 2)
        Movement { speed(0.6) chase("tower") }
        Events {
            OnCollision("beamU") { Actions { deleteCollider() changeVar("hp", -1) flash("ffffff", 0.15) sound("clang") } }
            OnCollision("beamD") { Actions { deleteCollider() changeVar("hp", -1) flash("ffffff", 0.15) sound("clang") } }
            OnCollision("beamL") { Actions { deleteCollider() changeVar("hp", -1) flash("ffffff", 0.15) sound("clang") } }
            OnCollision("beamR") { Actions { deleteCollider() changeVar("hp", -1) flash("ffffff", 0.15) sound("clang") } }
            OnElapsedTime(0) { repeatCount(-1) Actions { whenVar("hp", "<=", 0) { Actions {
                sound("pop") burst(16, "fdba74") updateBar("banished", 1) deleteEntity() } } } }
        } }

    Entity {
        id("tide")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(8, 2.2) {
                    Actions {
                        randomAction {
                            Actions { spawn("squid", 424, 848) }
                            Actions { spawn("squid", 424, 0) }
                            Actions { spawn("squid", 0, 424) }
                            Actions { spawn("squid", 848, 424) }
                        }
                    }
                }
                Phase(10, 1.6) {
                    Actions {
                        randomAction {
                            Actions { spawn("squid", 424, 848) }
                            Actions { spawn("crab", 420, 0) }
                            Actions { spawn("squid", 0, 424) }
                            Actions { spawn("crab", 840, 420) }
                        }
                    }
                }
                Phase(14, 1.2) {
                    Actions {
                        randomAction {
                            Actions { spawn("crab", 420, 840) spawn("squid", 424, 0) }
                            Actions { spawn("crab", 0, 420) }
                            Actions { spawn("squid", 848, 424) spawn("squid", 0, 424) }
                            Actions { spawn("crab", 420, 0) }
                        }
                    }
                }
                Phase(1, 5) {
                    Actions {
                        sound("win")
                        shake(4, 0.4)
                        burst("tower", 36, "fde047")
                        gameOver("-- DAWN BREAKS --")
                    }
                }
            }
        }
    }
}
