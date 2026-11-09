# Bombs drop on five fixed spots. Each one flashes, then blows up in a cross. Drag the courier to the crates and keep clear.
Window {
    size(8, 10)
    background("172554")
    music("bg_6")
    gravity(0)

    lives(3, "-- BLOWN UP --", "lose")
    winAt("crates", 6, "-- ALL CRATES DELIVERED --", "win")
    wallBox(0, 0, 800, 1000, "1e3a8a", 16)

    Entity {
        id("courier")
        startPosition(380, 60)
        color("f8fafc")
        Type { circle(24) }
        Movement { speed(6) Control { movementControl(0) } }
    }

    Entity { id("crate1") startPosition(80, 120) color("d97706") Type { rectangle(40, 40) }
        Events { OnCollision("courier") { Actions { sound("pickup") updateBar("crates", 1) burst(8, "d97706") deleteEntity() } } } }
    Entity { id("crate2") startPosition(680, 120) color("d97706") Type { rectangle(40, 40) }
        Events { OnCollision("courier") { Actions { sound("pickup") updateBar("crates", 1) burst(8, "d97706") deleteEntity() } } } }
    Entity { id("crate3") startPosition(375, 470) color("d97706") Type { rectangle(40, 40) }
        Events { OnCollision("courier") { Actions { sound("pickup") updateBar("crates", 1) burst(8, "d97706") deleteEntity() } } } }
    Entity { id("crate4") startPosition(80, 880) color("d97706") Type { rectangle(40, 40) }
        Events { OnCollision("courier") { Actions { sound("pickup") updateBar("crates", 1) burst(8, "d97706") deleteEntity() } } } }
    Entity { id("crate5") startPosition(680, 880) color("d97706") Type { rectangle(40, 40) }
        Events { OnCollision("courier") { Actions { sound("pickup") updateBar("crates", 1) burst(8, "d97706") deleteEntity() } } } }
    Entity { id("crate6") startPosition(375, 800) color("d97706") Type { rectangle(40, 40) }
        Events { OnCollision("courier") { Actions { sound("pickup") updateBar("crates", 1) burst(8, "d97706") deleteEntity() } } } }

    Entity {
        id("blastAcross")
        template(true)
        startPosition(0, 0)
        color("fb923c")
        Type { rectangle(300, 40) }
        Events {
            OnCollision("courier") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    shake(8, 0.4)
                    setPosition("courier", 380, 60, 0)
                    deleteEntity()
                }
            }
            OnElapsedTime(0.45) { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("blastUp")
        template(true)
        startPosition(0, 0)
        color("fdba74")
        Type { rectangle(40, 300) }
        Events {
            OnCollision("courier") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    shake(8, 0.4)
                    setPosition("courier", 380, 60, 0)
                    deleteEntity()
                }
            }
            OnElapsedTime(0.45) { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("bomb")
        template(true)
        startPosition(0, 0)
        color("94a3b8")
        Type { circle(22) }
        Events {
            OnElapsedTime(0.05) { Actions { flash("ef4444", 1.2) } }
            OnElapsedTime(1.3) {
                Actions {
                    sound("thud")
                    burst(14, "fb923c")
                    shake(3, 0.2)
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("bomber")
        startPosition(400, 960)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(2.2) {
                repeatCount(-1)
                Actions {
                    sound("bump")
                    randomAction {
                        Actions {
                            spawn("bomb", 200, 300)
                            delay(1.3) { Actions { spawn("blastAcross", 72, 302) spawn("blastUp", 202, 172) } }
                        }
                        Actions {
                            spawn("bomb", 550, 300)
                            delay(1.3) { Actions { spawn("blastAcross", 422, 302) spawn("blastUp", 552, 172) } }
                        }
                        Actions {
                            spawn("bomb", 200, 650)
                            delay(1.3) { Actions { spawn("blastAcross", 72, 652) spawn("blastUp", 202, 522) } }
                        }
                        Actions {
                            spawn("bomb", 550, 650)
                            delay(1.3) { Actions { spawn("blastAcross", 422, 652) spawn("blastUp", 552, 522) } }
                        }
                        Actions {
                            spawn("bomb", 375, 475)
                            delay(1.3) { Actions { spawn("blastAcross", 247, 477) spawn("blastUp", 377, 347) } }
                        }
                    }
                }
            }
        }
    }
}
