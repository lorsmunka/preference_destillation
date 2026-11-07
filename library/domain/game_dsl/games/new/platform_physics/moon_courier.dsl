# Fly a cargo drone in low moon gravity: thrusters add to its velocity, nothing slows it down.
# Land on the mine pad to load ore, land back on the base pad to deliver it.
Window {
    size(12, 8)
    background("030712")
    music("bg_6")
    gravity(150)

    lives(3, "-- THE DRONE IS SCRAP --", "lose")
    Bar("fuel") { initial(60) }
    Bar("delivered") { initial(0) }
    winAt("delivered", 5, "-- THE BASE IS STOCKED --", "win")

    Wave(30, 4) { Actions { spawn("meteor", random(300, 1150), 790) } }
    Wave(30, 3) { Actions { spawn("meteor", random(300, 1150), 790) } }
    Wave(90, 2) { Actions { spawn("meteor", random(300, 1150), 790) } }

    Entity {
        id("drone")
        startPosition(100, 142)
        color("e2e8f0")
        Type { rectangle(50, 30) }
        Movement {
            speed(0)
            AutoMove { direction(0, 1) }
        }
        var("cargo", 0)
        Events {
            OnSignal("thrustUp") { Actions { applyImpulse(0, 110) burst(6, "f97316") } }
            OnSignal("thrustLeft") { Actions { applyImpulse(-70, 25) burst(4, "f97316") } }
            OnSignal("thrustRight") { Actions { applyImpulse(70, 25) burst(4, "f97316") } }
            OnCollision { Actions { sound("bump") bounceBack(0) } }

            # the lit strip on top of each pad brings the drone to a stop
            OnCollision("baseDeck") {
                Actions {
                    setVelocity(0, 0)
                    whenVar("cargo", "==", 1) {
                        Actions {
                            sound("coin")
                            updateBar("delivered", 1)
                            changeVar("cargo", -1)
                            setColor("e2e8f0")
                            burst(20, "4ade80")
                        }
                    }
                    whenBar("fuel", "<", 20) { Actions { sound("sparkle") updateBar("fuel", 10) } }
                }
            }
            OnCollision("mineDeck") {
                Actions {
                    setVelocity(0, 0)
                    whenVar("cargo", "==", 0) {
                        Actions {
                            sound("pickup")
                            changeVar("cargo", 1)
                            setColor("f59e0b")
                            burst(14, "f59e0b")
                        }
                    }
                }
            }
            OnCollision("fuelDeck") {
                Actions {
                    setVelocity(0, 0)
                    whenBar("fuel", "<", 50) {
                        Actions {
                            sound("sparkle")
                            updateBar("fuel", 15)
                            burst(12, "38bdf8")
                        }
                    }
                }
            }

            OnSignal("crash") {
                Actions {
                    sound("death")
                    shake(9, 0.5)
                    burst(30, "f97316")
                    updateBar("lives", -1)
                    setPosition("drone", 100, 142, 0)
                    setVelocity(0, 0)
                    whenVar("cargo", "==", 1) { Actions { changeVar("cargo", -1) setColor("e2e8f0") } }
                }
            }
        }
    }

    # ---- pads: a one-way landing plate with a lit deck strip on top ----
    Entity { id("basePad") startPosition(40, 120) color("64748b") oneWay(true) Type { rectangle(180, 20) } }
    Entity { id("baseDeck") startPosition(40, 134) color("4ade80") Type { rectangle(180, 14) } }
    Entity { id("cliff") startPosition(40, 40) color("44403c") Type { rectangle(180, 80) }
        Events { OnCollision("drone") { Actions { broadcast("crash") } } } }

    Entity { id("minePad") startPosition(960, 60) color("64748b") oneWay(true) Type { rectangle(180, 20) } }
    Entity { id("mineDeck") startPosition(960, 74) color("f59e0b") Type { rectangle(180, 14) } }

    Entity { id("fuelPad") startPosition(570, 340) color("64748b") oneWay(true) Type { rectangle(140, 20) } }
    Entity { id("fuelDeck") startPosition(570, 354) color("38bdf8") Type { rectangle(140, 14) } }
    Entity { id("mesa") startPosition(580, 40) color("44403c") Type { rectangle(120, 300) }
        Events { OnCollision("drone") { Actions { broadcast("crash") } } } }

    # ---- hazards ----
    Entity { id("regolith") startPosition(0, 0) color("57534e") Type { rectangle(1200, 40) }
        Events { OnCollision("drone") { Actions { broadcast("crash") } } } }
    Entity { id("antenna1") startPosition(360, 40) color("94a3b8") Type { rectangle(24, 320) }
        Events {
            OnCollision("drone") { Actions { broadcast("crash") } }
            OnElapsedTime(1) { repeatCount(-1) Actions { flash("ef4444", 0.3) } } } }
    Entity { id("antenna2") startPosition(830, 40) color("94a3b8") Type { rectangle(24, 240) }
        Events {
            OnCollision("drone") { Actions { broadcast("crash") } }
            OnElapsedTime(1.3) { repeatCount(-1) Actions { flash("ef4444", 0.3) } } } }

    Entity {
        id("meteor")
        template(true)
        startPosition(0, 0)
        color("fb923c")
        gravity(0)
        Type { circle(14) }
        Movement { speed(2) AutoMove { direction(-0.5, -1) } }
        Events {
            OnCollision("drone") { Actions { broadcast("crash") deleteEntity() } }
            OnCollision("regolith") { Actions { sound("thud") burst(10, "fb923c") deleteEntity() } }
            OnElapsedTime(0.1) { repeatCount(-1) Actions { burst(1, "fdba74") } }
        }
    }

    # ---- thrusters: each puff costs one fuel ----
    Entity {
        id("thrustL")
        startPosition(20, 720)
        color("334155")
        Type { rectangle(170, 70) }
        Events {
            OnTapped("thrustL") {
                Actions {
                    whenBar("fuel", ">", 0) { Actions { sound("blip") updateBar("fuel", -1) broadcast("thrustLeft") } }
                    whenBar("fuel", "<=", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                }
            }
        }
    }
    Entity {
        id("thrustU")
        startPosition(210, 720)
        color("334155")
        Type { rectangle(170, 70) }
        Events {
            OnTapped("thrustU") {
                Actions {
                    whenBar("fuel", ">", 0) { Actions { sound("blip") updateBar("fuel", -1) broadcast("thrustUp") } }
                    whenBar("fuel", "<=", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                }
            }
        }
    }
    Entity {
        id("thrustR")
        startPosition(400, 720)
        color("334155")
        Type { rectangle(170, 70) }
        Events {
            OnTapped("thrustR") {
                Actions {
                    whenBar("fuel", ">", 0) { Actions { sound("blip") updateBar("fuel", -1) broadcast("thrustRight") } }
                    whenBar("fuel", "<=", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                }
            }
        }
    }
    Entity { id("lblL") startPosition(85, 738) color("e2e8f0") Type { text("<", 36) } }
    Entity { id("lblU") startPosition(280, 738) color("e2e8f0") Type { text("^", 36) } }
    Entity { id("lblR") startPosition(470, 738) color("e2e8f0") Type { text(">", 36) } }
    Entity { id("lblBase") startPosition(100, 70) color("4ade80") Type { text("BASE", 22) } }
    Entity { id("lblMine") startPosition(1030, 41) color("f59e0b") Type { text("MINE", 16) } }
    Entity { id("lblFuel") startPosition(612, 200) color("38bdf8") Type { text("FUEL", 22) } }
}
