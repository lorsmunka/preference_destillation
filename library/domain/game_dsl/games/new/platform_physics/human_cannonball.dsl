Window {
    size(10, 8)
    background("1e1b4b")
    music("bg_4")
    gravity(500)

    lives(3, "-- THE CROWD WENT HOME --", "lose")
    Bar("catches") { initial(0) }
    winAt("catches", 6, "-- STAR OF THE SHOW --", "win")

    Entity {
        id("cannon")
        startPosition(30, 60)
        color("475569")
        oneWay(true)
        Type { rectangle(110, 50) }
    }

    Entity {
        id("daredevil")
        startPosition(62, 112)
        color("fbbf24")
        Type { text("🤸", 40) }
        Movement { speed(0) AutoMove { direction(1, 0) } }
        var("loaded", 1)
        Events {
            OnSignal("fireLow") { Actions { whenVar("loaded", "==", 1) { Actions {
                setVelocity(330, 420)
                changeVar("loaded", -1)
                sound("laser")
                burst("cannon", 14, "f97316")
                shake(3, 0.2) } } } }
            OnSignal("fireMid") { Actions { whenVar("loaded", "==", 1) { Actions {
                setVelocity(400, 480)
                changeVar("loaded", -1)
                sound("laser")
                burst("cannon", 18, "f97316")
                shake(4, 0.2) } } } }
            OnSignal("fireHigh") { Actions { whenVar("loaded", "==", 1) { Actions {
                setVelocity(300, 700)
                changeVar("loaded", -1)
                sound("laser")
                burst("cannon", 24, "f97316")
                shake(6, 0.3) } } } }
            OnCollision("net") {
                Actions {
                    sound("bell")
                    updateBar("catches", 1)
                    flash("net", "ffffff", 0.3)
                    burst(24, "fde047")
                    broadcast("reload")
                }
            }
            OnCollision("balloons") {
                Actions {
                    sound("pop")
                    burst(16, "f472b6")
                    setVelocity(0, -120)
                }
            }
            OnCollision("sawdust") {
                Actions {
                    sound("thud")
                    shake(7, 0.4)
                    burst(18, "d97706")
                    updateBar("lives", -1)
                    broadcast("reload")
                }
            }
            OnSignal("reload") {
                Actions {
                    setPosition("daredevil", 62, 112, 0)
                    setVelocity(0, 0)
                    whenVar("loaded", "==", 0) { Actions { changeVar("loaded", 1) } }
                }
            }
        }
    }

    Entity {
        id("net")
        startPosition(420, 150)
        color("fbbf24")
        Type { rectangle(130, 22) }
        Movement { speed(1.3) Path { waypoint(420, 150) waypoint(840, 150) } }
    }

    Entity {
        id("balloons")
        startPosition(320, 200)
        color("f472b6")
        Type { circle(34) }
        Movement { speed(1.2) Path { waypoint(320, 200) waypoint(320, 620) dwell(0.5) } }
    }

    Entity { id("sawdust") startPosition(0, 0) color("a16207") Type { rectangle(1000, 30) } }

    Entity { id("padLow") startPosition(20, 720) color("22c55e") Type { rectangle(140, 60) }
        Events { OnTapped("padLow") { Actions { sound("click") broadcast("fireLow") } } } }
    Entity { id("padMid") startPosition(170, 720) color("eab308") Type { rectangle(140, 60) }
        Events { OnTapped("padMid") { Actions { sound("click") broadcast("fireMid") } } } }
    Entity { id("padHigh") startPosition(320, 720) color("ef4444") Type { rectangle(140, 60) }
        Events { OnTapped("padHigh") { Actions { sound("click") broadcast("fireHigh") } } } }
    Entity { id("lblLow") startPosition(60, 737) color("052e16") Type { text("LOW", 26) } }
    Entity { id("lblMid") startPosition(213, 737) color("422006") Type { text("MID", 26) } }
    Entity { id("lblHigh") startPosition(355, 737) color("450a0a") Type { text("HIGH", 26) } }
}
