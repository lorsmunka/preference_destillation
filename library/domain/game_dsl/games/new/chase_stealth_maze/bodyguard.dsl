Window {
    size(8, 16)
    background("1f2937")
    music("bg_3")
    gravity(0)

    lives(3, "-- THE VIP IS DOWN --", "lose")

    # The agent is solid: attackers cannot pass through you, so stand in their way.
    Entity {
        id("agent")
        startPosition(150, 60)
        color("0ea5e9")
        solid(true)
        Type { rectangle(50, 50) }
        Movement { speed(3.6) Control { movementControl(0) } }
    }

    # The VIP keeps a step behind you wherever you go.
    Entity {
        id("vip")
        startPosition(60, 60)
        color("fde047")
        Type { circle(20) }
        Movement { speed(3.4) chase("agent", 0.9) }
        Events {
            OnCollision("limo") { Actions { sound("win") shake(4, 0.4) burst(30, "fde047") gameOver("-- THE VIP IS SAFE --") } }
        }
    }

    Entity { id("hint") startPosition(240, 345) color("94a3b8") Type { text("Walk the VIP to the limo. Block the attackers.", 18) } }

    Entity { id("limo") startPosition(60, 1440) color("e5e7eb") Type { rectangle(220, 100) }
        Events { OnElapsedTime(1.5) { repeatCount(-1) Actions { burst(4, "fde047") } } } }

    # Crowd barriers zig-zag the route; parked cars give some cover.
    Entity { id("barrier1") startPosition(0, 400) color("f59e0b") solid(true) Type { rectangle(600, 30) } }
    Entity { id("barrier2") startPosition(200, 800) color("f59e0b") solid(true) Type { rectangle(600, 30) } }
    Entity { id("barrier3") startPosition(0, 1200) color("f59e0b") solid(true) Type { rectangle(600, 30) } }
    Entity { id("car1") startPosition(300, 180) color("475569") solid(true) Type { rectangle(120, 60) } }
    Entity { id("car2") startPosition(420, 560) color("475569") solid(true) Type { rectangle(120, 60) } }
    Entity { id("car3") startPosition(250, 980) color("475569") solid(true) Type { rectangle(120, 60) } }

    # Attackers come out of the alleys and go straight for the VIP.
    Entity {
        id("assassin")
        template(true)
        color("dc2626")
        Type { rectangle(30, 30) }
        Movement { speed(1.5) chase("vip") }
        Events {
            OnCollision("vip") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("vip", "ff0000", 0.4)
                    shake(6, 0.3)
                    burst(14, "dc2626")
                    deleteEntity()
                }
            }
            OnElapsedTime(7) { Actions { burst(6, "7f1d1d") deleteEntity() } }
        }
    }

    # A sniper's laser dot sweeps the last block. It hurts the VIP, but the
    # agent can take the aim: the dot snaps back to its start.
    Entity {
        id("sniperDot")
        startPosition(300, 1260)
        color("f43f5e")
        Type { circle(12) }
        Movement {
            speed(2.2)
            Path {
                waypoint(300, 1260)
                waypoint(760, 1260)
                waypoint(760, 1560)
                waypoint(300, 1560)
                dwell(0.7)
            }
        }
        Events {
            OnCollision("vip") {
                Actions {
                    sound("laser")
                    updateBar("lives", -1)
                    flash("vip", "ff0000", 0.4)
                    shake(7, 0.4)
                }
            }
            OnCollision("agent") {
                Actions {
                    sound("clang")
                    burst(10, "f43f5e")
                    setPosition("sniperDot", 300, 1260, 0)
                }
            }
        }
    }

    Entity { id("firstAid") startPosition(620, 700) color("22c55e") Type { cross(36, 36, 10) }
        Events { OnCollision("vip") { Actions { sound("sparkle") updateBar("lives", 1) burst(12, "22c55e") deleteEntity() } } } }

    Entity { id("alley1") startPosition(770, 200) color("450a0a") Type { rectangle(30, 80) }
        Events { OnElapsedTime(random(3, 5)) { repeatCount(-1) Actions { spawnHere("assassin") flash("ef4444", 0.3) } } } }
    Entity { id("alley2") startPosition(0, 600) color("450a0a") Type { rectangle(30, 80) }
        Events { OnElapsedTime(random(4, 6)) { repeatCount(-1) Actions { spawnHere("assassin") flash("ef4444", 0.3) } } } }
    Entity { id("alley3") startPosition(770, 1000) color("450a0a") Type { rectangle(30, 80) }
        Events { OnElapsedTime(random(4, 6)) { repeatCount(-1) Actions { spawnHere("assassin") flash("ef4444", 0.3) } } } }
    Entity { id("alley4") startPosition(0, 1340) color("450a0a") Type { rectangle(30, 80) }
        Events { OnElapsedTime(random(3, 5)) { repeatCount(-1) Actions { spawnHere("assassin") flash("ef4444", 0.3) } } } }
}
