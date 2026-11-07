Window {
    size(8, 12)
    background("070b1f")
    music("bg_4")
    gravity(0)

    Bar("orbs") {
        initial(0)
        OnValue(8) {
            Actions {
                sound("win")
                shake(5, 0.4)
                burst("ship", 24, "fbbf24")
                gameOver("-- ALL ORBS RECOVERED --")
            }
        }
    }
    lives(3, "-- DRONES GOT YOU --", "lose")

    Entity { id("bg1") startPosition(150, 900) color("ffffff")
        Type { star(5, 4, 2) }
        Movement { speed(0.85) AutoMove { direction(0, -1) } }
        Events { OnCollision {
            Actions { setPosition("bg1", 150, 1180, 0) } } } }
    Entity { id("bg2") startPosition(620, 600) color("a5b4fc")
        Type { star(4, 3, 1) }
        Movement { speed(0.85) AutoMove { direction(0, -1) } }
        Events { OnCollision {
            Actions { setPosition("bg2", 620, 1180, 0) } } } }
    Entity { id("bg3") startPosition(380, 1050) color("ffffff")
        Type { star(5, 3, 1) }
        Movement { speed(0.85) AutoMove { direction(0, -1) } }
        Events { OnCollision {
            Actions { setPosition("bg3", 380, 1180, 0) } } } }

    Entity {
        id("ship")
        startPosition(372, 560)
        color("4ade80")
        Type { circle(28) }
    }

    Entity { id("orb1") startPosition(120, 1060) color("fbbf24")
        Type { circle(16) } Events { OnCollision("ship") { Actions {
        sound("pickup") updateBar("orbs", 1) burst(8, "fbbf24")
        flash("ship", "fbbf24", 0.2) deleteEntity() } } } }
    Entity { id("orb2") startPosition(380, 1100) color("fbbf24")
        Type { circle(16) } Events { OnCollision("ship") { Actions {
        sound("pickup") updateBar("orbs", 1) burst(8, "fbbf24")
        flash("ship", "fbbf24", 0.2) deleteEntity() } } } }
    Entity { id("orb3") startPosition(660, 1040) color("fbbf24")
        Type { circle(16) } Events { OnCollision("ship") { Actions {
        sound("pickup") updateBar("orbs", 1) burst(8, "fbbf24")
        flash("ship", "fbbf24", 0.2) deleteEntity() } } } }
    Entity { id("orb4") startPosition(140, 780) color("fbbf24")
        Type { circle(16) } Events { OnCollision("ship") { Actions {
        sound("pickup") updateBar("orbs", 1) burst(8, "fbbf24")
        flash("ship", "fbbf24", 0.2) deleteEntity() } } } }
    Entity { id("orb5") startPosition(640, 740) color("fbbf24")
        Type { circle(16) } Events { OnCollision("ship") { Actions {
        sound("pickup") updateBar("orbs", 1) burst(8, "fbbf24")
        flash("ship", "fbbf24", 0.2) deleteEntity() } } } }
    Entity { id("orb6") startPosition(300, 560) color("fbbf24")
        Type { circle(16) } Events { OnCollision("ship") { Actions {
        sound("pickup") updateBar("orbs", 1) burst(8, "fbbf24")
        flash("ship", "fbbf24", 0.2) deleteEntity() } } } }
    Entity { id("orb7") startPosition(700, 520) color("fbbf24")
        Type { circle(16) } Events { OnCollision("ship") { Actions {
        sound("pickup") updateBar("orbs", 1) burst(8, "fbbf24")
        flash("ship", "fbbf24", 0.2) deleteEntity() } } } }
    Entity { id("orb8") startPosition(420, 300) color("fbbf24")
        Type { circle(16) } Events { OnCollision("ship") { Actions {
        sound("pickup") updateBar("orbs", 1) burst(8, "fbbf24")
        flash("ship", "fbbf24", 0.2) deleteEntity() } } } }

    Entity {
        id("drone1")
        startPosition(80, 900)
        color("f43f5e")
        Type { circle(30) }
        Movement { speed(1.7) Path { waypoint(80, 900) waypoint(680, 900) } }
        Events {
            OnCollision("ship") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("ship", "ff0000", 0.25)
                    shake(6, 0.3)
                    burst(12, "f43f5e")
                    setPosition("drone1", 80, 900, 0)
                }
            }
        }
    }

    Entity {
        id("drone2")
        startPosition(680, 420)
        color("22d3ee")
        Type { circle(30) }
        Movement { speed(1.7) Path { waypoint(680, 420) waypoint(80, 420) } }
        Events {
            OnCollision("ship") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("ship", "ff0000", 0.25)
                    shake(6, 0.3)
                    burst(12, "22d3ee")
                    setPosition("drone2", 680, 420, 0)
                }
            }
        }
    }

    joystick("ship", 90, 4.25)
}
