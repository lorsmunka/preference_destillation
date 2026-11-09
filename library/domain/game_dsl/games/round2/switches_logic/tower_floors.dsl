Window {
    size(8, 12)
    background("0b1020")
    music("bg_6")
    gravity(0)
    Bar("plates") {
        initial(0)
        OnValue(3) { Actions { broadcast("allOn") } }
    }

    # floor 1: take the key, open the shutter, pull the lever, pass the beam
    Level {
        Entity { id("title1") startPosition(500, 30) color("64748b") Type { text("FLOOR 1 / 2", 32) } }
        Entity {
            id("hero")
            startPosition(40, 80)
            color("4ade80")
            Type { rectangle(44, 44) }
            Movement { speed(5) Control { movementControl(0) } }
            Events {
                OnSignal("shutterKey") { Actions { sound("clang") flash("facc15", 0.3) } }
                OnCollision("stairs") {
                    Actions { sound("win2") burst(18, "fbbf24") nextLevel() }
                }
            }
        }
        Entity { id("key") startPosition(680, 320) color("facc15") Type { star(5, 28, 13) } key("shutterKey", "hero", "pickup") }
        Entity {
            id("sweeper")
            startPosition(60, 220)
            color("ef4444")
            Type { circle(26) }
            hostile("hero", "-- CAUGHT --", "lose")
            Movement { speed(3.2) Path { waypoint(60, 220) waypoint(720, 220) } }
        }
        Entity {
            id("sweeper2")
            startPosition(720, 380)
            color("f97316")
            Type { circle(22) }
            hostile("hero", "-- CAUGHT --", "lose")
            Movement { speed(2.4) Path { waypoint(720, 380) waypoint(300, 380) dwell(0.6) } }
        }
        Entity { id("shutter") startPosition(0, 520) color("7c3aed") Type { rectangle(800, 24) } door("shutterKey", "clang") }
        Entity {
            id("lever")
            startPosition(60, 660)
            color("ef4444")
            Type { rectangle(50, 90) }
            var("pulled", 0)
            Events {
                OnCollision("hero") {
                    Actions {
                        sound("click")
                        whenVar("pulled", "==", 0) { Actions { delay(0.02) { Actions {
                            changeVar("pulled", 1)
                            setColor("4ade80")
                            setEnable("beam", false)
                            setVisible("beam", false)
                        } } } }
                        whenVar("pulled", "==", 1) { Actions { delay(0.02) { Actions {
                            changeVar("pulled", -1)
                            setColor("ef4444")
                            setEnable("beam", true)
                            setVisible("beam", true)
                        } } } }
                    }
                }
            }
        }
        Entity {
            id("beam")
            startPosition(0, 860)
            color("fb7185")
            Type { rectangle(800, 14) }
            hostile("hero", "-- ZAPPED --", "lose")
        }
        Entity { id("stairs") startPosition(340, 1040) color("fbbf24") Type { rectangle(120, 90) } }
    }

    # floor 2: three plates open the vault door
    Level {
        Entity { id("title2") startPosition(500, 30) color("64748b") Type { text("FLOOR 2 / 2", 32) } }
        Entity { id("stepUp") startPosition(350, 250) color("64748b") Type { triangle(60, "up") }
            Events { OnTapped("stepUp") { Actions { gridStep("hero", 0, 1) } } } }
        Entity { id("stepDown") startPosition(350, 60) color("64748b") Type { triangle(60, "down") }
            Events { OnTapped("stepDown") { Actions { gridStep("hero", 0, -1) } } } }
        Entity { id("stepLeft") startPosition(210, 150) color("64748b") Type { triangle(60, "left") }
            Events { OnTapped("stepLeft") { Actions { gridStep("hero", -1, 0) } } } }
        Entity { id("stepRight") startPosition(490, 150) color("64748b") Type { triangle(60, "right") }
            Events { OnTapped("stepRight") { Actions { gridStep("hero", 1, 0) } } } }
        Entity { id("wallL1") startPosition(60, 940) color("1f2937") solid(true) Type { rectangle(100, 100) } }
        Entity { id("wallL2") startPosition(160, 940) color("1f2937") solid(true) Type { rectangle(100, 100) } }
        Entity { id("wallL3") startPosition(260, 940) color("1f2937") solid(true) Type { rectangle(100, 100) } }
        Entity { id("wallR1") startPosition(460, 940) color("1f2937") solid(true) Type { rectangle(100, 100) } }
        Entity { id("wallR2") startPosition(560, 940) color("1f2937") solid(true) Type { rectangle(100, 100) } }
        Entity { id("wallR3") startPosition(660, 940) color("1f2937") solid(true) Type { rectangle(100, 100) } }
        Entity { id("vaultDoor") startPosition(360, 940) color("db2777") Type { rectangle(100, 100) } door("allOn", "clang") }
        Entity {
            id("plateA")
            startPosition(60, 640)
            color("92400e")
            Type { rectangle(100, 100) }
            var("used", 0)
            Events {
                OnCollision("hero") {
                    Actions {
                        whenVar("used", "==", 0) {
                            Actions { changeVar("used", 1) setColor("4ade80") sound("pickup") updateBar("plates", 1) }
                        }
                    }
                }
            }
        }
        Entity {
            id("plateB")
            startPosition(660, 640)
            color("92400e")
            Type { rectangle(100, 100) }
            var("used", 0)
            Events {
                OnCollision("hero") {
                    Actions {
                        whenVar("used", "==", 0) {
                            Actions { changeVar("used", 1) setColor("4ade80") sound("pickup") updateBar("plates", 1) }
                        }
                    }
                }
            }
        }
        Entity {
            id("plateC")
            startPosition(360, 840)
            color("92400e")
            Type { rectangle(100, 100) }
            var("used", 0)
            Events {
                OnCollision("hero") {
                    Actions {
                        whenVar("used", "==", 0) {
                            Actions { changeVar("used", 1) setColor("4ade80") sound("pickup") updateBar("plates", 1) }
                        }
                    }
                }
            }
        }
        Entity {
            id("patrol")
            startPosition(160, 740)
            color("ef4444")
            Type { circle(34) }
            hostile("hero", "-- CAUGHT --", "lose")
            Movement { speed(2.6) Path { waypoint(160, 740) waypoint(560, 740) dwell(0.5) } }
        }
        goal("exit", 360, 1040, 100, 100, "-- TOWER CLEARED --")
        Entity {
            id("hero")
            startPosition(60, 440)
            color("22d3ee")
            Type { circle(40) }
            Movement { GridMove(100, 0.3) { direction(0, 0) } }
            Events {
                OnGridBlocked { Actions { sound("bump") setDirection("hero", 0, 0) } }
            }
        }
    }
}
