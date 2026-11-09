# Three cave levels. Collect enough treasure to open the gate, then reach the ladder.
Window {
    size(8, 12)
    background("1c1917")
    music("bg_6")
    gravity(0)

    Bar("treasure") {
        initial(0)
        OnValue(5) { Actions { sound("clang") broadcast("openGate") } }
        OnValue(11) { Actions { sound("clang") broadcast("openGate") } }
    }
    winAt("treasure", 20, "-- THE CAVE IS YOURS --", "win")
    lives(3, "-- LOST IN THE DARK --", "lose")

    Level {
        wallBox(0, 0, 800, 1200, "44403c", 20)
        Entity {
            id("hero")
            startPosition(60, 60)
            color("fbbf24")
            Type { circle(26) }
            Movement { speed(5) Control { movementControl(0) } }
            Events { OnCollision("ladder") { Actions { sound("jump") nextLevel() } } }
        }
        Entity {
            id("gate")
            startPosition(20, 700)
            color("78350f")
            solid(true)
            Type { rectangle(760, 30) }
            Events { OnSignal("openGate") { Actions { deleteEntity() } } }
        }
        Entity {
            id("ladder")
            startPosition(360, 1060)
            color("a16207")
            Type { rectangle(60, 100) }
        }
        Entity { id("gem1") startPosition(100, 300) color("22d3ee") Type { diamond(32, 40) }
            Events { OnCollision("hero") { Actions { sound("pickup") updateBar("treasure", 1) deleteEntity() } } } }
        Entity { id("gem2") startPosition(600, 200) color("22d3ee") Type { diamond(32, 40) }
            Events { OnCollision("hero") { Actions { sound("pickup") updateBar("treasure", 1) deleteEntity() } } } }
        Entity { id("gem3") startPosition(300, 520) color("22d3ee") Type { diamond(32, 40) }
            Events { OnCollision("hero") { Actions { sound("pickup") updateBar("treasure", 1) deleteEntity() } } } }
        Entity { id("gem4") startPosition(620, 560) color("22d3ee") Type { diamond(32, 40) }
            Events { OnCollision("hero") { Actions { sound("pickup") updateBar("treasure", 1) deleteEntity() } } } }
        Entity { id("gem5") startPosition(120, 600) color("22d3ee") Type { diamond(32, 40) }
            Events { OnCollision("hero") { Actions { sound("pickup") updateBar("treasure", 1) deleteEntity() } } } }
        HostilePatrol("bat1", "hero", "-- BITTEN --", 2, "c084fc") {
            waypoint(100, 450)
            waypoint(680, 450)
        }
    }

    Level {
        wallBox(0, 0, 800, 1200, "44403c", 20)
        Entity {
            id("hero")
            startPosition(60, 60)
            color("fbbf24")
            Type { circle(26) }
            Movement { speed(5) Control { movementControl(0) } }
            Events { OnCollision("ladder") { Actions { sound("jump") nextLevel() } } }
        }
        Entity {
            id("gate")
            startPosition(20, 800)
            color("78350f")
            solid(true)
            Type { rectangle(760, 30) }
            Events { OnSignal("openGate") { Actions { deleteEntity() } } }
        }
        Entity {
            id("ladder")
            startPosition(60, 1060)
            color("a16207")
            Type { rectangle(60, 100) }
        }
        Entity {
            id("gem")
            template(true)
            startPosition(0, 0)
            color("4ade80")
            Type { diamond(32, 40) }
            Events { OnCollision("hero") { Actions { sound("pickup") updateBar("treasure", 1) deleteEntity() } } }
        }
        Entity {
            id("keeper")
            startPosition(10, 10)
            color("00000000")
            Type { circle(1) }
            Events {
                OnElapsedTime { Actions {
                    spawnAtEmpty("gem", 120, 100, 100, 6, 6)
                    spawnAtEmpty("gem", 120, 100, 100, 6, 6)
                    spawnAtEmpty("gem", 120, 100, 100, 6, 6)
                } }
                OnElapsedTime(2.5) {
                    repeatCount(3)
                    Actions { spawnAtEmpty("gem", 120, 100, 100, 6, 6) }
                }
            }
        }
        HostilePatrol("bat2", "hero", "-- BITTEN --", 3, "c084fc") {
            waypoint(100, 300)
            waypoint(700, 300)
            waypoint(700, 650)
            waypoint(100, 650)
        }
        HostilePatrol("bat3", "hero", "-- BITTEN --", 2, "e879f9") {
            waypoint(700, 1000)
            waypoint(200, 1000)
            dwell(0.8)
        }
    }

    Level {
        wallBox(0, 0, 800, 1200, "44403c", 20)
        Entity {
            id("hero")
            startPosition(380, 60)
            color("fbbf24")
            Type { circle(26) }
            Movement { speed(5.5) Control { movementControl(0) } }
        }
        Entity {
            id("coin")
            template(true)
            startPosition(0, 0)
            color("facc15")
            Type { circle(18) }
            Movement { speed(3) AutoMove { direction(0, -1) } }
            Events {
                OnCollision("hero") {
                    Actions { sound("coin") updateBar("treasure", 1) burst(6, "facc15") deleteEntity() }
                }
                OnCollision { Actions { deleteEntity() } }
            }
        }
        Entity {
            id("rock")
            template(true)
            startPosition(0, 0)
            color("78716c")
            Type { polygon(6, 30) }
            Movement { speed(3.6) AutoMove { direction(0, -1) } }
            Events {
                OnCollision("hero") {
                    Actions { sound("hurt") shake(6, 0.3) updateBar("lives", -1) deleteEntity() }
                }
                OnCollision { Actions { deleteEntity() } }
            }
        }
        Wave(10, 0.8) { Actions { spawn("coin", random(60, 720), 1140) } }
        Wave(25, 0.5) {
            Actions {
                spawn("coin", random(60, 720), 1140)
                spawn("rock", random(60, 720), 1140)
            }
        }
    }
}
