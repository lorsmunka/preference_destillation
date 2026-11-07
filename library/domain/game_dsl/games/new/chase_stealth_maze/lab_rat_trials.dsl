Window {
    size(8, 12)
    background("0f1e2e")
    music("bg_1")
    gravity(0)

    # Every maze is an 8 x 8 grid of 100 px cells, rows from y 400 up to 1200.

    # ---------- trial 1: one key, one door, one handler ----------
    Level {
        Entity { id("title1") startPosition(630, 250) color("94a3b8") Type { text("TRIAL 1/3", 28) }
            Events { OnElapsedTime { Actions { sound("beep") flash("e0f2fe", 0.4) } } } }
        dpad("rat")

        Entity {
            id("rat")
            startPosition(10, 410)
            color("e5e7eb")
            Type { circle(40) }
            Movement { GridMove(100, 0.28) { direction(0, 0) } }
            Events {
                OnSignal("labDoor") { Actions { flash("67e8f9", 0.3) burst(10, "67e8f9") } }
                OnCollision("cheese") {
                    Actions {
                        sound("win2")
                        flash("facc15", 0.3)
                        burst(16, "facc15")
                        nextLevel()
                    }
                }
            }
        }

        Entity { id("wall1") startPosition(100, 800) color("334155") solid(true) Type { rectangle(600, 100) } }
        Entity { id("wall2") startPosition(0, 600) color("334155") solid(true) Type { rectangle(300, 100) } }
        Entity { id("wall3") startPosition(400, 600) color("334155") solid(true) Type { rectangle(300, 100) } }
        Entity { id("wall4") startPosition(200, 400) color("334155") solid(true) Type { rectangle(300, 100) } }
        Entity { id("wall5") startPosition(600, 500) color("334155") solid(true) Type { rectangle(100, 100) } }
        Entity { id("wall6") startPosition(100, 900) color("334155") solid(true) Type { rectangle(100, 200) } }
        Entity { id("wall7") startPosition(300, 1000) color("334155") solid(true) Type { rectangle(100, 200) } }
        Entity { id("wall8") startPosition(500, 1000) color("334155") solid(true) Type { rectangle(100, 200) } }
        Entity { id("wall9") startPosition(700, 1000) color("334155") solid(true) Type { rectangle(100, 100) } }

        Entity { id("labKey") startPosition(25, 1125) color("22d3ee") Type { rectangle(50, 50) }
            key("labDoor", "rat", "pickup") }
        Entity { id("labGate") startPosition(400, 1000) color("0e7490") Type { rectangle(100, 100) }
            door("labDoor", "clang") }
        Entity { id("cheese") startPosition(420, 1125) color("facc15") Type { rectangle(60, 50) }
            Events { OnElapsedTime(1.5) { repeatCount(-1) Actions { burst(4, "fde047") } } } }

        HostilePatrol("handler1", "rat", "-- CAUGHT BY THE HANDLER --", 2.2, "f87171") {
            waypoint(24, 724)
            waypoint(724, 724)
            dwell(1)
        }
    }

    # ---------- trial 2: two keys, two doors, two handlers ----------
    Level {
        Entity { id("title2") startPosition(630, 250) color("94a3b8") Type { text("TRIAL 2/3", 28) }
            Events { OnElapsedTime { Actions { sound("beep") flash("e0f2fe", 0.4) } } } }
        dpad("rat")

        Entity {
            id("rat")
            startPosition(10, 410)
            color("e5e7eb")
            Type { circle(40) }
            Movement { GridMove(100, 0.28) { direction(0, 0) } }
            Events {
                OnSignal("doorA") { Actions { flash("67e8f9", 0.3) burst(10, "67e8f9") } }
                OnSignal("doorB") { Actions { flash("a78bfa", 0.3) burst(10, "a78bfa") } }
                OnCollision("cheese") {
                    Actions {
                        sound("win2")
                        flash("facc15", 0.3)
                        burst(16, "facc15")
                        nextLevel()
                    }
                }
            }
        }

        Entity { id("wall1") startPosition(0, 800) color("334155") solid(true) Type { rectangle(300, 100) } }
        Entity { id("wall2") startPosition(400, 800) color("334155") solid(true) Type { rectangle(400, 100) } }
        Entity { id("wall3") startPosition(300, 1000) color("334155") solid(true) Type { rectangle(100, 200) } }
        Entity { id("wall4") startPosition(100, 900) color("334155") solid(true) Type { rectangle(100, 200) } }
        Entity { id("wall5") startPosition(500, 1000) color("334155") solid(true) Type { rectangle(200, 100) } }
        Entity { id("wall6") startPosition(600, 1100) color("334155") solid(true) Type { rectangle(100, 100) } }
        Entity { id("wall7") startPosition(100, 600) color("334155") solid(true) Type { rectangle(300, 100) } }
        Entity { id("wall8") startPosition(500, 600) color("334155") solid(true) Type { rectangle(200, 100) } }
        Entity { id("wall9") startPosition(300, 500) color("334155") solid(true) Type { rectangle(100, 100) } }
        Entity { id("wall10") startPosition(100, 400) color("334155") solid(true) Type { rectangle(100, 100) } }
        Entity { id("wall11") startPosition(500, 400) color("334155") solid(true) Type { rectangle(100, 100) } }

        Entity { id("keyA") startPosition(625, 425) color("22d3ee") Type { rectangle(50, 50) }
            key("doorA", "rat", "pickup") }
        Entity { id("gateA") startPosition(300, 800) color("0e7490") Type { rectangle(100, 100) }
            door("doorA", "clang") }
        Entity { id("keyB") startPosition(25, 1125) color("c4b5fd") Type { rectangle(50, 50) }
            key("doorB", "rat", "pickup") }
        Entity { id("gateB") startPosition(700, 1000) color("6d28d9") Type { rectangle(100, 100) }
            door("doorB", "clang") }
        Entity { id("cheese") startPosition(720, 1125) color("facc15") Type { rectangle(60, 50) }
            Events { OnElapsedTime(1.5) { repeatCount(-1) Actions { burst(4, "fde047") } } } }

        HostilePatrol("handler1", "rat", "-- CAUGHT BY THE HANDLER --", 2.4, "f87171") {
            waypoint(24, 724)
            waypoint(724, 724)
            dwell(0.8)
        }
        HostilePatrol("handler2", "rat", "-- CAUGHT BY THE HANDLER --", 1.6, "fb923c") {
            waypoint(424, 924)
            waypoint(424, 1124)
            dwell(1)
        }
    }

    # ---------- trial 3: the lab cat is loose ----------
    Level {
        Entity { id("title3") startPosition(630, 250) color("94a3b8") Type { text("TRIAL 3/3", 28) }
            Events { OnElapsedTime { Actions { sound("beep") flash("e0f2fe", 0.4) } } } }
        dpad("rat")

        Entity {
            id("rat")
            startPosition(10, 410)
            color("e5e7eb")
            Type { circle(40) }
            Movement { GridMove(100, 0.28) { direction(0, 0) } }
            Events {
                OnSignal("cageDoor") { Actions { flash("67e8f9", 0.3) burst(10, "67e8f9") } }
                OnCollision("cheese") {
                    Actions {
                        sound("win")
                        shake(5, 0.4)
                        burst(24, "facc15")
                        gameOver("-- ALL THREE TRIALS PASSED --")
                    }
                }
            }
        }

        Entity { id("wall1") startPosition(600, 600) color("334155") solid(true) Type { rectangle(100, 600) } }
        Entity { id("wall2") startPosition(500, 400) color("334155") solid(true) Type { rectangle(200, 100) } }
        Entity { id("wall3") startPosition(100, 400) color("334155") solid(true) Type { rectangle(100, 100) } }
        Entity { id("wall4") startPosition(300, 500) color("334155") solid(true) Type { rectangle(100, 100) } }
        Entity { id("wall5") startPosition(0, 600) color("334155") solid(true) Type { rectangle(200, 100) } }
        Entity { id("wall6") startPosition(300, 600) color("334155") solid(true) Type { rectangle(200, 100) } }
        Entity { id("wall7") startPosition(100, 800) color("334155") solid(true) Type { rectangle(400, 100) } }
        Entity { id("wall8") startPosition(100, 900) color("334155") solid(true) Type { rectangle(100, 200) } }
        Entity { id("wall9") startPosition(200, 1000) color("334155") solid(true) Type { rectangle(100, 100) } }
        Entity { id("wall10") startPosition(400, 900) color("334155") solid(true) Type { rectangle(100, 300) } }

        Entity { id("cageKey") startPosition(225, 925) color("22d3ee") Type { rectangle(50, 50) }
            key("cageDoor", "rat", "pickup") }
        Entity { id("cageGate") startPosition(700, 1000) color("0e7490") Type { rectangle(100, 100) }
            door("cageDoor", "clang") }
        Entity { id("cheese") startPosition(720, 1125) color("facc15") Type { rectangle(60, 50) }
            Events { OnElapsedTime(1.5) { repeatCount(-1) Actions { burst(4, "fde047") } } } }

        HostilePatrol("handler1", "rat", "-- CAUGHT BY THE HANDLER --", 2.2, "f87171") {
            waypoint(24, 724)
            waypoint(524, 724)
            dwell(0.8)
        }
        HostilePatrol("handler2", "rat", "-- CAUGHT BY THE HANDLER --", 1.8, "fb923c") {
            waypoint(724, 524)
            waypoint(724, 924)
            dwell(0.8)
        }

        # The cat glides after the rat but cannot pass the maze walls.
        Entity {
            id("cat")
            startPosition(520, 1120)
            color("f59e0b")
            Type { circle(30) }
            hostile("rat", "-- THE LAB CAT POUNCED --", "lose")
            Movement {
                speed(1)
                chase("rat")
            }
            Events {
                OnCollision("rat") { Actions { shake(8, 0.5) burst(20, "f59e0b") } }
                OnSignal("catnip") {
                    Actions {
                        retarget("toyMouse")
                        sound("pop")
                        delay(5) { Actions { retarget("rat") flash("ef4444", 0.4) } }
                    }
                }
            }
        }

        # Catnip sends the cat off after the wind-up mouse for five seconds.
        Entity { id("catnip") startPosition(425, 425) color("4ade80") Type { circle(16) }
            Events { OnCollision("rat") { Actions { sound("sparkle") burst(14, "4ade80") broadcast("catnip") deleteEntity() } } } }
        Entity { id("toyMouse") startPosition(535, 935) color("fda4af") Type { circle(15) }
            Movement { speed(1.5) Path { waypoint(535, 935) waypoint(535, 1135) dwell(0.4) } } }
    }
}
