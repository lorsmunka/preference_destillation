Window {
    size(8, 12)
    background("0b1020")
    music("bg_2")
    gravity(0)

    Level {
        Entity { id("titleL1") startPosition(550, 30) color("64748b")
            Type { text("DUNGEON 1/3", 32) } }
        Entity {
            id("hero")
            startPosition(40, 40)
            color("4ade80")
            Type { rectangle(46, 46) }
            Movement { speed(5.95) Control { movementControl(0) } }
            Events {
                OnCollision("key") {
                    Actions {
                        sound("pickup")
                        flash("facc15", 0.25)
                        burst(12, "facc15")
                    }
                }
                OnSignal("unlocked") {
                    Actions { sound("clang") burst(14, "7c3aed") }
                }
                OnCollision("exit") {
                    Actions {
                        sound("win2")
                        flash("fbbf24", 0.3)
                        burst(18, "fbbf24")
                        nextLevel()
                    }
                }
            }
        }
        Entity { id("exit") startPosition(640, 1070) color("fbbf24")
            Type { rectangle(110, 110) }
            Events { OnElapsedTime(2.0) { repeatCount(-1)
                Actions { burst(5, "fbbf24") }
            } } }
        Entity { id("key") startPosition(372, 540) color("facc15")
            Type { star(5, 24, 11) } key("unlocked", "hero") }
        Entity { id("barrier") startPosition(0, 940) color("1f2937")
            solid(true) Type { rectangle(440, 34) } }
        Entity { id("door") startPosition(440, 940) color("7c3aed")
            Type { rectangle(360, 34) } door("unlocked") }
        Entity { id("wall1") startPosition(520, 380) color("1f2937")
            solid(true) Type { rectangle(34, 380) } }
        Entity { id("guard1") startPosition(0, 0) color("ef4444")
            Type { circle(26) } hostile("hero", "-- CAUGHT --", "lose")
            Movement { speed(2.55) Path { waypoint(180, 800) waypoint(700, 800)
                dwell(1) } }
            Events { OnCollision("hero") {
                Actions {
                    shake(6, 0.35)
                    flash("hero", "ff0000", 0.3)
                    burst(14, "ef4444")
                }
            } } }
        Entity { id("guard2") startPosition(0, 0) color("ef4444")
            Type { circle(26) } hostile("hero", "-- CAUGHT --", "lose")
            Movement { speed(2.55) Path { waypoint(60, 600) waypoint(440, 600)
                dwell(1) } }
            Events { OnCollision("hero") {
                Actions {
                    shake(6, 0.35)
                    flash("hero", "ff0000", 0.3)
                    burst(14, "ef4444")
                }
            } } }
        Entity { id("guard3") startPosition(0, 0) color("f97316")
            Type { circle(26) } hostile("hero", "-- CAUGHT --", "lose")
            Movement { speed(2.55) Path { waypoint(60, 280) waypoint(700, 280) dwell(1) } }
            Events { OnCollision("hero") {
                Actions {
                    shake(6, 0.35)
                    flash("hero", "ff0000", 0.3)
                    burst(14, "f97316")
                }
            } } }
    }

    Level {
        Entity { id("titleL2") startPosition(550, 30) color("64748b")
            Type { text("DUNGEON 2/3", 32) } }
        Entity {
            id("hero")
            startPosition(40, 40)
            color("4ade80")
            Type { rectangle(46, 46) }
            Movement { speed(5.95) Control { movementControl(0) } }
            Events {
                OnCollision("key") {
                    Actions {
                        sound("pickup")
                        flash("facc15", 0.25)
                        burst(12, "facc15")
                    }
                }
                OnSignal("unlocked") {
                    Actions { sound("clang") burst(14, "7c3aed") }
                }
                OnCollision("exit") {
                    Actions {
                        sound("win2")
                        flash("fbbf24", 0.3)
                        burst(18, "fbbf24")
                        nextLevel()
                    }
                }
            }
        }
        Entity { id("exit") startPosition(640, 1070) color("fbbf24")
            Type { rectangle(110, 110) }
            Events { OnElapsedTime(2.0) { repeatCount(-1)
                Actions { burst(5, "fbbf24") }
            } } }
        Entity { id("key") startPosition(180, 800) color("facc15")
            Type { star(5, 24, 11) } key("unlocked", "hero") }
        Entity { id("barrier") startPosition(0, 940) color("1f2937")
            solid(true) Type { rectangle(440, 34) } }
        Entity { id("door") startPosition(440, 940) color("7c3aed")
            Type { rectangle(360, 34) } door("unlocked") }
        Entity { id("wall1") startPosition(220, 300) color("1f2937")
            solid(true) Type { rectangle(280, 34) } }
        Entity { id("wall2") startPosition(420, 500) color("1f2937")
            solid(true) Type { rectangle(34, 300) } }
        Entity { id("guard1") startPosition(0, 0) color("ef4444")
            Type { circle(26) } hostile("hero", "-- CAUGHT --", "lose")
            Movement { speed(2.55) Path { waypoint(500, 600) waypoint(700, 600)
                dwell(1) } }
            Events { OnCollision("hero") {
                Actions {
                    shake(6, 0.35)
                    flash("hero", "ff0000", 0.3)
                    burst(14, "ef4444")
                }
            } } }
        Entity { id("guard2") startPosition(0, 0) color("ef4444")
            Type { circle(26) } hostile("hero", "-- CAUGHT --", "lose")
            Movement { speed(2.55) Path { waypoint(60, 500) waypoint(380, 500)
                dwell(1) } }
            Events { OnCollision("hero") {
                Actions {
                    shake(6, 0.35)
                    flash("hero", "ff0000", 0.3)
                    burst(14, "ef4444")
                }
            } } }
        Entity { id("guard3") startPosition(0, 0) color("f97316")
            Type { circle(26) } hostile("hero", "-- CAUGHT --", "lose")
            Movement { speed(3.4) Path { waypoint(60, 200) waypoint(700, 200) dwell(1) } }
            Events { OnCollision("hero") {
                Actions {
                    shake(6, 0.35)
                    flash("hero", "ff0000", 0.3)
                    burst(14, "f97316")
                }
            } } }
        Entity { id("guard4") startPosition(0, 0) color("f97316")
            Type { circle(26) } hostile("hero", "-- CAUGHT --", "lose")
            Movement { speed(2.55) Path { waypoint(60, 700) waypoint(700, 700) dwell(1) } }
            Events { OnCollision("hero") {
                Actions {
                    shake(6, 0.35)
                    flash("hero", "ff0000", 0.3)
                    burst(14, "f97316")
                }
            } } }
    }

    Level {
        Entity { id("titleL3") startPosition(550, 30) color("64748b")
            Type { text("DUNGEON 3/3", 32) } }
        Entity {
            id("hero")
            startPosition(40, 40)
            color("4ade80")
            Type { rectangle(46, 46) }
            Movement { speed(5.95) Control { movementControl(0) } }
            Events {
                OnCollision("key") {
                    Actions {
                        sound("pickup")
                        flash("facc15", 0.25)
                        burst(12, "facc15")
                    }
                }
                OnSignal("unlocked") {
                    Actions { sound("clang") burst(14, "7c3aed") }
                }
                OnCollision("exit") {
                    Actions {
                        sound("win")
                        flash("fbbf24", 0.4)
                        shake(5, 0.4)
                        burst(20, "fbbf24")
                        gameOver("-- ESCAPED THE DUNGEON --")
                    }
                }
            }
        }
        Entity { id("exit") startPosition(640, 1070) color("fbbf24")
            Type { rectangle(110, 110) }
            Events { OnElapsedTime(2.0) { repeatCount(-1)
                Actions { burst(5, "fbbf24") }
            } } }
        Entity { id("key") startPosition(700, 400) color("facc15")
            Type { star(5, 24, 11) } key("unlocked", "hero") }
        Entity { id("barrier") startPosition(0, 940) color("1f2937")
            solid(true) Type { rectangle(440, 34) } }
        Entity { id("door") startPosition(440, 940) color("7c3aed")
            Type { rectangle(360, 34) } door("unlocked") }
        Entity { id("wall1") startPosition(60, 300) color("1f2937")
            solid(true) Type { rectangle(400, 34) } }
        Entity { id("wall2") startPosition(300, 500) color("1f2937")
            solid(true) Type { rectangle(34, 300) } }
        Entity { id("wall3") startPosition(500, 700) color("1f2937")
            solid(true) Type { rectangle(34, 200) } }
        Entity { id("guard1") startPosition(0, 0) color("ef4444")
            Type { circle(26) } hostile("hero", "-- CAUGHT --", "lose")
            Movement { speed(3.4) Path { waypoint(60, 800) waypoint(440, 800)
                dwell(1) } }
            Events { OnCollision("hero") {
                Actions {
                    shake(6, 0.35)
                    flash("hero", "ff0000", 0.3)
                    burst(14, "ef4444")
                }
            } } }
        Entity { id("guard2") startPosition(0, 0) color("ef4444")
            Type { circle(26) } hostile("hero", "-- CAUGHT --", "lose")
            Movement { speed(3.4) Path { waypoint(60, 400) waypoint(700, 400)
                dwell(1) } }
            Events { OnCollision("hero") {
                Actions {
                    shake(6, 0.35)
                    flash("hero", "ff0000", 0.3)
                    burst(14, "ef4444")
                }
            } } }
        Entity { id("guard3") startPosition(0, 0) color("ef4444")
            Type { circle(26) } hostile("hero", "-- CAUGHT --", "lose")
            Movement { speed(3.4) Path { waypoint(60, 600) waypoint(620, 600)
                dwell(1) } }
            Events { OnCollision("hero") {
                Actions {
                    shake(6, 0.35)
                    flash("hero", "ff0000", 0.3)
                    burst(14, "ef4444")
                }
            } } }
        Entity { id("guard4") startPosition(0, 0) color("f97316")
            Type { circle(26) } hostile("hero", "-- CAUGHT --", "lose")
            Movement { speed(3.4) Path { waypoint(60, 200) waypoint(700, 200) dwell(1) } }
            Events { OnCollision("hero") {
                Actions {
                    shake(6, 0.35)
                    flash("hero", "ff0000", 0.3)
                    burst(14, "f97316")
                }
            } } }
    }
}
