Window {
    size(8, 12)
    background("0f172a")
    music("bg_2")
    gravity(0)

    Level {
        Entity { id("titleL1") startPosition(640, 130) color("64748b")
            Type { text("MAZE 1/3", 32) }
            Events { OnElapsedTime(0) {
                Actions { sound("bell") flash("fbbf24", 0.5) } } } }
        dpad("runner")

        Entity { id("wall1") startPosition(300, 560) color("475569")
            solid(true) Type { rectangle(200, 200) } }
        Entity { id("guard1") startPosition(120, 840) color("ef4444")
            Type { diamond(90, 90) }
            Movement { speed(3.4) Path { waypoint(120, 840) waypoint(620, 840) } } }
        Entity { id("exit") startPosition(580, 1030) color("4ade80")
            Type { star(5, 60, 28) } }

        Entity {
            id("runner")
            startPosition(60, 440)
            color("22d3ee")
            Type { circle(40) }
            Movement { GridMove(100, 0.3){ direction(0, 0) } }
            Events {
                OnCollision("exit") {
                    Actions {
                        sound("win2")
                        flash("4ade80", 0.3)
                        burst(12, "4ade80")
                        nextLevel()
                    }
                }
                OnCollision("guard1") {
                    Actions {
                        sound("lose")
                        flash("ff0000", 0.25)
                        shake(6, 0.3)
                        burst(14, "ef4444")
                        gameOver("-- CAUGHT --")
                    }
                }
            }
        }
    }

    Level {
        Entity { id("titleL2") startPosition(640, 130) color("64748b")
            Type { text("MAZE 2/3", 32) }
            Events { OnElapsedTime(0) {
                Actions { sound("bell") flash("fbbf24", 0.5) } } } }
        dpad("runner")

        Entity { id("wall1") startPosition(300, 560) color("475569")
            solid(true) Type { rectangle(200, 200) } }
        Entity { id("wall2") startPosition(160, 800) color("475569")
            solid(true) Type { rectangle(140, 80) } }
        Entity { id("guard1") startPosition(120, 640) color("ef4444")
            Type { diamond(90, 90) }
            Movement { speed(3.4) Path { waypoint(120, 640) waypoint(620, 640) } } }
        Entity { id("guard2") startPosition(620, 940) color("ef4444")
            Type { diamond(90, 90) }
            Movement { speed(3.4) Path { waypoint(620, 940) waypoint(120, 940) } } }
        Entity { id("exit") startPosition(580, 1030) color("4ade80")
            Type { star(5, 60, 28) } }

        Entity {
            id("runner")
            startPosition(60, 440)
            color("22d3ee")
            Type { circle(40) }
            Movement { GridMove(100, 0.3){ direction(0, 0) } }
            Events {
                OnCollision("exit") {
                    Actions {
                        sound("win2")
                        flash("4ade80", 0.3)
                        burst(12, "4ade80")
                        nextLevel()
                    }
                }
                OnCollision("guard1") {
                    Actions {
                        sound("lose")
                        flash("ff0000", 0.25)
                        shake(6, 0.3)
                        burst(14, "ef4444")
                        gameOver("-- CAUGHT --")
                    }
                }
                OnCollision("guard2") {
                    Actions {
                        sound("lose")
                        flash("ff0000", 0.25)
                        shake(6, 0.3)
                        burst(14, "ef4444")
                        gameOver("-- CAUGHT --")
                    }
                }
            }
        }
    }

    Level {
        Entity { id("titleL3") startPosition(640, 130) color("64748b")
            Type { text("MAZE 3/3", 32) }
            Events { OnElapsedTime(0) {
                Actions { sound("bell") flash("fbbf24", 0.5) } } } }
        dpad("runner")

        Entity { id("wall1") startPosition(300, 560) color("475569")
            solid(true) Type { rectangle(200, 200) } }
        Entity { id("wall2") startPosition(160, 800) color("475569")
            solid(true) Type { rectangle(140, 80) } }
        Entity { id("wall3") startPosition(520, 800) color("475569")
            solid(true) Type { rectangle(140, 80) } }
        Entity { id("guard1") startPosition(120, 540) color("ef4444")
            Type { diamond(90, 90) }
            Movement { speed(3.4) Path { waypoint(120, 540) waypoint(620, 540) } } }
        Entity { id("guard2") startPosition(620, 740) color("ef4444")
            Type { diamond(90, 90) }
            Movement { speed(3.4) Path { waypoint(620, 740) waypoint(120, 740) } } }
        Entity { id("guard3") startPosition(120, 940) color("ef4444")
            Type { diamond(90, 90) }
            Movement { speed(3.4) Path { waypoint(120, 940) waypoint(620, 940) } } }
        Entity { id("exit") startPosition(580, 1030) color("4ade80")
            Type { star(5, 60, 28) } }

        Entity {
            id("runner")
            startPosition(60, 440)
            color("22d3ee")
            Type { circle(40) }
            Movement { GridMove(100, 0.3){ direction(0, 0) } }
            Events {
                OnCollision("exit") {
                    Actions {
                        sound("win")
                        flash("4ade80", 0.4)
                        shake(5, 0.4)
                        burst(20, "fbbf24")
                        gameOver("-- ALL MAZES CLEARED --")
                    }
                }
                OnCollision("guard1") {
                    Actions {
                        sound("lose")
                        flash("ff0000", 0.25)
                        shake(6, 0.3)
                        burst(14, "ef4444")
                        gameOver("-- CAUGHT --")
                    }
                }
                OnCollision("guard2") {
                    Actions {
                        sound("lose")
                        flash("ff0000", 0.25)
                        shake(6, 0.3)
                        burst(14, "ef4444")
                        gameOver("-- CAUGHT --")
                    }
                }
                OnCollision("guard3") {
                    Actions {
                        sound("lose")
                        flash("ff0000", 0.25)
                        shake(6, 0.3)
                        burst(14, "ef4444")
                        gameOver("-- CAUGHT --")
                    }
                }
            }
        }
    }
}
