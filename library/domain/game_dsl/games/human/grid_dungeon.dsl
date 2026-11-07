Window {
    size(8, 12)
    background("0b1020")
    music("bg_2")
    gravity(0)

    Bar("treasure") { initial(0) }

    dpad("hero")

    Entity { id("patrolMark1") startPosition(60, 772) color("6b2222")
        Type { rectangle(640, 4) }
        Events { OnElapsedTime(2) { repeatCount(-1)
            Actions { flash("ef4444", 0.4) } } } }
    Entity { id("patrolMark2") startPosition(60, 972) color("6b2222")
        Type { rectangle(640, 4) }
        Events { OnElapsedTime(2) { repeatCount(-1)
            Actions { flash("ef4444", 0.4) } } } }

    Entity { id("wall1") startPosition(60, 640) color("1f2937")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall2") startPosition(160, 640) color("1f2937")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall3") startPosition(260, 640) color("1f2937")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall4") startPosition(360, 640) color("1f2937")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall5") startPosition(560, 640) color("1f2937")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall6") startPosition(660, 640) color("1f2937")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("door1") startPosition(460, 640) color("7c3aed")
        Type { rectangle(100, 100) } door("unlock1")
        Events { OnSignal("unlock1") {
            Actions { sound("clang") burst(14, "7c3aed") flash("hero", "c4b5fd", 0.3) } } } }

    Entity { id("wall7") startPosition(60, 840) color("1f2937")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall8") startPosition(160, 840) color("1f2937")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall9") startPosition(360, 840) color("1f2937")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall10") startPosition(460, 840) color("1f2937")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall11") startPosition(560, 840) color("1f2937")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall12") startPosition(660, 840) color("1f2937")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("door2") startPosition(260, 840) color("7c3aed")
        Type { rectangle(100, 100) } door("unlock2")
        Events { OnSignal("unlock2") {
            Actions { sound("clang") burst(14, "7c3aed") flash("hero", "c4b5fd", 0.3) } } } }

    Entity { id("key1") startPosition(185, 565) color("facc15")
        Type { rectangle(50, 50) } key("unlock1", "hero") }
    Entity { id("key2") startPosition(385, 765) color("facc15")
        Type { rectangle(50, 50) } key("unlock2", "hero") }

    Entity { id("guard1") startPosition(560, 740) color("ef4444")
        Type { circle(34) } hostile("hero", "-- CAUGHT --", "lose")
        Movement { speed(2.55) Path { waypoint(60, 740) waypoint(560, 740)
            dwell(1) } } }
    Entity { id("guard2") startPosition(160, 940) color("ef4444")
        Type { circle(34) } hostile("hero", "-- CAUGHT --", "lose")
        Movement { speed(2.55) Path { waypoint(60, 940) waypoint(560, 940)
            dwell(1) } } }

    Entity { id("treasure") startPosition(280, 1060) color("fbbf24")
        Type { diamond(60, 60) } }

    Entity { id("exit") startPosition(360, 1040) color("4ade80")
        Type { rectangle(100, 100) } }

    Entity {
        id("hero")
        startPosition(60, 440)
        color("22d3ee")
        Type { circle(40) }
        Movement { GridMove(100, 0.3) { direction(0, 0) } }
        Events {
            OnCollision("exit") {
                Actions {
                    sound("win")
                    flash("4ade80", 0.4)
                    burst(20, "4ade80")
                    shake(5, 0.4)
                    gameOver("-- ESCAPED THE DUNGEON --")
                }
            }
            OnCollision("treasure") {
                Actions {
                    sound("coin")
                    updateBar("treasure", 1)
                    flash("fbbf24", 0.3)
                    burst(12, "fbbf24")
                    deleteEntity("treasure")
                }
            }
            OnSignal("unlock1") {
                Actions {
                    sound("pickup")
                    flash("facc15", 0.25)
                    burst(10, "facc15")
                }
            }
            OnSignal("unlock2") {
                Actions {
                    sound("pickup")
                    flash("facc15", 0.25)
                    burst(10, "facc15")
                }
            }
            OnCollision("guard1") {
                Actions {
                    flash("ff0000", 0.3)
                    shake(7, 0.4)
                    burst(16, "ef4444")
                }
            }
            OnCollision("guard2") {
                Actions {
                    flash("ff0000", 0.3)
                    shake(7, 0.4)
                    burst(16, "ef4444")
                }
            }
        }
    }
}
