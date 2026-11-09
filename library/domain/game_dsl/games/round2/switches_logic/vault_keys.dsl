Window {
    size(8, 12)
    background("0b1020")
    music("bg_2")
    gravity(0)
    Bar("bumps") {
        initial(12)
        OnValue(0) { Actions { gameOver("-- TOO CLUMSY --") } }
    }

    dpad("hero")

    # divider wall between the two halves, door1 in the gap
    Entity { id("wall1") startPosition(360, 440) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall2") startPosition(360, 640) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall3") startPosition(360, 740) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall4") startPosition(360, 840) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall5") startPosition(360, 940) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall6") startPosition(360, 1040) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("door1") startPosition(360, 540) color("7c3aed") Type { rectangle(100, 100) } door("k1", "clang") }

    # upper right vault: door2 closes the way to the exit
    Entity { id("wall7") startPosition(460, 940) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall8") startPosition(660, 940) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("door2") startPosition(560, 940) color("db2777") Type { rectangle(100, 100) } door("k2", "clang") }

    Entity { id("key1") startPosition(185, 765) color("facc15") Type { star(5, 26, 12) } key("k1", "hero", "pickup") }
    Entity { id("key2") startPosition(585, 565) color("f472b6") Type { star(5, 26, 12) } key("k2", "hero", "pickup") }

    Entity {
        id("watcher")
        startPosition(60, 640)
        color("ef4444")
        Type { circle(34) }
        hostile("hero", "-- CAUGHT --", "lose")
        Movement { speed(2.4) Path { waypoint(60, 640) waypoint(260, 640) dwell(0.5) } }
    }
    Entity {
        id("crawler")
        startPosition(460, 740)
        color("f97316")
        Type { circle(34) }
        hostile("hero", "-- CAUGHT --", "lose")
        Movement { speed(3.2) Path { waypoint(460, 740) waypoint(660, 740) waypoint(660, 640) waypoint(460, 640) } }
    }

    Entity {
        id("exit")
        startPosition(560, 1040)
        color("4ade80")
        Type { rectangle(100, 100) }
    }

    Entity {
        id("hero")
        startPosition(60, 440)
        color("22d3ee")
        Type { circle(40) }
        var("keys", 0)
        Movement { GridMove(100, 0.3) { direction(0, 0) } }
        Events {
            OnSignal("k1") { Actions { changeVar("keys", 1) flash("facc15", 0.3) } }
            OnSignal("k2") { Actions { changeVar("keys", 1) flash("f472b6", 0.3) } }
            OnGridBlocked {
                Actions {
                    sound("bump")
                    updateBar("bumps", -1)
                    setDirection("hero", 0, 0)
                }
            }
            OnCollision("exit") {
                Actions {
                    whenVar("keys", ">=", 2) {
                        Actions { sound("win") burst(20, "4ade80") gameOver("-- VAULT CRACKED --") }
                    }
                }
            }
        }
    }
}
