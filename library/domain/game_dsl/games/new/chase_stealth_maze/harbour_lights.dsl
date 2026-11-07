Window {
    size(8, 9)
    worldSize(32, 9)
    cameraFollow("skiff")
    background("0b1d33")
    music("bg_4")
    gravity(0)

    Bar("crates") {
        initial(0)
        OnValue(3) { Actions { sound("bell") broadcast("boomUp") } }
    }

    joystick("skiff", 70, 3.4)

    Entity {
        id("skiff")
        startPosition(220, 200)
        color("e7e5e4")
        Type { rectangle(60, 30) }
        Movement { speed(0) Control { movementControl(0) } }
        Events {
            OnSignal("spotted") { Actions { shake(4, 0.3) } }
            OnSignal("boomUp") { Actions { flash("4ade80", 0.3) burst(14, "4ade80") } }
            OnCollision("openSea") { Actions { sound("win") shake(5, 0.4) burst(30, "93c5fd") gameOver("-- OUT TO OPEN SEA --") } }
        }
    }

    Entity { id("hint") startPosition(40, 850) color("93c5fd") Type { text("Grab 3 crates, then slip past the boom", 22) } }

    # Piers stagger the channel: under one, over the next.
    Entity { id("pier1") startPosition(500, 400) color("78350f") solid(true) Type { rectangle(60, 500) } }
    Entity { id("pier2") startPosition(900, 0) color("78350f") solid(true) Type { rectangle(60, 520) } }
    Entity { id("pier3") startPosition(1400, 380) color("78350f") solid(true) Type { rectangle(60, 520) } }
    Entity { id("pier4") startPosition(1900, 0) color("78350f") solid(true) Type { rectangle(60, 520) } }
    Entity { id("pier5") startPosition(2750, 350) color("78350f") solid(true) Type { rectangle(60, 550) } }

    HostilePatrol("patrol1", "skiff", "-- BOARDED BY THE HARBOUR PATROL --", 2.4, "f87171") {
        waypoint(700, 60)
        waypoint(700, 790)
        dwell(0.8)
    }
    HostilePatrol("patrol2", "skiff", "-- BOARDED BY THE HARBOUR PATROL --", 2.6, "f87171") {
        waypoint(1650, 790)
        waypoint(1650, 60)
        dwell(0.8)
    }
    HostilePatrol("patrol3", "skiff", "-- BOARDED BY THE HARBOUR PATROL --", 2.8, "fb7185") {
        waypoint(2200, 60)
        waypoint(2200, 790)
        dwell(0.6)
    }

    # Searchlights sweep up and down. Get caught in one and the cutter casts off.
    Entity { id("beam1") startPosition(1080, 60) color("55fde047") Type { rectangle(120, 260) }
        Movement { speed(1.6) Path { waypoint(1080, 60) waypoint(1080, 600) dwell(0.6) } }
        Events { OnCollision("skiff") { Actions { sound("buzz") flash("skiff", "fde047", 0.3) broadcast("spotted") } } } }
    Entity { id("beam2") startPosition(2400, 600) color("55fde047") Type { rectangle(120, 260) }
        Movement { speed(1.8) Path { waypoint(2400, 600) waypoint(2400, 60) dwell(0.6) } }
        Events { OnCollision("skiff") { Actions { sound("buzz") flash("skiff", "fde047", 0.3) broadcast("spotted") } } } }

    Entity {
        id("cutter")
        startPosition(1250, 840)
        color("e11d48")
        Type { rectangle(90, 40) }
        hostile("skiff", "-- THE COASTGUARD CUT YOU OFF --", "lose")
        Movement { speed(2.5) chase("skiff") }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("spotted") { Actions { setEnable(true) burst(12, "e11d48") } }
        }
    }

    Entity { id("crate1") startPosition(1250, 120) color("d97706") Type { rectangle(36, 36) }
        Events { OnCollision("skiff") { Actions { sound("coin") updateBar("crates", 1) burst(10, "fbbf24") deleteEntity() } } } }
    Entity { id("crate2") startPosition(2050, 780) color("d97706") Type { rectangle(36, 36) }
        Events { OnCollision("skiff") { Actions { sound("coin") updateBar("crates", 1) burst(10, "fbbf24") deleteEntity() } } } }
    Entity { id("crate3") startPosition(2600, 120) color("d97706") Type { rectangle(36, 36) }
        Events { OnCollision("skiff") { Actions { sound("coin") updateBar("crates", 1) burst(10, "fbbf24") deleteEntity() } } } }

    Entity { id("boom") startPosition(3000, 0) color("a8a29e") Type { rectangle(40, 900) } door("boomUp", "clang") }
    Entity { id("openSea") startPosition(3080, 0) color("1e3a8a") Type { rectangle(120, 900) } }
}
