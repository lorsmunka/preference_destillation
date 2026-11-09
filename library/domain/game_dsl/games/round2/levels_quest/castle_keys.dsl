# Castle Keys: step through four castle floors, take each key, open the door, reach the throne.
Window {
    size(8, 12)
    background("1c1917")
    music("bg_4")
    gravity(0)
    lives(3, "-- THE GUARDS GOT YOU --", "lose")

    Level {
        dpad("knight")
        wallBox(0, 400, 800, 800, "57534e", 10)
        Entity { id("knight") startPosition(60, 440) color("38bdf8") Type { circle(36) }
            Movement { GridMove(100, 0.3) { direction(0, 0) } }
            Events { OnCollision("stairs") { Actions { sound("win2") burst(12, "fbbf24") nextLevel() } } } }
        Entity { id("key1") startPosition(660, 560) color("facc15") Type { rectangle(40, 40) } key("open1", "knight", "pickup") }
        Entity { id("door1") startPosition(440, 880) color("7c3aed") Type { rectangle(360, 30) } door("open1", "clang") }
        Entity { id("barrier1") startPosition(10, 880) color("57534e") solid(true) Type { rectangle(430, 30) } }
        Entity { id("stairs") startPosition(640, 1040) color("fbbf24") Type { star(5, 50, 22) } }
        HostilePatrol("guard1", "knight", "-- CAUGHT --", 3, "ef4444") { waypoint(120, 700) waypoint(700, 700) dwell(1) }
    }

    Level {
        dpad("knight")
        wallBox(0, 400, 800, 800, "57534e", 10)
        Entity { id("knight") startPosition(60, 440) color("38bdf8") Type { circle(36) }
            Movement { GridMove(100, 0.3) { direction(0, 0) } }
            Events { OnCollision("stairs") { Actions { sound("win2") burst(12, "fbbf24") nextLevel() } } } }
        Entity { id("key1") startPosition(100, 1040) color("facc15") Type { rectangle(40, 40) } key("open2", "knight", "pickup") }
        Entity { id("door1") startPosition(10, 760) color("7c3aed") Type { rectangle(300, 30) } door("open2", "clang") }
        Entity { id("barrier1") startPosition(310, 760) color("57534e") solid(true) Type { rectangle(480, 30) } }
        Entity { id("stairs") startPosition(640, 1040) color("fbbf24") Type { star(5, 50, 22) } }
        HostilePatrol("guard1", "knight", "-- CAUGHT --", 3.5, "ef4444") { waypoint(120, 600) waypoint(700, 600) }
        HostilePatrol("guard2", "knight", "-- CAUGHT --", 3, "f97316") { waypoint(700, 900) waypoint(120, 900) dwell(0.5) }
    }

    Level {
        dpad("knight")
        wallBox(0, 400, 800, 800, "57534e", 10)
        Entity { id("knight") startPosition(60, 440) color("38bdf8") Type { circle(36) }
            Movement { GridMove(100, 0.3) { direction(0, 0) } }
            Events { OnCollision("stairs") { Actions { sound("win2") burst(12, "fbbf24") nextLevel() } } } }
        Entity { id("key1") startPosition(700, 1040) color("facc15") Type { rectangle(40, 40) } key("open3", "knight", "pickup") }
        Entity { id("door1") startPosition(10, 700) color("7c3aed") Type { rectangle(500, 30) } door("open3", "clang") }
        Entity { id("barrier1") startPosition(510, 700) color("57534e") solid(true) Type { rectangle(280, 30) } }
        Entity { id("pillar") startPosition(300, 480) color("57534e") solid(true) Type { rectangle(100, 100) } }
        Entity { id("stairs") startPosition(60, 1040) color("fbbf24") Type { star(5, 50, 22) } }
        HostilePatrol("guard1", "knight", "-- CAUGHT --", 4, "ef4444") { waypoint(120, 600) waypoint(700, 600) }
        HostilePatrol("guard2", "knight", "-- CAUGHT --", 3, "f97316") { waypoint(700, 820) waypoint(120, 820) waypoint(120, 1000) waypoint(700, 1000) }
    }

    Level {
        dpad("knight")
        wallBox(0, 400, 800, 800, "57534e", 10)
        Entity { id("knight") startPosition(60, 440) color("38bdf8") Type { circle(36) }
            Movement { GridMove(100, 0.3) { direction(0, 0) } }
            Events { OnCollision("throne") { Actions { sound("win") shake(6, 0.4) gameOver("-- LONG LIVE THE KNIGHT --") } } } }
        Entity { id("key1") startPosition(380, 540) color("facc15") Type { rectangle(40, 40) } key("open4", "knight", "pickup") }
        Entity { id("door1") startPosition(10, 860) color("7c3aed") Type { rectangle(560, 30) } door("open4", "clang") }
        Entity { id("barrier1") startPosition(570, 860) color("57534e") solid(true) Type { rectangle(220, 30) } }
        Entity { id("throne") startPosition(380, 1060) color("f43f5e") Type { diamond(70, 80) } }
        HostilePatrol("guard1", "knight", "-- CAUGHT --", 4, "ef4444") { waypoint(120, 640) waypoint(700, 640) }
        HostilePatrol("guard2", "knight", "-- CAUGHT --", 4, "f97316") { waypoint(700, 760) waypoint(120, 760) }
        HostilePatrol("captain", "knight", "-- THE CAPTAIN WINS --", 3, "dc2626") { waypoint(120, 1000) waypoint(700, 1000) dwell(1) }
    }
}
