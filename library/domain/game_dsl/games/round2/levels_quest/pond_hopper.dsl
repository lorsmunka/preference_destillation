# Pond Hopper: hop the frog across ponds step by step, dodge the dragonflies and turtles, reach the lily pad.
Window {
    size(8, 12)
    background("14532d")
    music("bg_nature")
    gravity(0)
    lives(3, "-- THE FROG IS TOAST --", "lose")

    Level {
        dpad("hero")
        Entity { id("hero") startPosition(60, 440) color("4ade80") Type { circle(34) }
            Movement { GridMove(100, 0.25) { direction(0, 0) } }
            Events {
                OnCollision("dragonfly1") { Actions { sound("hurt") updateBar("lives", -1) setPosition("hero", 60, 440, 0) } }
                OnCollision("pad") { Actions { sound("win2") burst(12, "f9a8d4") nextLevel() } }
            } }
        Entity { id("pad") startPosition(640, 1040) color("f9a8d4") Type { ring(50, 24) } }
        Entity { id("rock1") startPosition(260, 640) color("57534e") solid(true) Type { rectangle(100, 100) } }
        Entity { id("dragonfly1") startPosition(80, 840) color("22d3ee") Type { diamond(60, 40) }
            Movement { speed(3.5) Path { waypoint(80, 840) waypoint(700, 840) } } }
    }

    Level {
        dpad("hero")
        Entity { id("hero") startPosition(60, 440) color("4ade80") Type { circle(34) }
            Movement { GridMove(100, 0.25) { direction(0, 0) } }
            Events {
                OnCollision("dragonfly1") { Actions { sound("hurt") updateBar("lives", -1) setPosition("hero", 60, 440, 0) } }
                OnCollision("dragonfly2") { Actions { sound("hurt") updateBar("lives", -1) setPosition("hero", 60, 440, 0) } }
                OnCollision("pad") { Actions { sound("win2") burst(12, "f9a8d4") nextLevel() } }
            } }
        Entity { id("pad") startPosition(60, 1040) color("f9a8d4") Type { ring(50, 24) } }
        Entity { id("rock1") startPosition(260, 640) color("57534e") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rock2") startPosition(460, 940) color("57534e") solid(true) Type { rectangle(100, 100) } }
        Entity { id("dragonfly1") startPosition(80, 740) color("22d3ee") Type { diamond(60, 40) }
            Movement { speed(4) Path { waypoint(80, 740) waypoint(700, 740) } } }
        Entity { id("dragonfly2") startPosition(700, 560) color("a5f3fc") Type { diamond(60, 40) }
            Movement { speed(3) Path { waypoint(700, 560) waypoint(80, 560) dwell(0.7) } } }
    }

    Level {
        dpad("hero")
        goal("lily", 340, 1040, 100, 80, "-- HOME AT LAST --")
        Entity { id("hero") startPosition(60, 440) color("4ade80") Type { circle(34) }
            Movement { GridMove(100, 0.25) { direction(0, 0) } }
            Events {
                OnCollision("dragonfly1") { Actions { sound("hurt") updateBar("lives", -1) setPosition("hero", 60, 440, 0) } }
                OnCollision("turtle") { Actions { sound("thud") updateBar("lives", -1) setPosition("hero", 60, 440, 0) } }
            } }
        Entity { id("rock1") startPosition(160, 640) color("57534e") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rock2") startPosition(560, 640) color("57534e") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rock3") startPosition(360, 840) color("57534e") solid(true) Type { rectangle(100, 100) } }
        Entity { id("dragonfly1") startPosition(80, 540) color("22d3ee") Type { diamond(60, 40) }
            Movement { speed(4.5) Path { waypoint(80, 540) waypoint(700, 540) } } }
        Entity { id("turtle") startPosition(700, 940) color("166534") Type { rectangle(80, 50) }
            Movement { speed(2) Path { waypoint(700, 940) waypoint(80, 940) } } }
        HostilePatrol("heron", "hero", "-- THE HERON GOT YOU --", 3, "e2e8f0") { waypoint(700, 740) waypoint(80, 740) dwell(1) }
    }
}
