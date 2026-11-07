Window {
    size(10, 8)
    background("e2e8f0")
    music("bg_nature")
    gravity(0)

    # Field 1: one hound. Lure it behind a boulder, then dash for the den.
    Level {
        Entity { id("title1") startPosition(20, 750) color("64748b") Type { text("SNOWFIELD 1/2", 26) } }
        Entity { id("fox") startPosition(40, 360) color("ea580c") Type { triangle(44, "right") }
            Movement { speed(4.4) Control { movementControl(0) } }
            Events { OnCollision("den") { Actions {
                sound("win2") flash("fde68a", 0.3) burst(16, "ea580c") nextLevel() } } } }
        Entity { id("den") startPosition(900, 320) color("78350f") Type { rectangle(80, 140) }
            Events { OnElapsedTime(1.5) { repeatCount(-1) Actions { burst(4, "fbbf24") } } } }
        Entity { id("hound") startPosition(500, 740) color("57534e") Type { circle(22) }
            hostile("fox", "-- THE HOUND CAUGHT YOU --", "lose")
            Movement { speed(2.9) chase("fox") }
            Events { OnCollision("fox") { Actions { shake(7, 0.4) burst(18, "ef4444") } } } }
        Entity { id("rock1") startPosition(260, 180) color("94a3b8") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rock2") startPosition(260, 520) color("94a3b8") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rock3") startPosition(560, 340) color("94a3b8") solid(true) Type { rectangle(120, 120) } }
        Entity { id("rock4") startPosition(760, 120) color("94a3b8") solid(true) Type { rectangle(90, 90) } }
        Entity { id("rock5") startPosition(760, 580) color("94a3b8") solid(true) Type { rectangle(90, 90) } }
    }

    # Field 2: two hounds and a hunter walking the middle of the field.
    Level {
        Entity { id("title2") startPosition(20, 750) color("64748b") Type { text("SNOWFIELD 2/2", 26) } }
        Entity { id("fox") startPosition(40, 360) color("ea580c") Type { triangle(44, "right") }
            Movement { speed(4.4) Control { movementControl(0) } }
            Events { OnCollision("den") { Actions {
                sound("win") shake(5, 0.4) burst(24, "fbbf24") gameOver("-- SAFE IN THE DEN --") } } } }
        Entity { id("den") startPosition(900, 320) color("78350f") Type { rectangle(80, 140) }
            Events { OnElapsedTime(1.5) { repeatCount(-1) Actions { burst(4, "fbbf24") } } } }
        Entity { id("hound1") startPosition(500, 740) color("57534e") Type { circle(22) }
            hostile("fox", "-- THE HOUNDS CAUGHT YOU --", "lose")
            Movement { speed(2.9) chase("fox") } }
        Entity { id("hound2") startPosition(560, 20) color("292524") Type { circle(22) }
            hostile("fox", "-- THE HOUNDS CAUGHT YOU --", "lose")
            Movement { speed(2.5) chase("fox") } }
        Entity { id("hunter") startPosition(470, 40) color("b91c1c") Type { rectangle(40, 64) }
            hostile("fox", "-- THE HUNTER SPOTTED YOU --", "lose")
            Movement { speed(2.2) Path { waypoint(470, 40) waypoint(470, 700) dwell(1.2) } }
            Events { OnCollision("fox") { Actions { flash("fox", "ff0000", 0.3) shake(8, 0.4) } } } }
        Entity { id("rock1") startPosition(250, 150) color("94a3b8") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rock2") startPosition(250, 550) color("94a3b8") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rock3") startPosition(640, 220) color("94a3b8") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rock4") startPosition(640, 480) color("94a3b8") solid(true) Type { rectangle(100, 100) } }
        Entity { id("rock5") startPosition(800, 640) color("94a3b8") solid(true) Type { rectangle(80, 80) } }
    }
}
