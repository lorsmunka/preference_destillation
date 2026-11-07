Window {
    size(8, 10)
    worldSize(8, 32)
    cameraFollow("diver")
    background("082f49")
    music("bg_6")
    gravity(0)

    lives(3, "-- THE SUB IS CRUSHED --", "lose")
    Bar("oxygen") {
        initial(30)
        OnValue(0) { Actions { sound("death") gameOver("-- OUT OF AIR --") } }
    }

    joystick("diver", 75, 3)

    Entity {
        id("diver")
        startPosition(380, 3050)
        color("fbbf24")
        Type { rectangle(52, 30) }
        Movement { speed(0) Control { movementControl(0) } }
        Events {
            OnCollision("wreck") { Actions { sound("win") shake(6, 0.5) burst(40, "fde68a") gameOver("-- YOU FOUND THE WRECK --") } }
        }
    }

    Entity { id("hint") startPosition(40, 3150) color("bae6fd") Type { text("Dive to the wreck. Mind your air.", 22) } }
    Entity { id("airGauge") startPosition(780, 3180) color("00000000") Type { circle(1) }
        Events { OnElapsedTime(1) { repeatCount(-1) Actions { updateBar("oxygen", -1) } } } }

    # Rock shelves zig-zag down the trench.
    Entity { id("shelf1") startPosition(0, 2600) color("334155") solid(true) Type { rectangle(560, 40) } }
    Entity { id("shelf2") startPosition(240, 2100) color("334155") solid(true) Type { rectangle(560, 40) } }
    Entity { id("shelf3") startPosition(0, 1600) color("334155") solid(true) Type { rectangle(560, 40) } }
    Entity { id("shelf4") startPosition(240, 1100) color("334155") solid(true) Type { rectangle(560, 40) } }
    Entity { id("shelf5") startPosition(0, 600) color("334155") solid(true) Type { rectangle(560, 40) } }

    Entity { id("wreck") startPosition(220, 40) color("78350f") Type { rectangle(300, 110) }
        Events { OnElapsedTime(2) { repeatCount(-1) Actions { burst(5, "fde68a") } } } }

    # The eel stalks you from a distance and spits homing jellyfish.
    Entity {
        id("eel")
        startPosition(40, 3100)
        color("65a30d")
        Type { rectangle(70, 20) }
        Movement { speed(2) chase("diver", 1.8) }
        Events { OnElapsedTime(5) { repeatCount(-1) Actions { spawnHere("jelly") sound("bubble") flash("bef264", 0.3) } } }
    }
    Entity {
        id("jelly")
        template(true)
        color("cce879f9")
        Type { circle(14) }
        Movement { speed(1.4) chase("diver") }
        Events {
            OnCollision("diver") { Actions { sound("hurt") updateBar("lives", -1) flash("diver", "ff0000", 0.3) burst(10, "e879f9") deleteEntity() } }
            OnElapsedTime(4) { Actions { burst(6, "f5d0fe") deleteEntity() } }
        }
    }

    # Drifting mines on their moorings.
    Entity { id("mine1") startPosition(40, 2350) color("dc2626") Type { star(10, 22, 14) }
        Movement { speed(1.8) Path { waypoint(40, 2350) waypoint(700, 2350) dwell(0.6) } }
        Events { OnCollision("diver") { Actions { sound("thud") updateBar("lives", -1) flash("diver", "ff0000", 0.4) shake(8, 0.5) burst(18, "f97316") } } } }
    Entity { id("mine2") startPosition(380, 1660) color("dc2626") Type { star(10, 22, 14) }
        Movement { speed(1.6) Path { waypoint(380, 1660) waypoint(380, 2040) dwell(0.5) } }
        Events { OnCollision("diver") { Actions { sound("thud") updateBar("lives", -1) flash("diver", "ff0000", 0.4) shake(8, 0.5) burst(18, "f97316") } } } }
    Entity { id("mine3") startPosition(40, 1350) color("dc2626") Type { star(10, 22, 14) }
        Movement { speed(2.2) Path { waypoint(40, 1350) waypoint(700, 1350) dwell(0.4) } }
        Events { OnCollision("diver") { Actions { sound("thud") updateBar("lives", -1) flash("diver", "ff0000", 0.4) shake(8, 0.5) burst(18, "f97316") } } } }
    Entity { id("mine4") startPosition(60, 1000) color("dc2626") Type { star(10, 22, 14) }
        Movement { speed(2) Path { waypoint(60, 1000) waypoint(700, 700) dwell(0.5) } }
        Events { OnCollision("diver") { Actions { sound("thud") updateBar("lives", -1) flash("diver", "ff0000", 0.4) shake(8, 0.5) burst(18, "f97316") } } } }
    Entity { id("mine5") startPosition(150, 300) color("b91c1c") Type { star(10, 22, 14) }
        Movement { speed(1.6) Path { waypoint(150, 300) waypoint(600, 300) waypoint(600, 450) waypoint(150, 450) dwell(0.3) } }
        Events { OnCollision("diver") { Actions { sound("thud") updateBar("lives", -1) flash("diver", "ff0000", 0.4) shake(8, 0.5) burst(18, "f97316") } } } }

    # Air pockets top up the tank.
    Entity { id("air1") startPosition(700, 2200) color("bae6fd") Type { ring(20) }
        Events { OnCollision("diver") { Actions { sound("bubble") updateBar("oxygen", 10) burst(12, "e0f2fe") deleteEntity() } } } }
    Entity { id("air2") startPosition(60, 1700) color("bae6fd") Type { ring(20) }
        Events { OnCollision("diver") { Actions { sound("bubble") updateBar("oxygen", 10) burst(12, "e0f2fe") deleteEntity() } } } }
    Entity { id("air3") startPosition(700, 1200) color("bae6fd") Type { ring(20) }
        Events { OnCollision("diver") { Actions { sound("bubble") updateBar("oxygen", 10) burst(12, "e0f2fe") deleteEntity() } } } }
    Entity { id("air4") startPosition(60, 700) color("bae6fd") Type { ring(20) }
        Events { OnCollision("diver") { Actions { sound("bubble") updateBar("oxygen", 10) burst(12, "e0f2fe") deleteEntity() } } } }
}
