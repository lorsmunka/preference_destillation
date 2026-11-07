Window {
    size(8, 12)
    worldSize(16, 24)
    cameraFollow("wanderer")
    background("0c1a12")
    music("bg_ambient")
    gravity(0)

    lives(3, "-- THE GHOSTS TOOK YOUR LIGHT --", "lose")
    Bar("lanterns") {
        initial(0)
        OnValue(5) { Actions { sound("win") shake(5, 0.5) burst("wanderer", 30, "fbbf24") gameOver("-- THE GARDEN GLOWS AGAIN --") } }
    }

    joystick("wanderer", 80, 3.6)
    wallBox(0, 0, 1600, 2400, "14532d", 30)

    Entity {
        id("wanderer")
        startPosition(240, 220)
        color("fef9c3")
        Type { circle(18) }
        Movement { speed(0) Control { movementControl(0) } }
        Events {
            OnSignal("glow") { Actions { flash("fde68a", 0.3) } }
            OnSignal("gateOpen") { Actions { burst(12, "fbbf24") } }
        }
    }

    Entity { id("hint") startPosition(60, 430) color("bbf7d0")
        Type { text("Light 5 lanterns. Each one lures the ghosts to the wisp", 20) }
        Events { OnSignal("gateOpen") { Actions { setText("hint", "The north gate is open") } } } }

    # Hedge rows, each with one gap; the last gap is a locked gate.
    Entity { id("row1") startPosition(30, 500) color("166534") solid(true) Type { rectangle(1270, 40) } }
    Entity { id("row2") startPosition(300, 1000) color("166534") solid(true) Type { rectangle(1270, 40) } }
    Entity { id("row3a") startPosition(30, 1500) color("166534") solid(true) Type { rectangle(1070, 40) } }
    Entity { id("row3b") startPosition(1350, 1500) color("166534") solid(true) Type { rectangle(220, 40) } }
    Entity { id("row4") startPosition(500, 1950) color("166534") solid(true) Type { rectangle(1070, 40) } }
    Entity { id("hedgeV1") startPosition(700, 540) color("166534") solid(true) Type { rectangle(40, 300) } }
    Entity { id("hedgeV2") startPosition(900, 1040) color("166534") solid(true) Type { rectangle(40, 300) } }
    Entity { id("hedgeV3") startPosition(500, 1540) color("166534") solid(true) Type { rectangle(40, 260) } }
    Entity { id("hedgeV4") startPosition(1100, 1990) color("166534") solid(true) Type { rectangle(40, 260) } }

    Entity { id("gate") startPosition(30, 1950) color("78350f") Type { rectangle(470, 40) } door("gateOpen", "clang") }
    Entity { id("oldKey") startPosition(1450, 1800) color("d6d3d1") Type { cross(36, 36, 10) } key("gateOpen", "wanderer", "pickup") }

    # A will-o'-wisp drifts through the middle of the garden. Lit lanterns
    # make the ghosts follow it for a few seconds instead of you.
    Entity { id("wisp") startPosition(100, 1400) color("a5f3fc") Type { circle(12) }
        Movement { speed(1.2) Path { waypoint(100, 1400) waypoint(1300, 1400) dwell(1.5) } }
        Events { OnElapsedTime(0.4) { repeatCount(-1) Actions { burst(2, "a5f3fc") } } } }

    Entity {
        id("ghostA")
        startPosition(1400, 800)
        color("cbd5e1")
        Type { circle(24) }
        Movement { speed(1.7) chase("wanderer") }
        Events {
            OnSignal("glow") { Actions { retarget("wisp") setOpacity(0.4)
                delay(4) { Actions { retarget("wanderer") setOpacity(1) } } } }
            OnCollision("wanderer") { Actions { sound("hurt") updateBar("lives", -1) flash("wanderer", "ff0000", 0.3)
                shake(6, 0.3) burst(14, "e2e8f0") setPosition("ghostA", 1400, 800, 0) } }
        }
    }
    Entity {
        id("ghostB")
        startPosition(800, 1800)
        color("cbd5e1")
        Type { circle(24) }
        Movement { speed(1.9) chase("wanderer") }
        Events {
            OnSignal("glow") { Actions { retarget("wisp") setOpacity(0.4)
                delay(4) { Actions { retarget("wanderer") setOpacity(1) } } } }
            OnCollision("wanderer") { Actions { sound("hurt") updateBar("lives", -1) flash("wanderer", "ff0000", 0.3)
                shake(6, 0.3) burst(14, "e2e8f0") setPosition("ghostB", 800, 1800, 0) } }
        }
    }
    Entity {
        id("ghostC")
        startPosition(300, 2250)
        color("e9d5ff")
        Type { circle(26) }
        Movement { speed(2.1) chase("wanderer") }
        Events {
            OnSignal("glow") { Actions { retarget("wisp") setOpacity(0.4)
                delay(4) { Actions { retarget("wanderer") setOpacity(1) } } } }
            OnCollision("wanderer") { Actions { sound("hurt") updateBar("lives", -1) flash("wanderer", "ff0000", 0.3)
                shake(6, 0.3) burst(14, "e9d5ff") setPosition("ghostC", 300, 2250, 0) } }
        }
    }

    Entity { id("lantern1") startPosition(1400, 300) color("fbbf24") Type { polygon(8, 18) }
        Events { OnCollision("wanderer") { Actions { sound("bell") updateBar("lanterns", 1) burst(16, "fbbf24") broadcast("glow") deleteEntity() } } } }
    Entity { id("lantern2") startPosition(200, 800) color("fbbf24") Type { polygon(8, 18) }
        Events { OnCollision("wanderer") { Actions { sound("bell") updateBar("lanterns", 1) burst(16, "fbbf24") broadcast("glow") deleteEntity() } } } }
    Entity { id("lantern3") startPosition(1450, 1250) color("fbbf24") Type { polygon(8, 18) }
        Events { OnCollision("wanderer") { Actions { sound("bell") updateBar("lanterns", 1) burst(16, "fbbf24") broadcast("glow") deleteEntity() } } } }
    Entity { id("lantern4") startPosition(250, 1750) color("fbbf24") Type { polygon(8, 18) }
        Events { OnCollision("wanderer") { Actions { sound("bell") updateBar("lanterns", 1) burst(16, "fbbf24") broadcast("glow") deleteEntity() } } } }
    Entity { id("lantern5") startPosition(1350, 2250) color("fbbf24") Type { polygon(8, 18) }
        Events { OnCollision("wanderer") { Actions { sound("bell") updateBar("lanterns", 1) burst(16, "fbbf24") broadcast("glow") deleteEntity() } } } }
}
