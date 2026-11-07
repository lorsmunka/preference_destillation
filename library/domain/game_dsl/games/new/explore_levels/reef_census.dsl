# Reef Census: swim up to six species to photograph them. The octopus only shows itself now
# and then, the crab hides when you come close (wait nearby and it peeks out again), and the
# shark leaves its lair to chase anyone who swims in.
Window {
    size(8, 12)
    worldSize(20, 16)
    cameraFollow("fish")
    background("0e7490")
    music("bg_ambient")
    gravity(0)

    lives(3, "-- STUNG ONE TOO MANY TIMES --", "lose")
    winAt("species", 6, "-- REEF CENSUS COMPLETE --", "win")

    Entity {
        id("fish")
        startPosition(1000, 760)
        color("ffffff")
        Type { text("🐠", 50) }
        facing("left")
        Movement { speed(0) Control { movementControl(0) } }
    }
    joystick("fish", 85, 3.2)

    Entity {
        id("caption")
        startPosition(1000, 860)
        color("ecfeff")
        Type { text("Photograph 6 species", 26) }
        Movement { speed(9) chase("fish", 1.1) }
    }

    Entity {
        id("turtle")
        color("ffffff")
        Type { text("🐢", 54) }
        facing("left")
        var("snap", 0)
        Movement { speed(1) Path { waypoint(300, 300) waypoint(900, 500) waypoint(600, 900) } }
        Events { OnCollision("fish") { Actions { whenVar("snap", "==", 0) { Actions {
            changeVar("snap", 1) sound("click") flash("ffffff", 0.3) updateBar("species", 1)
            setText("caption", "Turtle photographed!") } } } } }
    }

    Entity {
        id("ray")
        color("94a3b8")
        Type { diamond(90, 50) }
        var("snap", 0)
        Movement { speed(3.6) Path { waypoint(1100, 200) waypoint(1900, 320) waypoint(1500, 600) } }
        Events { OnCollision("fish") { Actions { whenVar("snap", "==", 0) { Actions {
            changeVar("snap", 1) sound("click") flash("ffffff", 0.3) updateBar("species", 1)
            setText("caption", "Ray photographed!") } } } } }
    }

    Entity {
        id("seahorse")
        startPosition(1800, 1420)
        color("fbbf24")
        Type { text("S", 46) }
        var("snap", 0)
        Events { OnCollision("fish") { Actions { whenVar("snap", "==", 0) { Actions {
            changeVar("snap", 1) sound("click") flash("ffffff", 0.3) updateBar("species", 1)
            setText("caption", "Seahorse photographed!") } } } } }
    }

    Entity {
        id("octopus")
        startPosition(380, 1280)
        color("ffffff")
        Type { text("🐙", 54) }
        var("snap", 0)
        Events { OnCollision("fish") { Actions { whenVar("snap", "==", 0) { Actions {
            changeVar("snap", 1) sound("click") flash("ffffff", 0.3) updateBar("species", 1)
            setText("caption", "Octopus photographed!") } } } } }
    }
    Entity {
        id("inkCloud")
        startPosition(330, 1230)
        color("801e1b4b")
        Type { circle(80) }
        Events {
            OnElapsedTime { Actions { setEnable("octopus", false) } }
            OnElapsedTime(4) { repeatCount(-1) Actions {
                setEnable("octopus", true) burst(8, "312e81")
                delay(1.6) { Actions { setEnable("octopus", false) } } } }
        }
    }

    Entity {
        id("crab")
        startPosition(1500, 60)
        color("ffffff")
        Type { text("🦀", 48) }
        var("snap", 0)
        Events {
            OnSignal("spooked") { Actions { burst(10, "fcd34d") setEnable(false) delay(3) { Actions { setEnable(true) } } } }
            OnCollision("fish") { Actions { whenVar("snap", "==", 0) { Actions {
                changeVar("snap", 1) sound("click") flash("ffffff", 0.3) updateBar("species", 1)
                setText("caption", "Crab photographed!") } } } }
        }
    }
    Entity {
        id("crabSand")
        startPosition(1370, 0)
        color("33fde68a")
        Type { rectangle(320, 250) }
        Events { OnCollision("fish") { Actions { sound("pop") broadcast("spooked") setText("caption", "The crab hid - wait here") } } }
    }

    Entity {
        id("buoy")
        color("00000000")
        Type { circle(4) }
        Movement { speed(1.5) Path { waypoint(250, 1450) waypoint(750, 1450) waypoint(750, 1050) waypoint(250, 1050) } }
    }
    Entity {
        id("lair")
        startPosition(150, 950)
        color("221e293b")
        Type { rectangle(700, 600) }
        Events { OnCollision("fish") { Actions { broadcast("intruder") } } }
    }
    Entity {
        id("shark")
        startPosition(250, 1450)
        color("64748b")
        Type { triangle(70, "left") }
        var("snap", 0)
        Movement { speed(2) chase("buoy") }
        Events {
            OnSignal("intruder") { Actions { sound("buzz") retarget("fish") updateSpeed(2.4)
                delay(4) { Actions { retarget("buoy") updateSpeed(2) } } } }
            OnCollision("fish") { Actions {
                sound("hurt") shake(8, 0.4) flash("fish", "ef4444", 0.4) updateBar("lives", -1) retarget("buoy")
                whenVar("snap", "==", 0) { Actions { changeVar("snap", 1) updateBar("species", 1) setText("caption", "Shark photographed... ouch!") } } } }
        }
    }

    Entity { id("jelly1") color("f0abfc") Type { circle(26) }
        Movement { speed(1.2) Path { waypoint(1500, 1100) waypoint(1500, 1550) } }
        Events { OnCollision("fish") { Actions { sound("hurt") flash("fish", "f0abfc", 0.4) updateBar("lives", -1) } } } }
    Entity { id("jelly2") color("f0abfc") Type { circle(26) }
        Movement { speed(1.5) Path { waypoint(1680, 1550) waypoint(1680, 1100) } }
        Events { OnCollision("fish") { Actions { sound("hurt") flash("fish", "f0abfc", 0.4) updateBar("lives", -1) } } } }
    Entity { id("jelly3") color("f0abfc") Type { circle(26) }
        Movement { speed(1.1) Path { waypoint(1900, 1150) waypoint(1950, 1550) dwell(0.6) } }
        Events { OnCollision("fish") { Actions { sound("hurt") flash("fish", "f0abfc", 0.4) updateBar("lives", -1) } } } }
}
