Window {
    size(8, 9)
    worldSize(36, 9)
    cameraFollow("hero")
    background("0c4a6e")
    music("bg_ambient")
    gravity(120)

    lives(3, "-- PUFFED OUT --", "lose")
    Bar("pearls") {
        initial(0)
        OnValue(4) { Actions { sound("sparkle") updateBar("lives", 1) } }
        OnValue(7) { Actions { sound("sparkle") updateBar("lives", 1) } }
    }

    goal("grotto", 3480, 250, 110, 400, "-- HOME TO THE GROTTO --")

    # the pufferfish swims on by itself; swipe up to puff (rise), down to deflate (sink)
    Entity { id("hero") startPosition(100, 420) color("fde68a") facing("left")
        Type { text("🐡", 44) }
        Movement { speed(1.6) AutoMove { direction(1, 0) } }
        Events {
            OnSwipe("up") { Actions { applyImpulse(0, 170) setScale(1.25) sound("bubble") burst(6, "e0f2fe") } }
            OnSwipe("down") { Actions { applyImpulse(0, -170) setScale(0.85) sound("blip") } }
            OnCollision { Actions { sound("bump") bounceBack(0) } }
            OnCollision("vent") { Actions { sound("bubble") applyImpulse(0, 220) burst(8, "a5f3fc") } }
            OnCollision("urchin") { Actions {
                sound("hurt") updateBar("lives", -1) flash("ff0000", 0.3) shake(6, 0.3)
                setVelocity(160, 240) } }
            OnCollision("spike") { Actions {
                sound("hurt") updateBar("lives", -1) flash("ff0000", 0.3) shake(6, 0.3)
                setVelocity(160, -240) } }
            OnSignal("inked") { Actions {
                sound("buzz") updateBar("lives", -1) flash("a855f7", 0.4) shake(5, 0.3)
                setScale(1) setVelocity(160, 0) } } } }

    Entity { id("urchin") template(true) startPosition(0, 0) color("7c3aed") Type { star(8, 32, 14) } }
    Entity { id("spike") template(true) startPosition(0, 0) color("475569") Type { triangle(70, "down") } }
    Entity { id("vent") template(true) startPosition(0, 0) color("3322d3ee") Type { rectangle(80, 260) }
        Events { OnElapsedTime(0.4) { repeatCount(-1) Actions { burst(2, "a5f3fc") } } } }
    Entity { id("pearl") template(true) startPosition(0, 0) color("f8fafc") Type { circle(10) }
        Events { OnCollision("hero") { Actions { sound("coin") updateBar("pearls", 1) burst(10, "f8fafc") deleteEntity() } } } }

    # the seabed lays out the reef once, at the start
    Entity { id("seabed") startPosition(0, 0) color("a16207") Type { rectangle(3600, 30) }
        Events { OnElapsedTime { Actions {
            spawn("urchin", 520, 30) spawn("urchin", 1150, 30) spawn("urchin", 1700, 30)
            spawn("urchin", 2300, 30) spawn("urchin", 2900, 30)
            spawn("spike", 800, 830) spawn("spike", 1450, 830) spawn("spike", 2050, 830)
            spawn("spike", 2650, 830) spawn("spike", 3150, 830)
            spawn("vent", 1000, 30) spawn("vent", 2500, 30)
            spawn("pearl", 650, 420) spawn("pearl", 1300, 620) spawn("pearl", 1600, 250)
            spawn("pearl", 2200, 520) spawn("pearl", 2700, 700) spawn("pearl", 3050, 300)
            spawn("pearl", 3300, 560) } } } }

    Entity { id("squid1") startPosition(1250, 300) color("c084fc") Type { text("🦑", 46) }
        Movement { speed(1.2) Path { waypoint(1250, 300) waypoint(1250, 700) } }
        Events { OnCollision("hero") { Actions { burst(10, "c084fc") broadcast("inked") } } } }
    Entity { id("squid2") startPosition(2150, 650) color("c084fc") Type { text("🦑", 46) }
        Movement { speed(1.4) Path { waypoint(2150, 650) waypoint(2150, 180) } }
        Events { OnCollision("hero") { Actions { burst(10, "c084fc") broadcast("inked") } } } }
    Entity { id("squid3") startPosition(3000, 350) color("c084fc") Type { text("🦑", 46) }
        Movement { speed(1.6) Path { waypoint(3000, 350) waypoint(3000, 800) dwell(0.4) } }
        Events { OnCollision("hero") { Actions { burst(10, "c084fc") broadcast("inked") } } } }
}
