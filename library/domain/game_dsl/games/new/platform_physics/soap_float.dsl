# A soap bubble rises on its own (negative gravity) through a kitchen.
Window {
    size(6, 10)
    background("e0f2fe")
    music("bg_1")
    gravity(300)
    lives(3, "-- POP! --", "glass")

    goal("skylight", 200, 930, 200, 70, "-- FLOATED OUT THE WINDOW --")

    Entity { id("hero") startPosition(278, 70) color("a5b4fc") gravity(-90)
        Type { ring(22, 18) }
        Movement { speed(0) AutoMove { direction(0, 1) } }
        Events {
            OnCollision { Actions { sound("bubble") bounceBack(0) } }
            OnSignal("pop") { Actions {
                sound("glass") burst(18, "a5b4fc") shake(5, 0.3) updateBar("lives", -1)
                setPosition("hero", 278, 70, 0) setVelocity(0, 0) } } } }

    Entity { id("knife1") startPosition(150, 300) color("94a3b8") Type { rectangle(300, 12) }
        Events { OnCollision("hero") { Actions { flash("ef4444", 0.3) broadcast("pop") } } } }
    Entity { id("knife2") startPosition(0, 540) color("94a3b8") Type { rectangle(250, 12) }
        Events { OnCollision("hero") { Actions { flash("ef4444", 0.3) broadcast("pop") } } } }
    Entity { id("knife3") startPosition(350, 540) color("94a3b8") Type { rectangle(250, 12) }
        Events { OnCollision("hero") { Actions { flash("ef4444", 0.3) broadcast("pop") } } } }
    Entity { id("fork") startPosition(0, 760) color("64748b") Type { rectangle(140, 12) }
        Movement { speed(1.6) Path { waypoint(0, 760) waypoint(460, 760) } }
        Events { OnCollision("hero") { Actions { flash("ef4444", 0.3) broadcast("pop") } } } }

    Entity { id("faucet") startPosition(470, 880) color("cbd5e1") Type { rectangle(70, 30) }
        Events { OnElapsedTime(1.4) { repeatCount(-1) Actions { spawn("drip", 495, 860) } } } }
    Entity { id("drip") template(true) startPosition(0, 0) color("38bdf8") Type { circle(8) }
        Movement { speed(0.5) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("hero") { Actions { sound("pop") broadcast("pop") deleteEntity() } }
            OnCollision { Actions { burst(4, "38bdf8") deleteEntity() } } } }

    Entity { id("blowL") startPosition(0, 0) color("7dd3fc") Type { rectangle(190, 56) }
        Events { OnTapped("blowL") { Actions { sound("blip") applyImpulse("hero", -110, 0) burst(5, "ffffff") } } } }
    Entity { id("hold") startPosition(205, 0) color("c4b5fd") Type { rectangle(190, 56) }
        Events { OnTapped("hold") { Actions { sound("click") setVelocity("hero", 0, 0) flash("hero", "ffffff", 0.2) } } } }
    Entity { id("blowR") startPosition(410, 0) color("7dd3fc") Type { rectangle(190, 56) }
        Events { OnTapped("blowR") { Actions { sound("blip") applyImpulse("hero", 110, 0) burst(5, "ffffff") } } } }
    Entity { id("lblL") startPosition(50, 14) color("0c4a6e") Type { text("< BLOW", 24) } }
    Entity { id("lblH") startPosition(262, 14) color("3b0764") Type { text("HOLD", 24) } }
    Entity { id("lblR") startPosition(455, 14) color("0c4a6e") Type { text("BLOW >", 24) } }
}
