Window {
    size(9, 12)
    background("2b1d0e")
    music("bg_5")
    gravity(0)

    Bar("home") {
        initial(0)
        OnValue(3) { Actions { sound("win") shake(4, 0.4) burst("worker", 24, "fde68a") gameOver("-- THE BROOD IS SAFE --") } }
    }

    Entity { id("hint") startPosition(380, 240) color("d6b98c") Type { text("Lead all three larvae up to the queen", 20) } }

    # The worker ant, and the brood trailing behind it in a line.
    Entity {
        id("worker")
        startPosition(150, 60)
        color("fb923c")
        Type { polygon(6, 18) }
        Movement { speed(3.2) Control { movementControl(0) } }
    }

    Entity { id("larva1") startPosition(100, 60) color("fef3c7") Type { circle(13) }
        var("safe", 0)
        Movement { speed(3.6) chase("worker", 0.55) }
        Events { OnCollision("chamber") { Actions { whenVar("safe", "==", 0) { Actions {
            changeVar("safe", 1) updateBar("home", 1) sound("pop") burst(10, "fde68a") } } } } } }
    Entity { id("larva2") startPosition(60, 60) color("fef3c7") Type { circle(13) }
        var("safe", 0)
        Movement { speed(3.6) chase("larva1", 0.5) }
        Events { OnCollision("chamber") { Actions { whenVar("safe", "==", 0) { Actions {
            changeVar("safe", 1) updateBar("home", 1) sound("pop") burst(10, "fde68a") } } } } } }
    Entity { id("larva3") startPosition(20, 60) color("fef3c7") Type { circle(13) }
        var("safe", 0)
        Movement { speed(3.6) chase("larva2", 0.5) }
        Events { OnCollision("chamber") { Actions { whenVar("safe", "==", 0) { Actions {
            changeVar("safe", 1) updateBar("home", 1) sound("pop") burst(10, "fde68a") } } } } } }

    # Soil layers, each open at one end, and a pebble in each tunnel.
    Entity { id("soilA") startPosition(0, 300) color("5b3a1a") solid(true) Type { rectangle(660, 60) } }
    Entity { id("soilB") startPosition(240, 620) color("5b3a1a") solid(true) Type { rectangle(660, 60) } }
    Entity { id("soilC") startPosition(0, 920) color("5b3a1a") solid(true) Type { rectangle(660, 60) } }
    Entity { id("pebble1") startPosition(380, 400) color("78716c") solid(true) Type { rectangle(120, 60) } }
    Entity { id("pebble2") startPosition(500, 720) color("78716c") solid(true) Type { rectangle(100, 60) } }

    Entity { id("chamber") startPosition(350, 1060) color("a16207") Type { rectangle(200, 110) }
        Events { OnElapsedTime(2) { repeatCount(-1) Actions { burst(4, "fde68a") } } } }

    # Spiders sweep the two middle tunnels and pause at each end.
    Entity {
        id("spider1")
        startPosition(20, 520)
        color("a21caf")
        Type { star(8, 22, 9) }
        hostile("worker", "-- A SPIDER GOT THE WORKER --", "lose")
        Movement { speed(2.0) Path { waypoint(20, 520) waypoint(840, 520) dwell(1.0) } }
        Events {
            OnCollision("larva1") { Actions { sound("lose") burst(16, "fef3c7") gameOver("-- A SPIDER SNATCHED A LARVA --") } }
            OnCollision("larva2") { Actions { sound("lose") burst(16, "fef3c7") gameOver("-- A SPIDER SNATCHED A LARVA --") } }
            OnCollision("larva3") { Actions { sound("lose") burst(16, "fef3c7") gameOver("-- A SPIDER SNATCHED A LARVA --") } }
        }
    }
    Entity {
        id("spider2")
        startPosition(840, 820)
        color("c026d3")
        Type { star(8, 22, 9) }
        hostile("worker", "-- A SPIDER GOT THE WORKER --", "lose")
        Movement { speed(2.3) Path { waypoint(840, 820) waypoint(20, 820) dwell(0.8) } }
        Events {
            OnCollision("larva1") { Actions { sound("lose") burst(16, "fef3c7") gameOver("-- A SPIDER SNATCHED A LARVA --") } }
            OnCollision("larva2") { Actions { sound("lose") burst(16, "fef3c7") gameOver("-- A SPIDER SNATCHED A LARVA --") } }
            OnCollision("larva3") { Actions { sound("lose") burst(16, "fef3c7") gameOver("-- A SPIDER SNATCHED A LARVA --") } }
        }
    }

    # A slow beetle stalks the back of the line.
    Entity { id("beetle") startPosition(780, 700) color("65a30d") Type { diamond(44, 36) }
        hostile("larva3", "-- THE BEETLE ATE THE LAST LARVA --", "lose")
        Movement { speed(0.8) chase("larva3") }
        Events { OnElapsedTime(3) { repeatCount(-1) Actions { flash("bef264", 0.3) } } } }
}
