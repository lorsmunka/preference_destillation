Window {
    size(10, 7)
    background("14532d")
    music("bg_nature")
    gravity(600)

    lives(4, "-- THE FROG WENT UNDER --", "lose")
    Bar("flies") {
        initial(0)
        OnValue(3) { Actions { sound("sparkle") updateBar("lives", 1) } }
    }

    Level {
        Entity { id("hero") startPosition(40, 175) color("4ade80")
            Type { text("🐸", 40) }
            Movement { speed(0) AutoMove { direction(1, 0) } }
            var("grounded", 0)
            Events {
                OnSignal("hop") { Actions { whenVar("grounded", "==", 1) { Actions {
                    setVelocity(200, 450) changeVar("grounded", -1) sound("jump") } } } }
                OnSignal("leap") { Actions { whenVar("grounded", "==", 1) { Actions {
                    setVelocity(300, 500) changeVar("grounded", -1) sound("jump") burst(6, "bbf7d0") } } } }
                OnSignal("back") { Actions { whenVar("grounded", "==", 1) { Actions {
                    setVelocity(-200, 450) changeVar("grounded", -1) sound("jump") } } } }
                # the sticky top of a pad stops the slide
                OnCollision("grip") { Actions {
                    setVelocity(0, 0)
                    whenVar("grounded", "==", 0) { Actions { changeVar("grounded", 1) } } } }
                OnCollision("pond") { Actions {
                    sound("bubble") burst(16, "7dd3fc") shake(4, 0.3) updateBar("lives", -1)
                    setPosition("hero", 40, 175, 0) setVelocity(0, 0) } }
                OnCollision("burrow") { Actions { sound("win2") burst(20, "fde047") nextLevel() } } } }

        Entity { id("bankL") startPosition(0, 0) color("65a30d") oneWay(true) Type { rectangle(130, 160) } }
        Entity { id("pad1") startPosition(300, 144) color("15803d") oneWay(true) Type { rectangle(120, 16) } }
        Entity { id("pad2") startPosition(600, 144) color("15803d") oneWay(true) Type { rectangle(120, 16) } }
        Entity { id("bankR") startPosition(870, 0) color("65a30d") oneWay(true) Type { rectangle(130, 160) } }
        Entity { id("burrow") startPosition(930, 160) color("713f12") Type { rectangle(70, 60) } }

        Entity { id("grip") template(true) startPosition(0, 0) color("00000000") Type { rectangle(120, 14) } }
        Entity { id("fly") template(true) startPosition(0, 0) color("1f2937") Type { text("🪰", 26) }
            Events { OnCollision("hero") { Actions { sound("coin") updateBar("flies", 1) burst(8, "fde047") deleteEntity() } } } }

        # the pond lays out the pad tops and flies, then sinks the second pad every few seconds
        Entity { id("pond") startPosition(130, 0) color("0e7490") Type { rectangle(740, 70) }
            Events {
                OnElapsedTime { Actions {
                    spawn("grip", 5, 154) spawn("grip", 300, 154) spawn("grip", 600, 154) spawn("grip", 875, 154)
                    spawn("fly", 330, 300) spawn("fly", 640, 330) } }
                OnElapsedTime(3.5) { repeatCount(-1) Actions {
                    flash("pad2", "ef4444", 0.6)
                    delay(0.6) { Actions { sound("bubble") setEnable("pad2", false)
                        delay(1.6) { Actions { setEnable("pad2", true) } } } } } } } }

        Entity { id("padBack") startPosition(20, 610) color("a3e635") Type { rectangle(300, 70) }
            Events { OnTapped("padBack") { Actions { broadcast("back") } } } }
        Entity { id("padHop") startPosition(350, 610) color("4ade80") Type { rectangle(300, 70) }
            Events { OnTapped("padHop") { Actions { broadcast("hop") } } } }
        Entity { id("padLeap") startPosition(680, 610) color("22c55e") Type { rectangle(300, 70) }
            Events { OnTapped("padLeap") { Actions { broadcast("leap") } } } }
        Entity { id("lblBack") startPosition(120, 630) color("14532d") Type { text("< BACK", 28) } }
        Entity { id("lblHop") startPosition(465, 630) color("14532d") Type { text("HOP", 28) } }
        Entity { id("lblLeap") startPosition(785, 630) color("14532d") Type { text("LEAP >>", 28) } }
    }

    Level {
        Entity { id("hero") startPosition(40, 175) color("4ade80")
            Type { text("🐸", 40) }
            Movement { speed(0) AutoMove { direction(1, 0) } }
            var("grounded", 0)
            Events {
                OnSignal("hop") { Actions { whenVar("grounded", "==", 1) { Actions {
                    setVelocity(200, 450) changeVar("grounded", -1) sound("jump") } } } }
                OnSignal("leap") { Actions { whenVar("grounded", "==", 1) { Actions {
                    setVelocity(300, 500) changeVar("grounded", -1) sound("jump") burst(6, "bbf7d0") } } } }
                OnSignal("back") { Actions { whenVar("grounded", "==", 1) { Actions {
                    setVelocity(-200, 450) changeVar("grounded", -1) sound("jump") } } } }
                OnCollision("grip") { Actions {
                    setVelocity(0, 0)
                    whenVar("grounded", "==", 0) { Actions { changeVar("grounded", 1) } } } }
                OnCollision("pond") { Actions {
                    sound("bubble") burst(16, "7dd3fc") shake(4, 0.3) updateBar("lives", -1)
                    setPosition("hero", 40, 175, 0) setVelocity(0, 0) } }
                # the heron swats a frog out of the air
                OnCollision("heron") { Actions { sound("hurt") flash("ff0000", 0.3) setVelocity(0, -300) } } } }

        Entity { id("bankL") startPosition(0, 0) color("65a30d") oneWay(true) Type { rectangle(130, 160) } }
        Entity { id("pad1") startPosition(280, 144) color("15803d") oneWay(true) Type { rectangle(120, 16) } }
        Entity { id("pad2") startPosition(580, 144) color("15803d") oneWay(true) Type { rectangle(120, 16) } }
        Entity { id("bankR") startPosition(870, 0) color("65a30d") oneWay(true) Type { rectangle(130, 160) } }
        goal("home", 900, 160, 100, 60, "-- HOME SWEET POND --")

        Entity { id("grip") template(true) startPosition(0, 0) color("00000000") Type { rectangle(120, 14) } }
        Entity { id("fly") template(true) startPosition(0, 0) color("1f2937") Type { text("🪰", 26) }
            Events { OnCollision("hero") { Actions { sound("coin") updateBar("flies", 1) burst(8, "fde047") deleteEntity() } } } }

        Entity { id("heron") startPosition(200, 320) color("e2e8f0") facing("left")
            Type { text("🦢", 44) }
            Movement { speed(1.5) Path { waypoint(200, 320) waypoint(760, 320) dwell(0.5) } } }

        # both pads sink here, on different beats
        Entity { id("pond") startPosition(130, 0) color("0e7490") Type { rectangle(740, 70) }
            Events {
                OnElapsedTime { Actions {
                    spawn("grip", 5, 154) spawn("grip", 280, 154) spawn("grip", 580, 154) spawn("grip", 875, 154)
                    spawn("fly", 470, 320) spawn("fly", 760, 300) } }
                OnElapsedTime(3) { repeatCount(-1) Actions {
                    flash("pad1", "ef4444", 0.5)
                    delay(0.5) { Actions { sound("bubble") setEnable("pad1", false)
                        delay(1.4) { Actions { setEnable("pad1", true) } } } } } }
                OnElapsedTime(4.2) { repeatCount(-1) Actions {
                    flash("pad2", "ef4444", 0.5)
                    delay(0.5) { Actions { sound("bubble") setEnable("pad2", false)
                        delay(1.8) { Actions { setEnable("pad2", true) } } } } } } } }

        Entity { id("padBack") startPosition(20, 610) color("a3e635") Type { rectangle(300, 70) }
            Events { OnTapped("padBack") { Actions { broadcast("back") } } } }
        Entity { id("padHop") startPosition(350, 610) color("4ade80") Type { rectangle(300, 70) }
            Events { OnTapped("padHop") { Actions { broadcast("hop") } } } }
        Entity { id("padLeap") startPosition(680, 610) color("22c55e") Type { rectangle(300, 70) }
            Events { OnTapped("padLeap") { Actions { broadcast("leap") } } } }
        Entity { id("lblBack") startPosition(120, 630) color("14532d") Type { text("< BACK", 28) } }
        Entity { id("lblHop") startPosition(465, 630) color("14532d") Type { text("HOP", 28) } }
        Entity { id("lblLeap") startPosition(785, 630) color("14532d") Type { text("LEAP >>", 28) } }
    }
}
