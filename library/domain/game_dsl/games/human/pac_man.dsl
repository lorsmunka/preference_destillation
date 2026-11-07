Window {
    size(8, 12)
    background("0a0a14")
    music("bg_4")
    gravity(0)

    Bar("pellets") {
        initial(0)
        OnValue(34) { Actions { sound("win") gameOver("-- MAZE CLEARED --") } }
    }
    lives(3, "-- GAME OVER --", "lose")

    Entity { id("w03") startPosition(360, 1080) color("1e3a8a")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("w11") startPosition(160, 980) color("1e3a8a")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("w15") startPosition(560, 980) color("1e3a8a")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("w21") startPosition(160, 880) color("1e3a8a")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("w23") startPosition(360, 880) color("1e3a8a")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("w25") startPosition(560, 880) color("1e3a8a")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("w41") startPosition(160, 680) color("1e3a8a")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("w43") startPosition(360, 680) color("1e3a8a")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("w45") startPosition(560, 680) color("1e3a8a")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("w51") startPosition(160, 580) color("1e3a8a")
        solid(true) Type { rectangle(100, 100) } }
    Entity { id("w55") startPosition(560, 580) color("1e3a8a")
        solid(true) Type { rectangle(100, 100) } }

    Entity { id("d01") startPosition(198, 1118) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d02") startPosition(298, 1118) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d04") startPosition(498, 1118) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d05") startPosition(598, 1118) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d10") startPosition(98, 1018) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d12") startPosition(298, 1018) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d13") startPosition(398, 1018) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d14") startPosition(498, 1018) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d16") startPosition(698, 1018) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d20") startPosition(98, 918) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d22") startPosition(298, 918) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d24") startPosition(498, 918) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d26") startPosition(698, 918) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d30") startPosition(98, 818) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d31") startPosition(198, 818) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d35") startPosition(598, 818) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d36") startPosition(698, 818) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d40") startPosition(98, 718) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d42") startPosition(298, 718) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d44") startPosition(498, 718) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d46") startPosition(698, 718) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d50") startPosition(98, 618) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d52") startPosition(298, 618) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d53") startPosition(398, 618) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d54") startPosition(498, 618) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d56") startPosition(698, 618) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d61") startPosition(198, 518) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d62") startPosition(298, 518) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d64") startPosition(498, 518) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }
    Entity { id("d65") startPosition(598, 518) color("fde68a")
        Type { circle(12) } Events { OnCollision("pac") { Actions {
        sound("blip") updateBar("pellets", 1) burst(6, "fde68a") deleteEntity() } } } }

    Entity { id("powTL") startPosition(88, 1108) color("fbbf24")
        Type { star(5, 22, 10) } Events { OnCollision("pac") { Actions {
        sound("pickup") updateBar("pellets", 1) broadcast("power")
        burst(16, "fbbf24") flash("pac", "fbbf24", 0.5)
        deleteEntity() } } } }
    Entity { id("powTR") startPosition(688, 1108) color("fbbf24")
        Type { star(5, 22, 10) } Events { OnCollision("pac") { Actions {
        sound("pickup") updateBar("pellets", 1) broadcast("power")
        burst(16, "fbbf24") flash("pac", "fbbf24", 0.5)
        deleteEntity() } } } }
    Entity { id("powBL") startPosition(88, 508) color("fbbf24")
        Type { star(5, 22, 10) } Events { OnCollision("pac") { Actions {
        sound("pickup") updateBar("pellets", 1) broadcast("power")
        burst(16, "fbbf24") flash("pac", "fbbf24", 0.5)
        deleteEntity() } } } }
    Entity { id("powBR") startPosition(688, 508) color("fbbf24")
        Type { star(5, 22, 10) } Events { OnCollision("pac") { Actions {
        sound("pickup") updateBar("pellets", 1) broadcast("power")
        burst(16, "fbbf24") flash("pac", "fbbf24", 0.5)
        deleteEntity() } } } }

    Entity {
        id("ghost2")
        startPosition(260, 780)
        color("ec4899")
        Type { circle(34) }
        var("scared", 0)
        Movement { speed(1.7) Path { waypoint(260, 1080) waypoint(260, 480) } }
        Events {
            OnElapsedTime(25.0) { Actions { updateSpeed(2.55) } }
            OnElapsedTime(50.0) { Actions { updateSpeed(3.4) } }
            OnSignal("power") {
                Actions { changeVar("scared", 1) setColor("2563eb")
                    delay(7.0) { Actions { changeVar("scared", -1)
                        whenVar("scared", "==", 0) { Actions {
                            setColor("ec4899") } } } } }
            }
            OnCollision("pac") {
                Actions {
                    whenVar("scared", "==", 0) { Actions {
                        sound("hurt")
                        updateBar("lives", -1)
                        flash("pac", "ff0000", 0.25)
                        shake(6, 0.3)
                        burst(12, "ec4899")
                        setPosition("ghost2", 260, 780, 0) } }
                    whenVar("scared", ">=", 1) { Actions {
                        sound("coin")
                        burst(14, "2563eb")
                        setPosition("ghost2", 260, 780, 0) } }
                }
            }
        }
    }

    Entity {
        id("ghost3")
        startPosition(460, 780)
        color("22d3ee")
        Type { circle(34) }
        var("scared", 0)
        Movement { speed(1.7) Path { waypoint(460, 1080) waypoint(460, 480) } }
        Events {
            OnElapsedTime(25.0) { Actions { updateSpeed(2.55) } }
            OnElapsedTime(50.0) { Actions { updateSpeed(3.4) } }
            OnSignal("power") {
                Actions { changeVar("scared", 1) setColor("2563eb")
                    delay(7.0) { Actions { changeVar("scared", -1)
                        whenVar("scared", "==", 0) { Actions {
                            setColor("22d3ee") } } } } }
            }
            OnCollision("pac") {
                Actions {
                    whenVar("scared", "==", 0) { Actions {
                        sound("hurt")
                        updateBar("lives", -1)
                        flash("pac", "ff0000", 0.25)
                        shake(6, 0.3)
                        burst(12, "22d3ee")
                        setPosition("ghost3", 460, 780, 0) } }
                    whenVar("scared", ">=", 1) { Actions {
                        sound("coin")
                        burst(14, "2563eb")
                        setPosition("ghost3", 460, 780, 0) } }
                }
            }
        }
    }

    Entity {
        id("pac")
        startPosition(360, 480)
        color("fcd34d")
        Type { circle(35) }
        Movement { GridMove(100, 0.32){ direction(0, 0) } }
    }

    dpad("pac")
}
