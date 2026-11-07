# Sprinkler Lawn: a cat that hates water crosses three gardens to reach the porch.
# Each sprinkler turns while it sprays, so its jets sweep round like the hands of a clock.
Window {
    size(8, 12)
    background("166534")
    music("bg_nature")
    gravity(0)

    lives(4, "-- SOAKED TO THE BONE --", "lose")

    Level {
        Entity {
            id("sign1")
            startPosition(20, 1160)
            color("dcfce7")
            Type { text("FRONT GARDEN", 28) }
        }

        joystick("hero", 80, 4.5)

        Entity {
            id("hero")
            startPosition(380, 30)
            color("fb923c")
            Type { circle(20) }
            Movement {
                speed(0)
                Control { movementControl(0) }
            }
            var("wet", 0)
            Events {
                OnSignal("soaked") {
                    Actions {
                        whenVar("wet", "==", 0) {
                            Actions {
                                changeVar("wet", 1)
                                sound("bubble")
                                updateBar("lives", -1)
                                flash("38bdf8", 1)
                                burst(12, "7dd3fc")
                                delay(1) { Actions { changeVar("wet", -1) } }
                            }
                        }
                    }
                }
                OnCollision("porch") {
                    Actions {
                        sound("win2")
                        burst(20, "fde68a")
                        nextLevel()
                    }
                }
            }
        }

        Entity {
            id("porch")
            startPosition(300, 1080)
            color("a16207")
            Type { rectangle(200, 70) }
        }

        Entity {
            id("sprinklerA")
            startPosition(185, 420)
            color("94a3b8")
            Type { circle(15) }
            Events {
                OnElapsedTime(2.0) {
                    repeatCount(-1)
                    Actions {
                        spawnHere("dN")
                        delay(0.5) { Actions { spawnHere("dE") } }
                        delay(1.0) { Actions { spawnHere("dS") } }
                        delay(1.5) { Actions { spawnHere("dW") } }
                    }
                }
            }
        }

        Entity {
            id("sprinklerB")
            startPosition(585, 760)
            color("94a3b8")
            Type { circle(15) }
            Events {
                OnElapsedTime(2.0) {
                    repeatCount(-1)
                    Actions {
                        spawnHere("dS")
                        delay(0.5) { Actions { spawnHere("dW") } }
                        delay(1.0) { Actions { spawnHere("dN") } }
                        delay(1.5) { Actions { spawnHere("dE") } }
                    }
                }
            }
        }

        Entity { id("dN") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(3.2) AutoMove { direction(0, 1) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } }
                OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dE") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(3.2) AutoMove { direction(1, 0) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } }
                OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dS") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(3.2) AutoMove { direction(0, -1) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } }
                OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dW") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(3.2) AutoMove { direction(-1, 0) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } }
                OnCollision { Actions { deleteEntity() } } } }
    }

    Level {
        Entity { id("sign2") startPosition(20, 1160) color("dcfce7") Type { text("BACK GARDEN", 28) } }
        joystick("hero", 80, 4.5)
        Entity { id("hero") startPosition(380, 30) color("fb923c") Type { circle(20) }
            Movement { speed(0) Control { movementControl(0) } }
            var("wet", 0)
            Events {
                OnSignal("soaked") { Actions { whenVar("wet", "==", 0) { Actions {
                    changeVar("wet", 1) sound("bubble") updateBar("lives", -1)
                    flash("38bdf8", 1) burst(12, "7dd3fc") delay(1) { Actions { changeVar("wet", -1) } } } } } }
                OnCollision("porch") { Actions { sound("win2") burst(20, "fde68a") nextLevel() } }
            } }
        Entity { id("porch") startPosition(40, 1080) color("a16207") Type { rectangle(200, 70) } }

        # eight-way sprinklers: a full turn every 2.4 s
        Entity { id("sprinklerA") startPosition(390, 300) color("94a3b8") Type { circle(15) }
            Events { OnElapsedTime(2.4) { repeatCount(-1) Actions {
                spawnHere("dN") delay(0.3) { Actions { spawnHere("dNE") } } delay(0.6) { Actions { spawnHere("dE") } }
                delay(0.9) { Actions { spawnHere("dSE") } } delay(1.2) { Actions { spawnHere("dS") } }
                delay(1.5) { Actions { spawnHere("dSW") } } delay(1.8) { Actions { spawnHere("dW") } }
                delay(2.1) { Actions { spawnHere("dNW") } } } } } }
        Entity { id("sprinklerB") startPosition(150, 700) color("94a3b8") Type { circle(15) }
            Events { OnElapsedTime(2.4) { repeatCount(-1) Actions {
                spawnHere("dS") delay(0.3) { Actions { spawnHere("dSW") } } delay(0.6) { Actions { spawnHere("dW") } }
                delay(0.9) { Actions { spawnHere("dNW") } } delay(1.2) { Actions { spawnHere("dN") } }
                delay(1.5) { Actions { spawnHere("dNE") } } delay(1.8) { Actions { spawnHere("dE") } }
                delay(2.1) { Actions { spawnHere("dSE") } } } } } }
        Entity { id("sprinklerC") startPosition(620, 820) color("94a3b8") Type { circle(15) }
            Events { OnElapsedTime(1.6) { repeatCount(-1) Actions {
                spawnHere("dW") delay(0.4) { Actions { spawnHere("dN") } }
                delay(0.8) { Actions { spawnHere("dE") } } delay(1.2) { Actions { spawnHere("dS") } } } } } }

        Entity { id("dN") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(3.4) AutoMove { direction(0, 1) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dNE") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(2.4) AutoMove { direction(1, 1) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dE") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(3.4) AutoMove { direction(1, 0) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dSE") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(2.4) AutoMove { direction(1, -1) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dS") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(3.4) AutoMove { direction(0, -1) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dSW") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(2.4) AutoMove { direction(-1, -1) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dW") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(3.4) AutoMove { direction(-1, 0) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dNW") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(2.4) AutoMove { direction(-1, 1) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
    }

    Level {
        Entity { id("sign3") startPosition(20, 1160) color("dcfce7") Type { text("THE BIG LAWN", 28) } }
        joystick("hero", 80, 4.5)
        Entity { id("hero") startPosition(380, 30) color("fb923c") Type { circle(20) }
            Movement { speed(0) Control { movementControl(0) } }
            var("wet", 0)
            Events {
                OnSignal("soaked") { Actions { whenVar("wet", "==", 0) { Actions {
                    changeVar("wet", 1) sound("bubble") updateBar("lives", -1)
                    flash("38bdf8", 1) burst(12, "7dd3fc") delay(1) { Actions { changeVar("wet", -1) } } } } } }
            } }
        goal("porch", 560, 1080, 200, 70, "-- WARM AND DRY ON THE PORCH --")

        # a robot mower crosses the lawn twice per lap; it soaks nobody, it just runs you over
        Entity { id("mower") startPosition(0, 0) color("dc2626") Type { rectangle(90, 60) }
            Movement { speed(2.8) Path { waypoint(20, 520) waypoint(690, 520) waypoint(690, 900) waypoint(20, 900) } }
            Events { OnCollision("hero") { Actions {
                sound("thud") updateBar("lives", -1) shake(8, 0.4) burst("hero", 16, "dc2626")
                setPosition("hero", 380, 30, 0.4) } } } }

        Entity { id("sprinklerA") startPosition(200, 300) color("94a3b8") Type { circle(15) }
            Events { OnElapsedTime(2.0) { repeatCount(-1) Actions {
                spawnHere("dN") delay(0.25) { Actions { spawnHere("dNE") } } delay(0.5) { Actions { spawnHere("dE") } }
                delay(0.75) { Actions { spawnHere("dSE") } } delay(1.0) { Actions { spawnHere("dS") } }
                delay(1.25) { Actions { spawnHere("dSW") } } delay(1.5) { Actions { spawnHere("dW") } }
                delay(1.75) { Actions { spawnHere("dNW") } } } } } }
        Entity { id("sprinklerB") startPosition(580, 700) color("94a3b8") Type { circle(15) }
            Events { OnElapsedTime(2.0) { repeatCount(-1) Actions {
                spawnHere("dN") delay(0.25) { Actions { spawnHere("dNW") } } delay(0.5) { Actions { spawnHere("dW") } }
                delay(0.75) { Actions { spawnHere("dSW") } } delay(1.0) { Actions { spawnHere("dS") } }
                delay(1.25) { Actions { spawnHere("dSE") } } delay(1.5) { Actions { spawnHere("dE") } }
                delay(1.75) { Actions { spawnHere("dNE") } } } } } }

        Entity { id("dN") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(3.6) AutoMove { direction(0, 1) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dNE") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(2.6) AutoMove { direction(1, 1) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dE") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(3.6) AutoMove { direction(1, 0) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dSE") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(2.6) AutoMove { direction(1, -1) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dS") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(3.6) AutoMove { direction(0, -1) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dSW") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(2.6) AutoMove { direction(-1, -1) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dW") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(3.6) AutoMove { direction(-1, 0) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
        Entity { id("dNW") template(true) startPosition(0, 0) color("7dd3fc") Type { circle(10) }
            Movement { speed(2.6) AutoMove { direction(-1, 1) } }
            Events { OnCollision("hero") { Actions { broadcast("soaked") deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
    }
}
