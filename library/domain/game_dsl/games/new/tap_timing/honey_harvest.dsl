# Honey Harvest: scoop honey from the hive into your pot, then pour the pot into the jar.
# Two scoops a trip are safe. Every scoop after that pays more, but the bees may sting
# and the pot spills. Pouring waits for the bees to calm down, and the sun is setting.
Window {
    size(7, 11)
    background("422006")
    music("bg_5")

    Bar("pot") {
        initial(0)
        OnValue(0) { Actions { setScale("honey", 0.3) } }
        OnValue(3) { Actions { setScale("honey", 0.7) } }
        OnValue(6) { Actions { setScale("honey", 1.1) } }
        OnValue(9) { Actions { setScale("honey", 1.5) } }
    }
    Bar("jarred") {
        initial(0)
        OnValue(10) { Actions { setText("status", "A third of the jar!") } }
        OnValue(20) { Actions { setText("status", "Ten more to go!") } }
    }
    winAt("jarred", 30, "-- THE JAR IS FULL --", "win")
    Bar("buzz") {
        initial(0)
        OnValue(3) { Actions { broadcast("angry") } }
        OnValue(0) { Actions { broadcast("calm") } }
    }
    Bar("mode") { initial(0) }
    Bar("stings") { initial(0) OnValue(3) { Actions { sound("lose") shake(8, 0.5) gameOver("-- STUNG ONE TIME TOO MANY --") } } }

    Entity {
        id("hive")
        startPosition(60, 560)
        color("d97706")
        Type { polygon(6, 100) }
        Events {
            OnTapped("hive") {
                Actions {
                    whenBar("mode", "==", 0) { Actions {
                        sound("pop")
                        updateBar("buzz", 1)
                        whenBar("buzz", "<=", 2) { Actions { randomAction {
                            Actions { updateBar("pot", 1) }
                            Actions { updateBar("pot", 1) }
                            Actions { updateBar("pot", 1) }
                            Actions { loopCount(2) updateBar("pot", 1) }
                        } } }
                        whenBar("buzz", ">=", 3) { Actions { randomAction {
                            Actions { sound("coin") loopCount(3) updateBar("pot", 1) }
                            Actions { sound("coin") loopCount(3) updateBar("pot", 1) }
                            Actions { sound("coin") loopCount(4) updateBar("pot", 1) }
                            Actions { broadcast("sting") }
                        } } }
                    } }
                }
            }
            OnSignal("sting") {
                Actions {
                    sound("hurt")
                    shake(8, 0.4)
                    flash("ef4444", 0.4)
                    burst("honey", 12, "f59e0b")
                    updateBar("stings", 1)
                    updateBar("mode", 2)
                    setText("status", "Stung! The pot spilled")
                }
            }
        }
    }

    Entity {
        id("bees")
        startPosition(40, 820)
        color("fde047")
        Type { star(5, 18, 8) }
        Movement { speed(1.5) Path { waypoint(40, 820) waypoint(250, 860) waypoint(270, 560) waypoint(30, 520) } }
        Events {
            OnSignal("angry") { Actions { setColor("ef4444") updateSpeed(4) sound("buzz") } }
            OnSignal("calm") { Actions { setColor("fde047") updateSpeed(1.5) } }
        }
    }

    Entity { id("honey") startPosition(250, 240) color("f59e0b") Type { circle(50) }
        Events { OnElapsedTime { Actions { setScale(0.3) } } } }

    Entity {
        id("jar")
        startPosition(470, 160)
        color("fcd34d")
        Type { rectangle(150, 200) }
        Events {
            OnTapped("jar") {
                Actions {
                    whenBar("mode", "==", 0) { Actions { whenBar("pot", ">", 0) { Actions {
                        sound("bubble")
                        updateBar("mode", 1)
                        setText("status", "Pouring...")
                    } } } }
                }
            }
        }
    }

    Entity {
        id("ladle")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.12) {
                repeatCount(-1)
                Actions {
                    whenBar("mode", "==", 1) { Actions { whenBar("pot", ">", 0) { Actions {
                        updateBar("pot", -1) updateBar("jarred", 1) burst("jar", 3, "f59e0b") } } } }
                    whenBar("mode", "==", 2) { Actions { whenBar("pot", ">", 0) { Actions {
                        updateBar("pot", -1) } } } }
                }
            }
            OnElapsedTime(0.9) {
                repeatCount(-1)
                Actions {
                    whenBar("mode", ">", 0) { Actions { whenBar("buzz", ">", 0) { Actions { updateBar("buzz", -1) } } } }
                    whenBar("mode", ">", 0) { Actions { whenBar("pot", "==", 0) { Actions { whenBar("buzz", "==", 0) { Actions {
                        whenBar("mode", "==", 2) { Actions { updateBar("mode", -2) } }
                        whenBar("mode", "==", 1) { Actions { updateBar("mode", -1) } }
                        setText("status", "Scoop away!")
                        sound("blip")
                    } } } } } }
                }
            }
        }
    }

    Entity { id("status") startPosition(40, 60) color("fef3c7") Type { text("Tap the hive to scoop", 38) } }

    Entity {
        id("sun")
        startPosition(540, 980)
        color("fb923c")
        Type { circle(50) }
        Events {
            OnElapsedTime { Actions { setPosition("sun", 540, 600, 30) } }
            Sequence {
                Phase(1, 10) { Actions { setText("dusk", "20 s of daylight") } }
                Phase(1, 10) { Actions { setText("dusk", "10 s of daylight") setColor("ef4444") } }
                Phase(1, 5) { Actions { setText("dusk", "5 s!") sound("beep") } }
                Phase(1, 5) { Actions { sound("lose") gameOver("-- THE SUN SET ON A HALF-EMPTY JAR --") } }
            }
        }
    }
    Entity { id("dusk") startPosition(40, 1030) color("fdba74") Type { text("30 s of daylight", 34) } }
}
