# Rooftop Garden: tap the sunflower for seeds and spend them on helpers.
# Watering cans and beehives harvest by themselves, the greenhouse doubles every tap.
# Aphids crawl onto the flower and nibble your seeds: tap them away.
# Fill the seed jar (150 seeds) to win; six aphids at once and the garden is lost.
Window {
    size(8, 11)
    background("064e3b")
    music("bg_nature")

    Bar("seeds") { initial(0) }
    winAt("seeds", 150, "-- THE SEED JAR IS FULL --", "win")
    Bar("greenhouse") { initial(0) }
    Bar("pests") {
        initial(0)
        OnValue(4) { Actions { setText("status", "Aphids everywhere!") flash("sunflower", "84cc16", 0.4) } }
        OnValue(6) { Actions { sound("lose") shake(8, 0.5) gameOver("-- THE APHIDS ATE THE GARDEN --") } }
    }

    Entity {
        id("sunflower")
        startPosition(230, 470)
        color("facc15")
        Type { star(12, 115, 70) }
        Events {
            OnTapped("sunflower") {
                Actions {
                    sound("pop")
                    updateBar("seeds", 1)
                    whenBar("greenhouse", "==", 1) { Actions { updateBar("seeds", 1) } }
                    burst(4, "fde68a")
                    setScale(1.06)
                    delay(0.08) { Actions { setScale(1) } }
                }
            }
        }
    }

    Entity { id("aphid") template(true) startPosition(0, 0) color("65a30d") Type { circle(14) }
        Events {
            OnTapped { Actions { sound("click") burst(8, "a3e635") updateBar("pests", -1) deleteEntity() } }
            OnElapsedTime(random(1, 2)) { repeatCount(-1) Actions {
                whenBar("seeds", ">", 0) { Actions { updateBar("seeds", -1) setScale(1.2) delay(0.15) { Actions { setScale(1) } } } }
            } }
        } }

    Entity {
        id("weather")
        startPosition(40, 980)
        color("bbf7d0")
        Type { text("Spring", 36) }
        Events {
            Sequence {
                Phase(3, 7) { Actions { broadcast("aphid") } }
                Phase(1, 1) { Actions { setText("weather", "Summer") } }
                Phase(8, 4) { Actions { broadcast("aphid") } }
                Phase(1, 1) { Actions { setText("weather", "Heatwave") setColor("fca5a5") sound("beep") } }
                Phase(-1, random(1.5, 3)) { Actions { broadcast("aphid") } }
            }
            OnSignal("aphid") { Actions {
                spawn("aphid", random(240, 440), random(480, 680))
                updateBar("pests", 1)
                sound("buzz")
            } }
        }
    }

    # helpers, parked until bought
    Entity { id("can1") startPosition(40, 300) color("38bdf8") Type { rectangle(60, 44) }
        Events { OnElapsedTime(1) { repeatCount(-1) Actions { updateBar("seeds", 1) burst(3, "7dd3fc") } } } }
    Entity { id("can2") startPosition(40, 370) color("38bdf8") Type { rectangle(60, 44) }
        Events { OnElapsedTime(1) { repeatCount(-1) Actions { updateBar("seeds", 1) burst(3, "7dd3fc") } } } }
    Entity { id("can3") startPosition(40, 440) color("38bdf8") Type { rectangle(60, 44) }
        Events { OnElapsedTime(1) { repeatCount(-1) Actions { updateBar("seeds", 1) burst(3, "7dd3fc") } } } }
    Entity { id("hive1") startPosition(40, 560) color("f59e0b") Type { polygon(6, 30) }
        Events { OnElapsedTime(1.5) { repeatCount(-1) Actions { burst(5, "fde047") loopCount(2) updateBar("seeds", 1) } } } }
    Entity { id("hive2") startPosition(40, 640) color("f59e0b") Type { polygon(6, 30) }
        Events { OnElapsedTime(1.5) { repeatCount(-1) Actions { burst(5, "fde047") loopCount(2) updateBar("seeds", 1) } } } }

    Entity {
        id("buyCan")
        startPosition(30, 60)
        color("7dd3fc")
        Type { text("Can 12", 36) }
        var("owned", 0)
        Events {
            OnTapped("buyCan") {
                Actions {
                    whenVar("owned", "<", 3) { Actions { whenBar("seeds", ">=", 12) { Actions {
                        sound("coin")
                        updateBar("seeds", -12)
                        changeVar("owned", 1)
                        whenVar("owned", "==", 1) { Actions { setEnable("can1", true) } }
                        whenVar("owned", "==", 2) { Actions { setEnable("can2", true) } }
                        whenVar("owned", "==", 3) { Actions { setEnable("can3", true) setText("buyCan", "Cans: done") } }
                    } } } }
                }
            }
            OnElapsedTime(0.25) {
                repeatCount(-1)
                Actions {
                    whenVar("owned", "<", 3) { Actions {
                        whenBar("seeds", ">=", 12) { Actions { setOpacity(1) } }
                        whenBar("seeds", "<", 12) { Actions { setOpacity(0.4) } }
                    } }
                    whenVar("owned", "==", 3) { Actions { setOpacity(0.25) } }
                }
            }
        }
    }

    Entity {
        id("buyHive")
        startPosition(250, 60)
        color("fcd34d")
        Type { text("Hive 30", 36) }
        var("owned", 0)
        Events {
            OnTapped("buyHive") {
                Actions {
                    whenVar("owned", "<", 2) { Actions { whenBar("seeds", ">=", 30) { Actions {
                        sound("coin")
                        updateBar("seeds", -30)
                        changeVar("owned", 1)
                        whenVar("owned", "==", 1) { Actions { setEnable("hive1", true) } }
                        whenVar("owned", "==", 2) { Actions { setEnable("hive2", true) setText("buyHive", "Hives: done") } }
                    } } } }
                }
            }
            OnElapsedTime(0.25) {
                repeatCount(-1)
                Actions {
                    whenVar("owned", "<", 2) { Actions {
                        whenBar("seeds", ">=", 30) { Actions { setOpacity(1) } }
                        whenBar("seeds", "<", 30) { Actions { setOpacity(0.4) } }
                    } }
                    whenVar("owned", "==", 2) { Actions { setOpacity(0.25) } }
                }
            }
        }
    }

    Entity {
        id("buyHouse")
        startPosition(480, 60)
        color("86efac")
        Type { text("Greenhouse 45", 32) }
        Events {
            OnTapped("buyHouse") {
                Actions {
                    whenBar("greenhouse", "==", 0) { Actions { whenBar("seeds", ">=", 45) { Actions {
                        sound("sparkle")
                        updateBar("seeds", -45)
                        updateBar("greenhouse", 1)
                        setText("buyHouse", "Greenhouse built")
                        setText("status", "Every tap now gives 2 seeds")
                        burst(16, "86efac")
                    } } } }
                }
            }
            OnElapsedTime(0.25) {
                repeatCount(-1)
                Actions {
                    whenBar("greenhouse", "==", 0) { Actions {
                        whenBar("seeds", ">=", 45) { Actions { setOpacity(1) } }
                        whenBar("seeds", "<", 45) { Actions { setOpacity(0.4) } }
                    } }
                    whenBar("greenhouse", "==", 1) { Actions { setOpacity(0.25) } }
                }
            }
        }
    }

    # the seed jar: one segment for every 15 seeds
    Entity { id("jar1") startPosition(680, 160) color("fde68a") Type { rectangle(80, 40) } }
    Entity { id("jar2") startPosition(680, 205) color("fde68a") Type { rectangle(80, 40) } }
    Entity { id("jar3") startPosition(680, 250) color("fde68a") Type { rectangle(80, 40) } }
    Entity { id("jar4") startPosition(680, 295) color("fcd34d") Type { rectangle(80, 40) } }
    Entity { id("jar5") startPosition(680, 340) color("fcd34d") Type { rectangle(80, 40) } }
    Entity { id("jar6") startPosition(680, 385) color("fcd34d") Type { rectangle(80, 40) } }
    Entity { id("jar7") startPosition(680, 430) color("fbbf24") Type { rectangle(80, 40) } }
    Entity { id("jar8") startPosition(680, 475) color("fbbf24") Type { rectangle(80, 40) } }
    Entity { id("jar9") startPosition(680, 520) color("f59e0b") Type { rectangle(80, 40) } }
    Entity { id("jar10") startPosition(680, 565) color("f59e0b") Type { rectangle(80, 40) } }

    Entity {
        id("status")
        startPosition(40, 900)
        color("ecfccb")
        Type { text("Tap the sunflower!", 36) }
        Events {
            OnElapsedTime {
                Actions {
                    setEnable("can1", false) setEnable("can2", false) setEnable("can3", false)
                    setEnable("hive1", false) setEnable("hive2", false)
                }
            }
            OnElapsedTime(0.2) {
                repeatCount(-1)
                Actions {
                    whenBar("seeds", ">=", 15) { Actions { setVisible("jar1", true) } }
                    whenBar("seeds", "<", 15) { Actions { setVisible("jar1", false) } }
                    whenBar("seeds", ">=", 30) { Actions { setVisible("jar2", true) } }
                    whenBar("seeds", "<", 30) { Actions { setVisible("jar2", false) } }
                    whenBar("seeds", ">=", 45) { Actions { setVisible("jar3", true) } }
                    whenBar("seeds", "<", 45) { Actions { setVisible("jar3", false) } }
                    whenBar("seeds", ">=", 60) { Actions { setVisible("jar4", true) } }
                    whenBar("seeds", "<", 60) { Actions { setVisible("jar4", false) } }
                    whenBar("seeds", ">=", 75) { Actions { setVisible("jar5", true) } }
                    whenBar("seeds", "<", 75) { Actions { setVisible("jar5", false) } }
                    whenBar("seeds", ">=", 90) { Actions { setVisible("jar6", true) } }
                    whenBar("seeds", "<", 90) { Actions { setVisible("jar6", false) } }
                    whenBar("seeds", ">=", 105) { Actions { setVisible("jar7", true) } }
                    whenBar("seeds", "<", 105) { Actions { setVisible("jar7", false) } }
                    whenBar("seeds", ">=", 120) { Actions { setVisible("jar8", true) } }
                    whenBar("seeds", "<", 120) { Actions { setVisible("jar8", false) } }
                    whenBar("seeds", ">=", 135) { Actions { setVisible("jar9", true) } }
                    whenBar("seeds", "<", 135) { Actions { setVisible("jar9", false) } }
                    whenBar("seeds", ">=", 150) { Actions { setVisible("jar10", true) } }
                    whenBar("seeds", "<", 150) { Actions { setVisible("jar10", false) } }
                }
            }
        }
    }
    Entity { id("jarLabel") startPosition(670, 620) color("fef3c7") Type { text("JAR", 32) } }
}
