# Tap GO only while the light is green. Tapping on red costs a life.
Window {
    size(7, 12)
    background("1f2937")
    music("bg_3")
    lives(3, "YOU JUMPED THE LIGHT", "lose")
    winAt("score", 8, "GREEN LIGHT MASTER", "win")
    Bar("green") { initial(0) }

    Entity {
        id("count")
        startPosition(300, 1000)
        color("fde68a")
        Type { text("3", 90) }
        Events {
            Sequence {
                Phase(1, 1) { Actions { setText("count", "2") sound("blip") } }
                Phase(1, 1) { Actions { setText("count", "1") sound("blip") } }
                Phase(1, 1) { Actions { setText("count", "GO") sound("bell") } }
                Phase(1, 1) { Actions { setText("count", "") } }
            }
        }
    }

    Entity {
        id("pole")
        startPosition(280, 380)
        color("0f172a")
        Type { rectangle(140, 420) }
    }
    Entity {
        id("light")
        startPosition(310, 520)
        color("ef4444")
        Type { circle(40) }
        Events {
            OnElapsedTime(4.5) {
                repeatCount(-1)
                Actions {
                    setColor("22c55e")
                    updateBar("green", 1)
                    delay(1.3) { Actions {
                        setColor("ef4444")
                        whenBar("green", "==", 1) { Actions { updateBar("green", -1) } }
                    } }
                }
            }
        }
    }
    Entity {
        id("go")
        startPosition(200, 80)
        color("38bdf8")
        Type { rectangle(300, 200) }
        Events {
            OnTapped("go") {
                Actions {
                    whenBar("green", "==", 0) { Actions {
                        updateBar("lives", -1)
                        sound("buzz")
                        flash("ef4444", 0.3)
                    } }
                    whenBar("green", "==", 1) { Actions {
                        updateBar("score", 1)
                        updateBar("green", -1)
                        sound("coin")
                        burst(10, "22c55e")
                    } }
                }
            }
        }
    }
}
