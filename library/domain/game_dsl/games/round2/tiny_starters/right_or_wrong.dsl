Window {
    size(8, 8)
    background("172554")
    winAt(6, "-- ALL GREEN --", "win")
    Bar { initial(0) OnValue(-3) { Actions { sound("lose") gameOver("-- TOO MANY WRONG TAPS --") } } }

    Entity {
        id("hint")
        startPosition(170, 700)
        color("dbeafe")
        Type { text("Tap the green circle only", 36) }
    }

    Entity {
        id("good")
        startPosition(120, 300)
        color("22c55e")
        Type { circle(80) }
        Events {
            OnTapped {
                Actions {
                    sound("coin")
                    updateBar(1)
                    setPosition("bad", random(60, 560), random(100, 500), 0)
                }
            }
        }
    }

    Entity {
        id("bad")
        startPosition(500, 300)
        color("ef4444")
        Type { circle(80) }
        Events {
            OnTapped {
                Actions {
                    sound("buzz")
                    updateBar(-1)
                    flash("ffffff", 0.2)
                }
            }
        }
    }
}
