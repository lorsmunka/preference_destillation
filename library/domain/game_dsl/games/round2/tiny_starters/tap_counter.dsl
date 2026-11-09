Window {
    size(6, 8)
    background("1f2937")
    winAt(20, "-- 20 TAPS! --", "win")

    Entity {
        id("button")
        startPosition(180, 280)
        color("22c55e")
        Type { circle(120) }
        Events {
            OnTapped("button") {
                Actions {
                    sound("click")
                    updateBar(1)
                    flash("fef08a", 0.1)
                }
            }
        }
    }

    Entity {
        id("hint")
        startPosition(100, 640)
        color("e5e7eb")
        Type { text("Tap 20 times in 12 seconds", 32) }
    }

    Entity {
        id("clock")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(12.0) {
                Actions {
                    sound("lose")
                    gameOver("-- TIME IS UP --")
                }
            }
        }
    }
}
