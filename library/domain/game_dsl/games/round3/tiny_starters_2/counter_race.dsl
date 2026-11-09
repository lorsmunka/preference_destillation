Window {
    size(8, 9)
    background("1f2937")
    music("bg_1")
    Bar("you") {
        initial(0)
        OnValue(20) { Actions { sound("win") gameOver("YOU WIN THE RACE") } }
    }
    Bar("rival") {
        initial(0)
        OnValue(20) { Actions { sound("lose") gameOver("THE RIVAL WINS") } }
    }
    Entity {
        id("small")
        startPosition(100, 200)
        color("22c55e")
        Type { rectangle(240, 200) }
        Events {
            OnTapped("small") { Actions { sound("click") updateBar("you", 1) } }
        }
    }
    Entity {
        id("big")
        startPosition(460, 200)
        color("f97316")
        Type { rectangle(240, 200) }
        Events {
            OnTapped("big") {
                Actions {
                    sound("bell")
                    updateBar("you", 3)
                    setEnable("big", false)
                    delay(2) { Actions { setEnable("big", true) } }
                }
            }
        }
    }
    Entity {
        id("rival")
        startPosition(100, 640)
        color("a855f7")
        Type { circle(40) }
        Events {
            OnElapsedTime(0.55) { repeatCount(-1) Actions { updateBar("rival", 1) flash("ffffff", 0.1) } }
        }
    }
    Entity {
        id("title")
        startPosition(180, 760)
        color("e5e7eb")
        Type { text("First to 20 wins", 40) }
    }
}
