Window {
    size(8, 9)
    background("422006")
    music("bg_nature")
    Bar {
        initial(0)
        OnValue(8) { Actions { sound("win") gameOver("ALL WHACKED") } }
    }
    Entity {
        id("moleA")
        startPosition(100, 600)
        color("a16207")
        Type { circle(50) }
        Events {
            OnTapped("moleA") {
                Actions {
                    sound("thud")
                    updateBar(1)
                    setPosition("moleA", random(60, 640), random(100, 700), 0)
                }
            }
            OnElapsedTime(2) { repeatCount(-1) Actions { setPosition("moleA", random(60, 640), random(100, 700), 0) } }
        }
    }
    Entity {
        id("moleB")
        startPosition(500, 300)
        color("92400e")
        Type { circle(40) }
        Events {
            OnTapped("moleB") {
                Actions {
                    sound("pop")
                    updateBar(1)
                    setPosition("moleB", random(60, 640), random(100, 700), 0)
                }
            }
            OnElapsedTime(1.4) { repeatCount(-1) Actions { setPosition("moleB", random(60, 640), random(100, 700), 0) } }
        }
    }
    Entity {
        id("stone")
        startPosition(300, 400)
        color("6b7280")
        Type { polygon(6, 45) }
        Events {
            OnTapped("stone") { Actions { sound("clang") updateBar(-1) flash("ef4444", 0.3) } }
            OnElapsedTime(2.5) { repeatCount(-1) Actions { setPosition("stone", random(60, 640), random(100, 700), 0) } }
        }
    }
    Entity {
        id("clock") startPosition(20, 800) color("fde68a") Type { text("Tap 8 moles, not the stone", 34) }
        Events { OnElapsedTime(25) { Actions { sound("lose") gameOver("THE MOLES WIN") } } }
    }
}
