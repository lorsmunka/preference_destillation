Window {
    size(8, 12)
    background("365314")
    joystick("dog", 80, 5)
    winAt(6, "GOOD DOG", "win")
    Entity {
        id("dog")
        startPosition(380, 600)
        color("b45309")
        Type { circle(30) }
        Movement { speed(0) Control { movementControl(0) } }
    }
    Entity {
        id("bone")
        startPosition(600, 900)
        color("fef3c7")
        Type { rectangle(50, 20) }
        Events {
            OnCollision("dog") {
                Actions {
                    sound("pickup")
                    updateBar(1)
                    setPosition("bone", random(60, 700), random(300, 1100), 0)
                }
            }
        }
    }
    Entity {
        id("crow")
        startPosition(60, 1000)
        color("111827")
        Type { triangle(40, "down") }
        Movement { speed(2) chase("bone", 0.3) }
        Events {
            OnCollision("bone") { Actions { sound("blip") setPosition("bone", random(60, 700), random(300, 1100), 0) } }
        }
    }
    Entity {
        id("clock")
        startPosition(20, 1150)
        color("fde68a")
        Type { text("Fetch 6 bones before time runs out", 26) }
        Events {
            OnElapsedTime(40) { Actions { sound("lose") gameOver("THE BONES ARE GONE") } }
        }
    }
}
