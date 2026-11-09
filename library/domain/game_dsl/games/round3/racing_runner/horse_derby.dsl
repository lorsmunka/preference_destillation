Window {
    size(10, 8)
    background("365314")
    music("bg_nature")
    gravity(0)

    Entity { id("finish") startPosition(900, 0) color("f8fafc")
        Type { rectangle(16, 800) } }
    Entity { id("fenceTop") startPosition(0, 760) color("78350f")
        Type { rectangle(1000, 20) } }

    Entity {
        id("hero")
        startPosition(20, 120)
        color("ffffff")
        Type { text("🐎", 70) }
        facing("left")
        Movement { speed(1.5) AutoMove { direction(1, 0) } }
        Events {
            OnCollision("finish") {
                Actions {
                    sound("win")
                    shake(8, 0.5)
                    gameOver("-- YOUR HORSE WINS --")
                }
            }
            OnSignal("spur") { Actions { updateSpeed(3.6) } }
            OnElapsedTime(0.8) { repeatCount(-1) Actions { updateSpeed(1.5) } }
        }
    }
    Entity {
        id("rivalOne")
        startPosition(20, 330)
        color("ffffff")
        Type { text("🐎", 70) }
        facing("left")
        Movement { speed(2.2) AutoMove { direction(1, 0) } }
        Events {
            OnCollision("finish") { Actions { sound("lose") gameOver("-- RIVAL ONE WINS --") } }
            OnElapsedTime(3) { repeatCount(-1) Actions { updateSpeed(random(1.8, 3)) } }
        }
    }
    Entity {
        id("rivalTwo")
        startPosition(20, 540)
        color("ffffff")
        Type { text("🐎", 70) }
        facing("left")
        Movement { speed(2.4) AutoMove { direction(1, 0) } }
        Events {
            OnCollision("finish") { Actions { sound("lose") gameOver("-- RIVAL TWO WINS --") } }
            OnElapsedTime(2) { repeatCount(-1) Actions { updateSpeed(random(1.5, 3.2)) } }
        }
    }

    Entity { id("spurButton") startPosition(40, 20) color("f59e0b")
        Type { rectangle(240, 70) }
        Events {
            OnTapped("spurButton") {
                Actions {
                    broadcast("spur")
                    sound("click")
                    burst("hero", 6, "fde68a")
                }
            }
        } }
    Entity { id("spurLabel") startPosition(70, 35) color("1c1917")
        Type { text("SPUR!", 36) } }
}
