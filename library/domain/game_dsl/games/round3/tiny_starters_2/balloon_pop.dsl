Window {
    size(8, 11)
    background("bae6fd")
    music("bg_5")
    lives(3, "TOO MANY ESCAPED", "lose")
    winAt(10, "POPPED THEM ALL", "win")
    Entity {
        id("balloon")
        template(true)
        startPosition(0, 0)
        color("f43f5e")
        Type { circle(42) }
        Movement { speed(2) AutoMove { direction(0, 1) } }
        Events {
            OnTapped {
                Actions {
                    sound("pop")
                    burst(10, "f43f5e")
                    updateBar(1)
                    deleteEntity()
                }
            }
            OnCollision { Actions { updateBar("lives", -1) sound("blip") deleteEntity() } }
        }
    }
    Entity {
        id("spawner")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1.1) { repeatCount(-1) Actions { spawn("balloon", random(40, 720), 20) } }
        }
    }
    Entity {
        id("banner")
        startPosition(20, 20)
        color("0369a1")
        Type { text("Tap balloons before they escape", 28) }
    }
    Entity {
        id("cloud")
        startPosition(100, 900)
        color("f8fafc")
        Type { rectangle(200, 50) }
        Movement { speed(1) AutoMove { direction(1, 0) } }
        Events {
            OnCollision { Actions { setPosition("cloud", 0, 900, 0) } }
        }
    }
}
