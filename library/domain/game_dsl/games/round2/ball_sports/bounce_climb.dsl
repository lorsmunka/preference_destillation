# Drag sideways to land on the springs and climb to the golden basket.
Window {
    size(8, 12)
    background("0c4a6e")
    gravity(700)
    music("bg_5")

    lives(3, "TOO MANY FALLS", "lose")

    goal("basket", 340, 1100, 120, 70, "SLAM DUNK!")

    Entity {
        id("spring1")
        startPosition(300, 120)
        color("22c55e")
        Type { rectangle(200, 20) }
        Events {
            OnCollision("hero") { Actions { sound("jump") setVelocity("hero", 0, 640) } }
        }
    }
    Entity {
        id("spring2")
        startPosition(80, 380)
        color("84cc16")
        Type { rectangle(180, 20) }
        Events {
            OnCollision("hero") { Actions { sound("jump") setVelocity("hero", 0, 680) } }
        }
    }
    Entity {
        id("spring3")
        startPosition(520, 640)
        color("eab308")
        Type { rectangle(180, 20) }
        Events {
            OnCollision("hero") { Actions { sound("jump") setVelocity("hero", 0, 700) burst(6, "fde047") } }
        }
    }
    Entity {
        id("spring4")
        startPosition(120, 880)
        color("f97316")
        Type { rectangle(160, 20) }
        Events {
            OnCollision("hero") { Actions { sound("sparkle") setVelocity("hero", 0, 760) burst(8, "fdba74") } }
        }
    }

    Entity {
        id("spikes")
        startPosition(0, 0)
        color("ef4444")
        Type { rectangle(800, 24) }
        Events {
            OnCollision("hero") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    setPosition("hero", 360, 150, 0)
                    setVelocity("hero", 0, 0)
                }
            }
        }
    }

    Entity {
        id("hero")
        startPosition(360, 180)
        color("f8fafc")
        Type { circle(26) }
        Movement { speed(5) Control { movementControl(2) } }
    }
}
