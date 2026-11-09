Window {
    size(9, 10)
    background("7dd3fc")
    gravity(500)
    music("bg_1")

    Bar("cans") { initial(0) }
    Bar("throws") {
        initial(6)
        OnValue(0) { Actions { sound("lose") gameOver("NO BALLS LEFT") } }
    }
    winAt("cans", 6, "ALL CANS DOWN", "win")

    Entity {
        id("ground")
        startPosition(0, 0)
        color("65a30d")
        solid(true)
        Type { rectangle(900, 40) }
    }
    Entity {
        id("shelf")
        startPosition(560, 300)
        color("92400e")
        solid(true)
        Type { rectangle(300, 24) }
    }

    Entity {
        id("can")
        template(true)
        startPosition(0, 0)
        color("ef4444")
        Type { rectangle(50, 70) }
        Events {
            OnCollision("ball") {
                Actions {
                    sound("clang")
                    updateBar("cans", 1)
                    burst(8, "fca5a5")
                    deleteEntity()
                }
            }
        }
    }
    Entity {
        id("canSetter")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.05) {
                Actions {
                    spawn("can", 580, 330)
                    spawn("can", 650, 330)
                    spawn("can", 720, 330)
                    spawn("can", 790, 330)
                    spawn("can", 615, 410)
                    spawn("can", 755, 410)
                }
            }
        }
    }

    Entity {
        id("ball")
        startPosition(80, 60)
        color("1d4ed8")
        Type { circle(22) }
        Movement { speed(0) AutoMove { direction(0, 0) } }
    }

    Entity {
        id("padLow")
        startPosition(60, 840)
        color("22c55e")
        Type { rectangle(220, 90) }
        Events {
            OnTapped {
                Actions {
                    sound("pop")
                    updateBar("throws", -1)
                    setPosition("ball", 80, 60, 0)
                    setVelocity("ball", 520, 500)
                }
            }
        }
    }
    Entity {
        id("padMid")
        startPosition(340, 840)
        color("eab308")
        Type { rectangle(220, 90) }
        Events {
            OnTapped {
                Actions {
                    sound("pop")
                    updateBar("throws", -1)
                    setPosition("ball", 80, 60, 0)
                    setVelocity("ball", 520, 560)
                }
            }
        }
    }
    Entity {
        id("padHigh")
        startPosition(620, 840)
        color("ef4444")
        Type { rectangle(220, 90) }
        Events {
            OnTapped {
                Actions {
                    sound("pop")
                    updateBar("throws", -1)
                    setPosition("ball", 80, 60, 0)
                    setVelocity("ball", 440, 650)
                }
            }
        }
    }
}
