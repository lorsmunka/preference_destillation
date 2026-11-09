Window {
    size(8, 12)
    background("78350f")
    music("bg_nature")
    winAt("harvest", 6, "PUMPKIN FEAST", "win")
    Bar("harvest") { initial(0) }
    Bar("crows") {
        initial(0)
        OnValue(3) { Actions { sound("lose") gameOver("CROWS ATE THE PATCH") } }
    }

    Entity {
        id("seedPumpkin")
        template(true)
        startPosition(0, 0)
        color("65a30d")
        Type { circle(30) }
        var("age", 0)
        Events {
            OnElapsedTime(2) { Actions { setColor("eab308") setScale(1.4) } }
            OnElapsedTime(4) { Actions { setColor("f97316") setScale(1.8) changeVar("age", 1) } }
            OnTapped { Actions {
                whenVar("age", ">=", 1) { Actions {
                    updateBar("harvest", 1)
                    sound("coin")
                    burst(10, "fb923c")
                    deleteEntity()
                } }
            } }
        }
    }
    Entity {
        id("crow")
        startPosition(40, 900)
        color("1e1b4b")
        Type { triangle(50, "down") }
        Movement {
            speed(2)
            Path {
                waypoint(40, 900)
                waypoint(700, 700)
                waypoint(40, 500)
                waypoint(700, 300)
            }
        }
        Events {
            OnTapped("crow") { Actions {
                setPosition("crow", 40, 900, 0)
                sound("bump")
            } }
            OnElapsedTime(7) {
                repeatCount(-1)
                Actions { updateBar("crows", 1) sound("hurt") shake(5, 0.3) }
            }
        }
    }
    Entity {
        id("plot1")
        startPosition(120, 250)
        color("451a03")
        Type { rectangle(160, 160) }
        Events { OnTapped("plot1") { Actions { spawn("seedPumpkin", 170, 300) sound("click") } } }
    }
    Entity {
        id("plot2")
        startPosition(320, 250)
        color("451a03")
        Type { rectangle(160, 160) }
        Events { OnTapped("plot2") { Actions { spawn("seedPumpkin", 370, 300) sound("click") } } }
    }
    Entity {
        id("plot3")
        startPosition(520, 250)
        color("451a03")
        Type { rectangle(160, 160) }
        Events { OnTapped("plot3") { Actions { spawn("seedPumpkin", 570, 300) sound("click") } } }
    }
    Entity {
        id("hint")
        startPosition(80, 1100)
        color("fef3c7")
        Type { text("Tap a plot to plant, tap orange pumpkins", 28) }
    }
}
