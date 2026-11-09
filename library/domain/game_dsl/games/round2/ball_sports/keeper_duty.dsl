Window {
    size(8, 12)
    background("14532d")
    gravity(250)
    music("bg_3")

    Bar("saves") { initial(0) }
    Bar("goals") {
        initial(0)
        OnValue(3) { Actions { sound("lose") gameOver("3 GOALS CONCEDED") } }
    }
    winAt("saves", 12, "CLEAN SHEET HERO", "win")

    Entity { id("postLeft") startPosition(40, 80) color("f8fafc") solid(true)
        Type { rectangle(20, 200) } }
    Entity { id("postRight") startPosition(740, 80) color("f8fafc") solid(true)
        Type { rectangle(20, 200) } }
    Entity {
        id("line")
        startPosition(60, 0)
        color("00000000")
        Type { rectangle(680, 40) }
    }

    Entity {
        id("keeper")
        startPosition(340, 60)
        color("facc15")
        Type { rectangle(120, 70) }
        Movement { speed(8) Control { movementControl(2) } }
    }

    Entity {
        id("striker")
        startPosition(400, 1120)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(6, 2.2) { Actions { spawn("shot", random(100, 700), 1120) } }
                Phase(-1, 1.3) { Actions { spawn("shot", random(100, 700), 1120) } }
            }
        }
    }

    Entity {
        id("shot")
        template(true)
        startPosition(0, 0)
        color("ffffff")
        Type { circle(22) }
        Movement { speed(2) AutoMove { direction(0, -1) } }
        Events {
            OnElapsedTime(0.05) { Actions { applyImpulse(random(-160, 160), 0) } }
            OnCollision("keeper") {
                Actions {
                    sound("pickup")
                    updateBar("saves", 1)
                    burst(10, "facc15")
                    deleteEntity()
                }
            }
            OnCollision("line") {
                Actions {
                    sound("buzz")
                    updateBar("goals", 1)
                    shake(5, 0.3)
                    deleteEntity()
                }
            }
            OnCollision { Actions { bounceBack() } }
        }
    }
}
