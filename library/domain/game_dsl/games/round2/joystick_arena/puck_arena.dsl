# Air hockey against a robot that chases the puck. Drag your striker, put the puck in the top goal five times.
Window {
    size(7, 12)
    background("082f49")
    music("bg_1")
    gravity(0)

    lives(3, "-- THE ROBOT WINS --", "lose")
    winAt("goals", 5, "-- YOU WIN THE MATCH --", "win")

    Entity {
        id("goalTop")
        startPosition(200, 1140)
        color("22c55e")
        Type { rectangle(300, 60) }
    }
    Entity {
        id("goalBottom")
        startPosition(200, 0)
        color("ef4444")
        Type { rectangle(300, 60) }
    }

    Entity {
        id("striker")
        startPosition(310, 200)
        color("38bdf8")
        Type { circle(40) }
        Movement { speed(6) Control { movementControl(0) } }
    }

    Entity {
        id("bot")
        startPosition(310, 900)
        color("f97316")
        Type { circle(40) }
        Movement { speed(2.4) chase("puck", 0.5) }
    }

    Entity {
        id("puck")
        startPosition(335, 560)
        color("f8fafc")
        Type { circle(22) }
        Movement { speed(4) AutoMove { direction(1, -1) } }
        Events {
            OnCollision("striker") {
                Actions {
                    sound("bump")
                    bounceBack()
                }
            }
            OnCollision("bot") {
                Actions {
                    sound("click")
                    bounceBack()
                }
            }
            OnCollision("goalTop") {
                Actions {
                    sound("coin")
                    updateBar("goals", 1)
                    burst(14, "22c55e")
                    setPosition("puck", 335, 560, 0)
                    setVelocity(-220, -260)
                }
            }
            OnCollision("goalBottom") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    shake(8, 0.4)
                    burst(14, "ef4444")
                    setPosition("puck", 335, 560, 0)
                    setVelocity(220, 260)
                }
            }
            OnCollision { Actions { bounceBack(0) } }
        }
    }
}
