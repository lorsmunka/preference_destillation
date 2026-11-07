Window {
    size(9, 12)
    background("0a0e27")
    music("bg_5")
    gravity(0)

    Bar("kills") {
        initial(0)
        OnValue(30) {
            Actions {
                sound("win")
                shake(8, 0.6)
                burst("ship", 40, "4ade80")
                gameOver("-- INVASION REPELLED --")
            }
        }
    }

    Entity { id("wallL") startPosition(0, 160) color("1e293b")
        Type { rectangle(8, 920) } }
    Entity { id("wallR") startPosition(892, 160) color("1e293b")
        Type { rectangle(8, 920) } }

    Entity { id("dangerLine") startPosition(0, 150) color("450a0a")
        Type { rectangle(900, 4) }
        Events { OnCollision("inv") {
            Actions {
                sound("lose")
                shake(8, 0.5)
                burst(20, "ef4444")
                gameOver("-- THEY LANDED --")
            } } } }

    Entity {
        id("ship")
        startPosition(420, 80)
        color("4ade80")
        Type { rectangle(80, 24) }
        Movement { speed(6.8) Control { movementControl(2) } }
        Events {
            OnElapsedTime(0.7) {
                repeatCount(-1)
                Actions { sound("laser") spawnHere("bullet") }
            }
            OnCollision("bomb") {
                Actions {
                    sound("lose")
                    shake(8, 0.5)
                    burst(20, "ef4444")
                    flash("ff0000", 0.3)
                    gameOver("-- SHIP DOWN --")
                }
            }
        }
    }

    Entity {
        id("bullet")
        template(true)
        startPosition(0, 0)
        color("fde68a")
        Type { rectangle(6, 18) }
        Movement { speed(10.2) AutoMove { direction(0, 1) } }
        Events {
            OnCollision("inv") {
                Actions {
                    sound("pop")
                    burst(10, "c084fc")
                    updateBar("kills", 1)
                    deleteCollider()
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("bomb")
        template(true)
        startPosition(0, 0)
        color("c084fc")
        Type { triangle(28, "down") }
        Movement { speed(3.4) AutoMove { direction(0, -1) } }
        Events {
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("inv")
        template(true)
        startPosition(0, 0)
        color("c084fc")
        Type { star(5, 24, 11) }
        Movement { speed(1.7) AutoMove { direction(1, 0) } }
        Events {
            OnCollision("wallL") { Actions {
                setVelocity("inv", 0, -170)
                delay(0.15) { Actions { setVelocity("inv", 170, 0) } } } }
            OnCollision("wallR") { Actions {
                setVelocity("inv", 0, -170)
                delay(0.15) { Actions { setVelocity("inv", -170, 0) } } } }
            OnElapsedTime(random(4, 9)) {
                repeatCount(-1)
                Actions { spawnHere("bomb") }
            }
        }
    }

    Entity {
        id("director")
        startPosition(10, 10)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.05) {
                repeatCount(1)
                Actions {
                    spawn("inv", 150, 1000) spawn("inv", 260, 1000)
                    spawn("inv", 370, 1000) spawn("inv", 480, 1000)
                    spawn("inv", 590, 1000) spawn("inv", 700, 1000)
                    spawn("inv", 150, 920)  spawn("inv", 260, 920)
                    spawn("inv", 370, 920)  spawn("inv", 480, 920)
                    spawn("inv", 590, 920)  spawn("inv", 700, 920)
                    spawn("inv", 150, 840)  spawn("inv", 260, 840)
                    spawn("inv", 370, 840)  spawn("inv", 480, 840)
                    spawn("inv", 590, 840)  spawn("inv", 700, 840)
                    spawn("inv", 150, 760)  spawn("inv", 260, 760)
                    spawn("inv", 370, 760)  spawn("inv", 480, 760)
                    spawn("inv", 590, 760)  spawn("inv", 700, 760)
                    spawn("inv", 150, 680)  spawn("inv", 260, 680)
                    spawn("inv", 370, 680)  spawn("inv", 480, 680)
                    spawn("inv", 590, 680)  spawn("inv", 700, 680)
                }
            }
        }
    }
}
