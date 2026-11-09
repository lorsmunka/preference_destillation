# Light four beacons. The ghost hunts you, but every lit beacon sends it after the wisp for a few seconds.
Window {
    size(10, 10)
    background("0f0a1e")
    music("bg_ambient")
    gravity(0)

    lives(3, "-- THE GHOST GOT YOU --", "lose")
    Bar("lit") {
        initial(0)
        OnValue(4) {
            Actions {
                broadcast("allLit")
            }
        }
    }
    joystick("scout", 80, 6.5)
    wallBox(0, 0, 1000, 1000, "3b0764", 12)

    Entity {
        id("scout")
        startPosition(480, 480)
        color("a5f3fc")
        Type { circle(22) }
        Movement { speed(0) Control { movementControl(0) } }
        Events {
            OnSignal("allLit") {
                Actions {
                    sound("win")
                    shake(6, 0.5)
                    burst(30, "fde047")
                    gameOver("-- THE NIGHT IS LIT --")
                }
            }
        }
    }

    Entity {
        id("beaconNorth")
        startPosition(470, 880)
        color("581c87")
        Type { triangle(60, "up") }
        var("unlit", 0)
        Events {
            OnCollision("scout") {
                Actions {
                    whenVar("unlit", "==", 0) {
                        Actions {
                            sound("bell")
                            setColor("fde047")
                            changeVar("unlit", 1)
                            updateBar("lit", 1)
                            broadcast("lit")
                        }
                    }
                }
            }
        }
    }
    Entity {
        id("beaconEast")
        startPosition(860, 470)
        color("581c87")
        Type { triangle(60, "left") }
        var("unlit", 0)
        Events {
            OnCollision("scout") {
                Actions {
                    whenVar("unlit", "==", 0) {
                        Actions {
                            sound("bell")
                            setColor("fde047")
                            changeVar("unlit", 1)
                            updateBar("lit", 1)
                            broadcast("lit")
                        }
                    }
                }
            }
        }
    }
    Entity {
        id("beaconWest")
        startPosition(80, 560)
        color("581c87")
        Type { triangle(60, "right") }
        var("unlit", 0)
        Events {
            OnCollision("scout") {
                Actions {
                    whenVar("unlit", "==", 0) {
                        Actions {
                            sound("bell")
                            setColor("fde047")
                            changeVar("unlit", 1)
                            updateBar("lit", 1)
                            broadcast("lit")
                        }
                    }
                }
            }
        }
    }
    Entity {
        id("beaconSouth")
        startPosition(560, 90)
        color("581c87")
        Type { triangle(60, "down") }
        var("unlit", 0)
        Events {
            OnCollision("scout") {
                Actions {
                    whenVar("unlit", "==", 0) {
                        Actions {
                            sound("bell")
                            setColor("fde047")
                            changeVar("unlit", 1)
                            updateBar("lit", 1)
                            broadcast("lit")
                        }
                    }
                }
            }
        }
    }

    Entity {
        id("wisp")
        startPosition(200, 200)
        color("c4b5fd")
        Type { star(4, 16, 6) }
        Movement {
            speed(3)
            Path {
                waypoint(200, 200)
                waypoint(800, 300)
                waypoint(750, 800)
                waypoint(250, 750)
            }
        }
    }

    Entity {
        id("ghost")
        startPosition(100, 880)
        color("e2e8f0")
        Type { circle(34) }
        Movement { speed(1.6) chase("scout") }
        Events {
            OnSignal("lit") {
                Actions {
                    sound("buzz")
                    flash("c4b5fd", 0.5)
                    retarget("wisp")
                    delay(3.5) {
                        Actions { retarget("scout") }
                    }
                }
            }
            OnCollision("scout") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("scout", "ef4444", 0.3)
                    shake(6, 0.3)
                    burst(12, "e2e8f0")
                    setPosition("ghost", 100, 880, 0)
                    setPosition("scout", 480, 480, 0)
                }
            }
        }
    }
}
