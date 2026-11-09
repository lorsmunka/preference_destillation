# Two arenas. A knight with a loose blade that trails behind. Sweep the blade through goblins in the camp, then take down the ogre.
Window {
    size(10, 8)
    background("1c1917")
    music("bg_4")
    gravity(0)

    lives(4, "-- THE KNIGHT HAS FALLEN --", "lose")
    Bar("goblins") {
        initial(0)
        OnValue(10) {
            Actions { nextLevel() }
        }
    }

    Level {
        joystick("knight", 80, 6)
        wallBox(0, 0, 1000, 800, "57534e", 14)

        Entity {
            id("knight")
            startPosition(480, 360)
            color("e2e8f0")
            Type { circle(26) }
            Movement { speed(0) Control { movementControl(0) } }
        }

        Entity {
            id("blade")
            startPosition(560, 360)
            color("fde047")
            Type { diamond(22, 70) }
            Movement { speed(9) chase("knight", 0.9) }
            Events {
                OnCollision("goblin") {
                    Actions {
                        sound("clang")
                        burst(8, "fde047")
                        updateBar("goblins", 1)
                        deleteCollider()
                    }
                }
            }
        }

        Entity {
            id("goblin")
            template(true)
            startPosition(0, 0)
            color("65a30d")
            Type { circle(22) }
            Movement { speed(1.5) chase("knight") }
            Events {
                OnCollision("knight") {
                    Actions {
                        sound("hurt")
                        updateBar("lives", -1)
                        flash("knight", "ef4444", 0.25)
                        shake(5, 0.3)
                        burst(8, "65a30d")
                        deleteEntity()
                    }
                }
            }
        }

        Entity {
            id("campWest")
            startPosition(40, 700)
            color("78350f")
            Type { triangle(50, "up") }
            Events {
                OnElapsedTime(2.4) {
                    repeatCount(-1)
                    Actions { sound("pop") spawnHere("goblin") }
                }
            }
        }

        Entity {
            id("campEast")
            startPosition(900, 700)
            color("78350f")
            Type { triangle(50, "up") }
            Events {
                OnElapsedTime(3.1) {
                    repeatCount(-1)
                    Actions { sound("pop") spawnHere("goblin") }
                }
            }
        }
    }

    Level {
        joystick("knight", 80, 6)
        wallBox(0, 0, 1000, 800, "44403c", 14)

        Entity {
            id("knight")
            startPosition(480, 120)
            color("e2e8f0")
            Type { circle(26) }
            Movement { speed(0) Control { movementControl(0) } }
        }

        Entity {
            id("blade")
            startPosition(560, 120)
            color("fde047")
            Type { diamond(22, 70) }
            Movement { speed(9) chase("knight", 0.9) }
            Events {
                OnCollision("goblin") {
                    Actions {
                        sound("clang")
                        burst(8, "fde047")
                        deleteCollider()
                    }
                }
                OnCollision("ogre") {
                    Actions {
                        sound("thud")
                        flash("ogre", "ffffff", 0.2)
                    }
                }
            }
        }

        Entity {
            id("goblin")
            template(true)
            startPosition(0, 0)
            color("84cc16")
            Type { circle(20) }
            Movement { speed(1.9) chase("knight") }
            Events {
                OnCollision("knight") {
                    Actions {
                        sound("hurt")
                        updateBar("lives", -1)
                        flash("knight", "ef4444", 0.25)
                        shake(5, 0.3)
                        burst(8, "84cc16")
                        deleteEntity()
                    }
                }
            }
        }

        Entity {
            id("banner")
            startPosition(130, 600)
            color("dc2626")
            Type { rectangle(26, 90) }
        }

        Entity {
            id("ogre")
            startPosition(780, 560)
            color("b45309")
            Type { polygon(6, 60) }
            var("hp", 8)
            Movement { speed(1.3) chase("knight") }
            Events {
                OnCollision("knight") {
                    Actions {
                        sound("lose")
                        updateBar("lives", -1)
                        flash("knight", "ef4444", 0.3)
                        shake(8, 0.4)
                        setPosition("knight", 480, 120, 0)
                    }
                }
                OnCollision("blade") {
                    Actions {
                        changeVar("hp", -1)
                        whenVar("hp", "<=", 4) { Actions { setColor("dc2626") } }
                        whenVar("hp", "<=", 0) {
                            Actions {
                                sound("win")
                                shake(10, 0.7)
                                burst(30, "f59e0b")
                                gameOver("-- THE OGRE FALLS --")
                            }
                        }
                    }
                }
                OnCollision("banner") {
                    Actions {
                        sound("thud")
                        shake(6, 0.4)
                        updateSpeed(0.3)
                        delay(2.0) {
                            Actions {
                                updateSpeed(1.3)
                                retarget("knight")
                            }
                        }
                    }
                }
                OnElapsedTime(9.0) {
                    repeatCount(-1)
                    Actions {
                        sound("buzz")
                        retarget("banner")
                    }
                }
            }
        }

        Entity {
            id("horn")
            startPosition(480, 740)
            color("00000000")
            Type { circle(1) }
            Events {
                OnElapsedTime(4.0) {
                    repeatCount(-1)
                    Actions {
                        sound("laser")
                        spawn("goblin", 40, 700)
                        spawn("goblin", 900, 100)
                    }
                }
            }
        }
    }
}
