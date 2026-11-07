Window {
    size(7, 10)
    background("1e1b4b")
    music("bg_2")
    gravity(0)

    lives(3, "-- THE VILLAGE BURNED --", "lose")
    Bar("dragon") {
        initial(12)
        OnValue(6) {
            Actions {
                broadcast("fury")
            }
        }
        OnValue(0) {
            Actions {
                sound("win")
                shake(9, 0.7)
                burst("dragon", 40, "4ade80")
                gameOver("-- THE DRAGON FLEES --")
            }
        }
    }

    Entity {
        id("village")
        startPosition(0, 0)
        color("78350f")
        Type { rectangle(700, 40) }
    }

    Entity {
        id("knight")
        startPosition(290, 70)
        color("cbd5e1")
        Type { rectangle(120, 24) }
        Movement {
            speed(7)
            Control { movementControl(2) }
        }
    }

    Entity {
        id("dragon")
        startPosition(40, 860)
        color("16a34a")
        Type { star(7, 50, 26) }
        Movement {
            speed(1.6)
            Path {
                waypoint(40, 860)
                waypoint(560, 860)
            }
        }
        Events {
            Sequence {
                Phase(5, 2.2) {
                    Actions {
                        sound("blip")
                        spawnHere("fireball")
                    }
                }
                Phase(6, 1.8) {
                    Actions {
                        sound("blip")
                        spawnHere("fireball")
                        delay(0.4) { Actions { spawnHere("fireball") } }
                    }
                }
                Phase(-1, 1.3) {
                    Actions {
                        setColor("dc2626")
                        delay(0.3) {
                            Actions {
                                sound("blip")
                                spawnHere("fireball")
                                spawnHere("fireball")
                                setColor("16a34a")
                            }
                        }
                    }
                }
            }
            OnSignal("fury") {
                Actions {
                    sound("buzz")
                    shake(6, 0.4)
                    flash("f97316", 0.6)
                    updateSpeed(2.8)
                }
            }
        }
    }

    Entity {
        id("fireball")
        template(true)
        color("f97316")
        Type { circle(16) }
        var("bounced", 0)
        Movement {
            speed(3)
            AutoMove { direction(0, -1) }
        }
        Events {
            OnElapsedTime {
                Actions { setVelocity(random(-90, 90), -300) }
            }
            OnCollision("knight") {
                Actions {
                    whenVar("bounced", "==", 0) {
                        Actions {
                            changeVar("bounced", 1)
                            bounceBack()
                            updateSpeed(5)
                            setColor("7dd3fc")
                            sound("clang")
                            burst(8, "fde68a")
                        }
                    }
                }
            }
            OnCollision("dragon") {
                Actions {
                    whenVar("bounced", "==", 1) {
                        Actions {
                            updateBar("dragon", -1)
                            flash("dragon", "ffffff", 0.2)
                            shake(4, 0.2)
                            sound("thud")
                            burst(14, "7dd3fc")
                            deleteEntity()
                        }
                    }
                }
            }
            OnCollision("village") {
                Actions {
                    updateBar("lives", -1)
                    flash("village", "ef4444", 0.3)
                    shake(6, 0.3)
                    sound("hurt")
                    burst(16, "f97316")
                    deleteEntity()
                }
            }
            OnCollision {
                Actions { deleteEntity() }
            }
        }
    }
}
