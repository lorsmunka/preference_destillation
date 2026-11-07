# Pumpkin Patrol: flying saucers hover over the county fair, then dive to beam up the prize pumpkins.
# The farmer's corn popper fires on its own; shoot a saucer that is carrying a pumpkin and the pumpkin falls back.
Window {
    size(9, 12)
    background("1e1b4b")
    music("bg_3")
    gravity(0)

    winAt("downed", 15, "-- THE FAIR IS SAVED --", "win")
    Bar("pumpkins") {
        initial(4)
        OnValue(0) { Actions { sound("lose") shake(8, 0.6) gameOver("-- ALL PUMPKINS ABDUCTED --") } }
    }

    Wave(20, 3) { Actions { spawn("ufo", random(40, 770), 1080) } }
    Wave(30, 2.2) { Actions { spawn("ufo", random(40, 770), 1080) } }
    Wave(600, 1.6) {
        Actions {
            randomAction {
                Actions { spawn("ufo", random(40, 770), 1080) }
                Actions { spawn("ufo", random(40, 380), 1080) spawn("ufo", random(420, 770), 1000) }
            }
        }
    }

    Entity {
        id("field")
        startPosition(0, 0)
        color("365314")
        Type { rectangle(900, 110) }
        Events {
            OnElapsedTime {
                Actions {
                    spawn("pumpkin", 110, 110)
                    spawn("pumpkin", 310, 110)
                    spawn("pumpkin", 530, 110)
                    spawn("pumpkin", 730, 110)
                }
            }
        }
    }

    Entity {
        id("farmer")
        startPosition(415, 20)
        color("2563eb")
        Type { rectangle(70, 70) }
        Movement { speed(7) Control { movementControl(2) } }
        Events {
            OnElapsedTime(0.55) { repeatCount(-1) Actions { spawnHere("corn") } }
        }
    }

    Entity { id("corn") template(true) color("fde047") Type { circle(8) }
        Movement { speed(8) AutoMove { direction(0, 1) } }
        Events { OnCollision { Actions { deleteEntity() } } } }

    Entity { id("pumpkin") template(true) color("f97316") Type { circle(30) } }

    Entity { id("dropped") template(true) color("fb923c") Type { circle(30) }
        Movement { speed(2.5) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("field") { Actions {
                spawnHere("pumpkin") updateBar("pumpkins", 1) sound("coin") burst(12, "fb923c") deleteEntity() } }
        } }

    Entity {
        id("ufo")
        template(true)
        color("94a3b8")
        Type { diamond(100, 36) }
        var("hp", 2)
        var("dive", 0)
        var("loot", 0)
        Movement { speed(1.6) AutoMove { direction(1, 0) } }
        Events {
            OnElapsedTime {
                Actions { randomAction { Actions { setVelocity(160, 0) } Actions { setVelocity(-160, 0) } } }
            }
            OnElapsedTime(random(2.5, 6)) {
                Actions { changeVar("dive", 1) setVelocity(0, -170) setColor("4ade80") sound("buzz") }
            }
            OnCollision {
                Actions {
                    whenVar("dive", "==", 0) { Actions { bounceBack(0) } }
                    whenVar("dive", "==", 1) { Actions { deleteEntity() } }
                }
            }
            OnCollision("pumpkin") {
                Actions {
                    whenVar("dive", "==", 1) { Actions { whenVar("loot", "==", 0) { Actions {
                        deleteCollider()
                        changeVar("loot", 1)
                        updateBar("pumpkins", -1)
                        setColor("f97316")
                        setVelocity(0, 220)
                        sound("beep")
                        flash("ffffff", 0.3)
                    } } } }
                }
            }
            OnCollision("field") {
                Actions {
                    whenVar("dive", "==", 1) { Actions { whenVar("loot", "==", 0) { Actions { setVelocity(0, 220) setColor("94a3b8") } } } }
                }
            }
            OnCollision("corn") {
                Actions {
                    deleteCollider()
                    changeVar("hp", -1)
                    flash("ffffff", 0.1)
                    whenVar("hp", "<=", 0) {
                        Actions {
                            sound("pop")
                            burst(18, "a3e635")
                            shake(3, 0.2)
                            updateBar("downed", 1)
                            whenVar("loot", "==", 1) { Actions { spawnHere("dropped") } }
                            deleteEntity()
                        }
                    }
                }
            }
        }
    }
}
