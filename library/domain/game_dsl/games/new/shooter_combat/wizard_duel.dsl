Window {
    size(10, 8)
    background("312e81")
    music("bg_3")
    gravity(0)

    lives(3, "-- OUT-SPELLED --", "lose")
    Bar("mana") { initial(5) }
    Bar("rival") {
        initial(10)
        OnValue(5) { Actions { broadcast("rage") } }
        OnValue(0) { Actions { broadcast("defeated") } }
    }

    # Round one: the apprentice paces up and down and mixes plain hexes with homing wisps.
    Level {
        Entity {
            id("mage")
            startPosition(40, 360)
            color("a78bfa")
            Type { circle(34) }
            Movement { speed(6) Control { movementControl(1) } }
            Events {
                OnElapsedTime(1.2) {
                    repeatCount(-1)
                    Actions { whenBar("mana", "<", 5) { Actions { updateBar("mana", 1) } } }
                }
                OnSignal("cast") {
                    Actions {
                        whenBar("mana", ">", 0) { Actions { spawnHere("bolt") updateBar("mana", -1) sound("laser") } }
                    }
                }
                OnSignal("ward") {
                    Actions {
                        whenBar("mana", ">=", 3) { Actions { updateBar("mana", -3) spawn("ward", 170, 0) sound("sparkle") } }
                    }
                }
            }
        }

        Entity { id("boltRune") startPosition(380, 12) color("fde047") Type { star(5, 34, 15) }
            Events { OnTapped("boltRune") { Actions { broadcast("cast") flash("ffffff", 0.1) } } } }
        Entity { id("wardRune") startPosition(530, 12) color("60a5fa") Type { ring(34, 22) }
            Events { OnTapped("wardRune") { Actions { broadcast("ward") flash("ffffff", 0.1) } } } }

        Entity {
            id("apprentice")
            startPosition(900, 360)
            color("f472b6")
            Type { circle(40) }
            Movement {
                speed(1.8)
                Path { waypoint(900, 60) waypoint(900, 660) }
            }
            Events {
                Sequence {
                    Phase(3, 2.0) { Actions { sound("blip") spawnHere("hex") } }
                    Phase(2, 2.4) { Actions { sound("bubble") spawnHere("wisp") } }
                    Phase(-1, 1.6) {
                        Actions {
                            randomAction {
                                Actions { sound("blip") spawnHere("hex") spawnHere("hex") spawnHere("hex") }
                                Actions { sound("bubble") spawnHere("wisp") }
                                Actions { sound("blip") spawnHere("hex") delay(0.3) { Actions { spawnHere("hex") } } }
                            }
                        }
                    }
                }
                OnCollision("bolt") {
                    Actions { deleteCollider() updateBar("rival", -1) flash("ffffff", 0.15) burst(10, "fde047") sound("thud") }
                }
                OnSignal("rage") { Actions { updateSpeed(3.2) setColor("dc2626") shake(5, 0.4) sound("buzz") } }
                OnSignal("defeated") {
                    Actions {
                        sound("win2")
                        burst(30, "f472b6")
                        setOpacity(0.3)
                        delay(1.5) { Actions { nextLevel() } }
                    }
                }
            }
        }

        Entity { id("bolt") template(true) color("fde047") Type { rectangle(28, 10) }
            Movement { speed(8) AutoMove { direction(1, 0) } }
            Events { OnCollision { Actions { deleteEntity() } } } }

        Entity { id("ward") template(true) color("60a5fa") Type { rectangle(18, 800) }
            Events {
                OnElapsedTime(1.8) { Actions { setOpacity(0.4) } }
                OnElapsedTime(2.5) { Actions { deleteEntity() } }
            } }

        Entity { id("hex") template(true) color("f472b6") Type { diamond(26, 26) }
            Movement { speed(3.6) AutoMove { direction(-1, 0) } }
            Events {
                OnElapsedTime { Actions { setVelocity(-360, random(-110, 110)) } }
                OnCollision("mage") { Actions {
                    sound("hurt") updateBar("lives", -1) flash("mage", "ff0000", 0.3) shake(5, 0.3) deleteEntity() } }
                OnCollision("ward") { Actions { burst(6, "60a5fa") deleteEntity() } }
                OnCollision { Actions { deleteEntity() } }
            } }

        Entity { id("wisp") template(true) color("f0abfc") Type { circle(14) }
            Movement { speed(1.6) chase("mage") }
            Events {
                OnCollision("mage") { Actions {
                    sound("hurt") updateBar("lives", -1) flash("mage", "ff0000", 0.3) shake(5, 0.3) deleteEntity() } }
                OnCollision("bolt") { Actions { deleteCollider() sound("pop") burst(8, "f0abfc") deleteEntity() } }
                OnCollision("ward") { Actions { burst(6, "60a5fa") deleteEntity() } }
                OnElapsedTime(6) { Actions { burst(6, "f0abfc") deleteEntity() } }
            } }
    }

    # Round two: the archmage blinks around the right side and throws fans and wisp pairs.
    Level {
        Entity {
            id("mage")
            startPosition(40, 360)
            color("a78bfa")
            Type { circle(34) }
            Movement { speed(6) Control { movementControl(1) } }
            Events {
                OnElapsedTime(1.2) {
                    repeatCount(-1)
                    Actions { whenBar("mana", "<", 5) { Actions { updateBar("mana", 1) } } }
                }
                OnSignal("cast") {
                    Actions {
                        whenBar("mana", ">", 0) { Actions { spawnHere("bolt") updateBar("mana", -1) sound("laser") } }
                    }
                }
                OnSignal("ward") {
                    Actions {
                        whenBar("mana", ">=", 3) { Actions { updateBar("mana", -3) spawn("ward", 170, 0) sound("sparkle") } }
                    }
                }
            }
        }

        Entity { id("boltRune") startPosition(380, 12) color("fde047") Type { star(5, 34, 15) }
            Events { OnTapped("boltRune") { Actions { broadcast("cast") flash("ffffff", 0.1) } } } }
        Entity { id("wardRune") startPosition(530, 12) color("60a5fa") Type { ring(34, 22) }
            Events { OnTapped("wardRune") { Actions { broadcast("ward") flash("ffffff", 0.1) } } } }

        Entity {
            id("archmage")
            startPosition(880, 360)
            color("22d3ee")
            Type { star(6, 46, 24) }
            Events {
                OnElapsedTime {
                    Actions { updateBar("rival", 16) sound("bell") shake(4, 0.3) }
                }
                OnElapsedTime(3) {
                    repeatCount(-1)
                    Actions { burst(8, "22d3ee") setPosition("archmage", random(780, 900), random(40, 700), 0.25) }
                }
                Sequence {
                    Phase(3, 1.7) { Actions { sound("blip") spawnHere("hex") spawnHere("hex") spawnHere("hex") } }
                    Phase(-1, 1.2) {
                        Actions {
                            randomAction {
                                Actions { sound("bubble") spawnHere("wisp") delay(0.4) { Actions { spawnHere("wisp") } } }
                                Actions { sound("blip") spawnHere("hex") spawnHere("hex") spawnHere("hex") spawnHere("hex") }
                                Actions { sound("blip") spawn("hex", 940, random(40, 740)) spawn("hex", 940, random(40, 740)) }
                            }
                        }
                    }
                }
                OnCollision("bolt") {
                    Actions { deleteCollider() updateBar("rival", -1) flash("ffffff", 0.15) burst(10, "fde047") sound("thud") }
                }
                OnSignal("rage") { Actions { setColor("dc2626") setScale(1.2) shake(6, 0.5) sound("buzz") } }
                OnSignal("defeated") {
                    Actions {
                        sound("win")
                        shake(8, 0.6)
                        burst(40, "22d3ee")
                        gameOver("-- GRAND MAGUS --")
                    }
                }
            }
        }

        Entity { id("bolt") template(true) color("fde047") Type { rectangle(28, 10) }
            Movement { speed(8) AutoMove { direction(1, 0) } }
            Events { OnCollision { Actions { deleteEntity() } } } }

        Entity { id("ward") template(true) color("60a5fa") Type { rectangle(18, 800) }
            Events {
                OnElapsedTime(1.8) { Actions { setOpacity(0.4) } }
                OnElapsedTime(2.5) { Actions { deleteEntity() } }
            } }

        Entity { id("hex") template(true) color("f472b6") Type { diamond(26, 26) }
            Movement { speed(3.6) AutoMove { direction(-1, 0) } }
            Events {
                OnElapsedTime { Actions { setVelocity(-400, random(-130, 130)) } }
                OnCollision("mage") { Actions {
                    sound("hurt") updateBar("lives", -1) flash("mage", "ff0000", 0.3) shake(5, 0.3) deleteEntity() } }
                OnCollision("ward") { Actions { burst(6, "60a5fa") deleteEntity() } }
                OnCollision { Actions { deleteEntity() } }
            } }

        Entity { id("wisp") template(true) color("f0abfc") Type { circle(14) }
            Movement { speed(1.9) chase("mage") }
            Events {
                OnCollision("mage") { Actions {
                    sound("hurt") updateBar("lives", -1) flash("mage", "ff0000", 0.3) shake(5, 0.3) deleteEntity() } }
                OnCollision("bolt") { Actions { deleteCollider() sound("pop") burst(8, "f0abfc") deleteEntity() } }
                OnCollision("ward") { Actions { burst(6, "60a5fa") deleteEntity() } }
                OnElapsedTime(6) { Actions { burst(6, "f0abfc") deleteEntity() } }
            } }
    }
}
