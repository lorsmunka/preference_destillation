Window {
    size(8, 12)
    background("020617")
    music("bg_6")
    gravity(380)

    Bar("lit") { initial(0) }
    winAt("lit", 20, "-- GRAND FINALE --", "win")
    Bar("drifted") {
        initial(0)
        OnValue(6) { Actions { sound("lose") shake(6, 0.5) gameOver("-- THE LANTERNS DRIFTED AWAY --") } }
    }

    # three launch tubes; each needs 0.8 s to reload
    Entity {
        id("tubeL")
        startPosition(100, 0)
        color("dc2626")
        Type { rectangle(60, 70) }
        var("ready", 1)
        Events {
            OnTapped("tubeL") {
                Actions {
                    whenVar("ready", "==", 1) {
                        Actions {
                            sound("laser")
                            spawn("rocket", 122, 70)
                            burst(8, "fbbf24")
                            changeVar("ready", -1)
                            setOpacity(0.4)
                            delay(0.8) { Actions { changeVar("ready", 1) setOpacity(1) } }
                        }
                    }
                }
            }
        }
    }
    Entity {
        id("tubeM")
        startPosition(370, 0)
        color("16a34a")
        Type { rectangle(60, 70) }
        var("ready", 1)
        Events {
            OnTapped("tubeM") {
                Actions {
                    whenVar("ready", "==", 1) {
                        Actions {
                            sound("laser")
                            spawn("rocket", 392, 70)
                            burst(8, "fbbf24")
                            changeVar("ready", -1)
                            setOpacity(0.4)
                            delay(0.8) { Actions { changeVar("ready", 1) setOpacity(1) } }
                        }
                    }
                }
            }
        }
    }
    Entity {
        id("tubeR")
        startPosition(640, 0)
        color("2563eb")
        Type { rectangle(60, 70) }
        var("ready", 1)
        Events {
            OnTapped("tubeR") {
                Actions {
                    whenVar("ready", "==", 1) {
                        Actions {
                            sound("laser")
                            spawn("rocket", 662, 70)
                            burst(8, "fbbf24")
                            changeVar("ready", -1)
                            setOpacity(0.4)
                            delay(0.8) { Actions { changeVar("ready", 1) setOpacity(1) } }
                        }
                    }
                }
            }
        }
    }

    # a rocket climbs, slowed by gravity, and bursts into embers after one second
    Entity {
        id("rocket")
        template(true)
        startPosition(0, 0)
        color("fbbf24")
        Type { rectangle(16, 34) }
        Movement { speed(9) AutoMove { direction(0, 1) } }
        Events {
            OnElapsedTime(0.08) { repeatCount(-1) Actions { burst(1, "fde68a") } }
            OnElapsedTime(1.0) {
                Actions {
                    sound("pop")
                    burst(26, "f472b6")
                    shake(2, 0.15)
                    spawnHere("emberL")
                    spawnHere("emberR")
                    spawnHere("emberU")
                    deleteEntity()
                }
            }
            OnCollision("lantern") {
                Actions {
                    burst(20, "fde047")
                    spawnHere("emberL")
                    spawnHere("emberR")
                    deleteEntity()
                }
            }
        }
    }

    Entity { id("emberL") template(true) startPosition(0, 0) color("f472b6") Type { circle(7) }
        Movement { speed(2.6) AutoMove { direction(-1, 0.6) } }
        Events { OnElapsedTime(0.9) { Actions { deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
    Entity { id("emberR") template(true) startPosition(0, 0) color("a78bfa") Type { circle(7) }
        Movement { speed(2.6) AutoMove { direction(1, 0.6) } }
        Events { OnElapsedTime(0.9) { Actions { deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }
    Entity { id("emberU") template(true) startPosition(0, 0) color("fde047") Type { circle(7) }
        Movement { speed(2.2) AutoMove { direction(0, 1) } }
        Events { OnElapsedTime(0.9) { Actions { deleteEntity() } } OnCollision { Actions { deleteEntity() } } } }

    # paper lanterns float in from the left; they ignore gravity
    Entity {
        id("lantern")
        template(true)
        startPosition(0, 0)
        color("f97316")
        gravity(0)
        Type { text("🏮", 40) }
        Movement { speed(0.9) AutoMove { direction(1, 0.15) } }
        Events {
            OnCollision("rocket") { Actions { sound("sparkle") updateBar("lit", 1) burst(16, "fde047") deleteEntity() } }
            OnCollision("emberL") { Actions { sound("sparkle") updateBar("lit", 1) burst(16, "fde047") deleteEntity() } }
            OnCollision("emberR") { Actions { sound("sparkle") updateBar("lit", 1) burst(16, "fde047") deleteEntity() } }
            OnCollision("emberU") { Actions { sound("sparkle") updateBar("lit", 1) burst(16, "fde047") deleteEntity() } }
            OnCollision { Actions { sound("buzz") updateBar("drifted", 1) deleteEntity() } }
        }
    }

    Entity {
        id("moon")
        startPosition(640, 1040)
        color("fef9c3")
        Type { circle(50) }
        Events {
            Sequence {
                Phase(8, 2.4) { Actions { spawn("lantern", 0, random(320, 720)) } }
                Phase(12, 1.8) { Actions { spawn("lantern", 0, random(320, 720)) } }
                Phase(-1, 1.3) { Actions { flash("fde68a", 0.2) spawn("lantern", 0, random(320, 720)) } }
            }
        }
    }
}
