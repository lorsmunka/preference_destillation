Window {
    size(9, 14)
    background("0c4a6e")
    music("bg_nature")
    Bar("water") {
        initial(5)
        OnValue(0) { Actions { sound("lose") gameOver("THE TANK TURNED GREEN") } }
        OnValue(10) { Actions { broadcast("sparkling") } }
    }
    Bar("fishHappy") {
        initial(0)
        OnValue(6) { Actions { sound("win") gameOver("A PERFECT AQUARIUM") } }
    }

    Entity {
        id("fish")
        startPosition(100, 700)
        color("fb923c")
        Type { diamond(90, 50) }
        facing("right")
        Movement {
            speed(1.5)
            Path {
                waypoint(100, 700)
                waypoint(700, 900)
                waypoint(200, 1000)
                waypoint(750, 600)
            }
        }
        Events {
            OnCollision("flake") {
                Actions {
                    updateBar("fishHappy", 1)
                    sound("pickup")
                    burst(6, "fde68a")
                    deleteCollider()
                }
            }
            OnSignal("sparkling") { Actions { flash("fish", "ffffff", 0.5) } }
        }
    }
    Entity {
        id("flake")
        template(true)
        startPosition(0, 0)
        color("fde68a")
        Type { circle(12) }
        Movement { speed(1) AutoMove { direction(0, -1) } }
        Events {
            OnElapsedTime(5) { Actions { deleteEntity() } }
        }
    }
    Entity {
        id("dirt")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(3) {
                repeatCount(-1)
                Actions { updateBar("water", -1) setText("waterText", "Water is getting dirty") }
            }
        }
    }
    Entity {
        id("waterText")
        startPosition(60, 1300)
        color("bae6fd")
        Type { text("Feed the fish, clean the water", 36) }
    }
    Entity {
        id("feedButton")
        startPosition(80, 120)
        color("f59e0b")
        Type { rectangle(340, 130) }
        Events {
            OnTapped("feedButton") {
                Actions {
                    spawn("flake", random(150, 700), 1250)
                    spawn("flake", random(150, 700), 1200)
                    sound("blip")
                }
            }
        }
    }
    Entity {
        id("cleanButton")
        startPosition(480, 120)
        color("14b8a6")
        Type { rectangle(340, 130) }
        Events {
            OnTapped("cleanButton") {
                Actions {
                    updateBar("water", 1)
                    setText("waterText", "Fresh water")
                    sound("bubble")
                    burst(8, "e0f2fe")
                }
            }
        }
    }
    Entity {
        id("feedLabel")
        startPosition(170, 160)
        color("422006")
        Type { text("FEED", 44) }
    }
    Entity {
        id("cleanLabel")
        startPosition(560, 160)
        color("042f2e")
        Type { text("CLEAN", 44) }
    }
}
