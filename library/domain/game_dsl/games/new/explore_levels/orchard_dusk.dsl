Window {
    size(8, 9)
    worldSize(30, 9)
    cameraFollow("picker")
    background("3f6212")
    music("bg_nature")
    gravity(0)

    Bar("basket") {
        initial(0)
    }
    Bar("daylight") {
        initial(90)
        OnValue(25) { Actions { broadcast("dusk") } }
        OnValue(0) { Actions { sound("lose") gameOver("-- NIGHT FELL ON THE ORCHARD --") } }
    }
    winAt("crates", 4, "-- THE HARVEST IS IN --", "win")

    # Drawn first, so it tints the ground under everything once dusk comes.
    Entity {
        id("duskShade")
        startPosition(0, 0)
        color("552e1065")
        Type { rectangle(3000, 900) }
        Events {
            OnElapsedTime { Actions { setVisible(false) } }
            OnSignal("dusk") { Actions { setVisible(true) } }
        }
    }

    Entity {
        id("sun")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1) { repeatCount(-1) Actions { updateBar("daylight", -1) } }
        }
    }

    Entity {
        id("cart")
        startPosition(40, 380)
        color("a16207")
        Type { rectangle(110, 80) }
        Events {
            OnCollision("picker") {
                Actions {
                    whenBar("basket", "==", 5) {
                        Actions {
                            updateBar("basket", -5)
                            updateBar("crates", 1)
                            sound("coin")
                            burst(20, "ef4444")
                            flash("fde047", 0.3)
                            whenBar("crates", "==", 1) { Actions { setText("cartSign", "1 of 4 crates") } }
                            whenBar("crates", "==", 2) { Actions { setText("cartSign", "2 of 4 crates") } }
                            whenBar("crates", "==", 3) { Actions { setText("cartSign", "3 of 4 crates") } }
                        }
                    }
                }
            }
        }
    }
    Entity {
        id("cartSign")
        startPosition(30, 480)
        color("fef3c7")
        Type { text("CART: bring 5 apples", 26) }
        Events {
            OnSignal("dusk") { Actions { setText("cartSign", "Hurry, it is getting dark") setColor("fdba74") } }
        }
    }

    Entity {
        id("picker")
        startPosition(200, 400)
        color("fde68a")
        Type { rectangle(44, 54) }
        Movement { speed(6) Control { movementControl(0) } }
        Events {
            OnCollision("apple") {
                Actions {
                    whenBar("basket", "<", 5) {
                        Actions {
                            updateBar("basket", 1)
                            sound("pickup")
                            burst(6, "ef4444")
                            deleteCollider()
                        }
                    }
                }
            }
        }
    }

    Entity { id("pond") startPosition(1350, 300) color("0e7490") solid(true) Type { rectangle(260, 300) } }
    Entity { id("hedge1") startPosition(950, 0) color("365314") solid(true) Type { rectangle(40, 330) } }
    Entity { id("hedge2") startPosition(2000, 570) color("365314") solid(true) Type { rectangle(40, 330) } }

    Entity {
        id("apple")
        template(true)
        color("ef4444")
        Type { circle(14) }
        Events {
            OnElapsedTime(7) { Actions { setColor("92400e") } }
            OnElapsedTime(10) { Actions { burst(4, "92400e") deleteEntity() } }
        }
    }

    Entity { id("tree1") startPosition(320, 620) color("166534") Type { circle(80) }
        Events { OnElapsedTime(random(3, 5)) { repeatCount(-1) Actions { spawn("apple", random(260, 460), random(420, 580)) } } } }
    Entity { id("tree2") startPosition(760, 140) color("166534") Type { circle(80) }
        Events { OnElapsedTime(random(3, 5)) { repeatCount(-1) Actions { spawn("apple", random(700, 900), random(320, 460)) } } } }
    Entity { id("tree3") startPosition(1180, 640) color("166534") Type { circle(80) }
        Events { OnElapsedTime(random(3, 5)) { repeatCount(-1) Actions { spawn("apple", random(1100, 1320), random(420, 600)) } } } }
    Entity { id("tree4") startPosition(1760, 120) color("15803d") Type { circle(80) }
        Events { OnElapsedTime(random(2.5, 4)) { repeatCount(-1) Actions { spawn("apple", random(1700, 1900), random(300, 460)) } } } }
    Entity { id("tree5") startPosition(2250, 620) color("15803d") Type { circle(80) }
        Events { OnElapsedTime(random(2.5, 4)) { repeatCount(-1) Actions { spawn("apple", random(2180, 2400), random(420, 590)) } } } }
    Entity { id("tree6") startPosition(2700, 200) color("15803d") Type { circle(80) }
        Events { OnElapsedTime(random(2, 3.5)) { repeatCount(-1) Actions { spawn("apple", random(2620, 2860), random(340, 520)) } } } }

    # Crows steal one apple per swoop, and fly faster after dusk.
    Entity {
        id("crow1")
        color("ffffff")
        Type { text("🐦", 40) }
        facing("left")
        Movement { speed(2.5) Path { waypoint(500, 760) waypoint(1300, 200) waypoint(500, 200) } }
        Events {
            OnCollision("picker") { Actions { whenBar("basket", ">", 0) { Actions { updateBar("basket", -1) sound("hurt") flash("picker", "ef4444", 0.3) } } } }
            OnSignal("dusk") { Actions { updateSpeed(4) } }
        }
    }
    Entity {
        id("crow2")
        color("ffffff")
        Type { text("🐦", 40) }
        facing("left")
        Movement { speed(2.8) Path { waypoint(1700, 700) waypoint(2500, 300) dwell(0.6) } }
        Events {
            OnCollision("picker") { Actions { whenBar("basket", ">", 0) { Actions { updateBar("basket", -1) sound("hurt") flash("picker", "ef4444", 0.3) } } } }
            OnSignal("dusk") { Actions { updateSpeed(4.5) } }
        }
    }
    Entity {
        id("crow3")
        color("ffffff")
        Type { text("🐦", 40) }
        facing("left")
        Movement { speed(3) Path { waypoint(2950, 800) waypoint(2450, 100) waypoint(2950, 100) } }
        Events {
            OnCollision("picker") { Actions { whenBar("basket", ">", 0) { Actions { updateBar("basket", -1) sound("hurt") flash("picker", "ef4444", 0.3) } } } }
            OnSignal("dusk") { Actions { updateSpeed(4.5) } }
        }
    }
}
