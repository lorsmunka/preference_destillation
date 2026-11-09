# Lantern Caves: carry the lantern through three caves, grab the crystals, find the exit.
Window {
    size(8, 12)
    background("0b1020")
    music("bg_ambient")
    lives(3, "-- THE DARK WINS --", "lose")
    Bar("crystals") { initial(0) }

    Level {
        joystick("player", 80, 6)
        wallBox(0, 0, 800, 1200, "334155", 16)
        Entity {
            id("player")
            startPosition(60, 200)
            color("fde68a")
            Type { circle(26) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnCollision("crystal1") { Actions { sound("coin") updateBar("crystals", 1) deleteCollider() } }
                OnCollision("crystal2") { Actions { sound("coin") updateBar("crystals", 1) deleteCollider() } }
                OnCollision("exit") { Actions { sound("win2") burst(14, "4ade80") nextLevel() } }
            }
        }
        Entity { id("crystal1") startPosition(380, 700) color("38bdf8") Type { diamond(36, 44) } }
        Entity { id("crystal2") startPosition(120, 980) color("38bdf8") Type { diamond(36, 44) } }
        Entity { id("rock1") startPosition(200, 400) color("1e293b") solid(true) Type { rectangle(400, 30) } }
        Entity { id("rock2") startPosition(300, 820) color("1e293b") solid(true) Type { rectangle(440, 30) } }
        Entity { id("exit") startPosition(680, 1060) color("4ade80") Type { star(5, 40, 18) } }
        Entity {
            id("bat1")
            startPosition(100, 560)
            color("a855f7")
            Type { triangle(40, "down") }
            Movement { speed(3) Path { waypoint(100, 560) waypoint(700, 560) } }
            Events { OnCollision("player") { Actions { sound("hurt") updateBar("lives", -1) setPosition("player", 60, 200, 0) } } }
        }
    }

    Level {
        joystick("player", 80, 6)
        wallBox(0, 0, 800, 1200, "334155", 16)
        Entity {
            id("player")
            startPosition(60, 200)
            color("fde68a")
            Type { circle(26) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnCollision("crystal1") { Actions { sound("coin") updateBar("crystals", 1) deleteCollider() } }
                OnCollision("crystal2") { Actions { sound("coin") updateBar("crystals", 1) deleteCollider() } }
                OnCollision("crystal3") { Actions { sound("coin") updateBar("crystals", 1) deleteCollider() } }
                OnCollision("exit") { Actions { sound("win2") burst(14, "4ade80") nextLevel() } }
            }
        }
        Entity { id("crystal1") startPosition(700, 300) color("38bdf8") Type { diamond(36, 44) } }
        Entity { id("crystal2") startPosition(100, 700) color("38bdf8") Type { diamond(36, 44) } }
        Entity { id("crystal3") startPosition(640, 1000) color("38bdf8") Type { diamond(36, 44) } }
        Entity { id("rock1") startPosition(0, 460) color("1e293b") solid(true) Type { rectangle(560, 30) } }
        Entity { id("rock2") startPosition(240, 860) color("1e293b") solid(true) Type { rectangle(560, 30) } }
        Entity { id("exit") startPosition(80, 1060) color("4ade80") Type { star(5, 40, 18) } }
        Entity {
            id("bat1")
            startPosition(100, 360)
            color("a855f7")
            Type { triangle(40, "down") }
            Movement { speed(4) Path { waypoint(100, 360) waypoint(700, 360) } }
            Events { OnCollision("player") { Actions { sound("hurt") updateBar("lives", -1) setPosition("player", 60, 200, 0) } } }
        }
        Entity {
            id("bat2")
            startPosition(700, 640)
            color("c084fc")
            Type { triangle(40, "down") }
            Movement { speed(3) Path { waypoint(700, 640) waypoint(100, 640) dwell(0.5) } }
            Events { OnCollision("player") { Actions { sound("hurt") updateBar("lives", -1) setPosition("player", 60, 200, 0) } } }
        }
    }

    Level {
        joystick("player", 80, 6)
        wallBox(0, 0, 800, 1200, "334155", 16)
        Entity {
            id("player")
            startPosition(60, 200)
            color("fde68a")
            Type { circle(26) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnCollision("crystal1") { Actions { sound("coin") updateBar("crystals", 1) deleteCollider() } }
                OnCollision("crystal2") { Actions { sound("coin") updateBar("crystals", 1) deleteCollider() } }
                OnCollision("heart") {
                    Actions {
                        sound("win")
                        gameOver("-- YOU FOUND THE HEART OF THE CAVE --")
                    }
                }
            }
        }
        Entity { id("crystal1") startPosition(380, 520) color("38bdf8") Type { diamond(36, 44) } }
        Entity { id("crystal2") startPosition(680, 900) color("38bdf8") Type { diamond(36, 44) } }
        Entity { id("rock1") startPosition(160, 360) color("1e293b") solid(true) Type { rectangle(640, 30) } }
        Entity { id("rock2") startPosition(0, 700) color("1e293b") solid(true) Type { rectangle(600, 30) } }
        Entity { id("rock3") startPosition(200, 1000) color("1e293b") solid(true) Type { rectangle(600, 30) } }
        Entity {
            id("heart")
            startPosition(80, 1090)
            color("f43f5e")
            Type { star(6, 36, 16) }
            Events { OnElapsedTime(1.0) { repeatCount(-1) Actions { burst(4, "f43f5e") } } }
        }
        Entity {
            id("bat1")
            startPosition(100, 500)
            color("a855f7")
            Type { triangle(40, "down") }
            Movement { speed(5) Path { waypoint(100, 500) waypoint(700, 500) } }
            Events { OnCollision("player") { Actions { sound("hurt") updateBar("lives", -1) setPosition("player", 60, 200, 0) } } }
        }
        Entity {
            id("bat2")
            startPosition(700, 860)
            color("c084fc")
            Type { triangle(40, "down") }
            Movement { speed(4) Path { waypoint(700, 860) waypoint(100, 860) } }
            Events { OnCollision("player") { Actions { sound("hurt") updateBar("lives", -1) setPosition("player", 60, 200, 0) } } }
        }
        Entity {
            id("wisp")
            startPosition(700, 200)
            color("f472b6")
            Type { circle(20) }
            Movement { speed(1.6) chase("player") }
            Events { OnCollision("player") { Actions { sound("hurt") updateBar("lives", -1) setPosition("player", 60, 200, 0) setPosition("wisp", 700, 200, 0) } } }
        }
    }
}
