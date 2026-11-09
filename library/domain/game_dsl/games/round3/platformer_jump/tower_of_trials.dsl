Window {
    size(6, 12)
    background("1e1b4b")
    music("bg_4")
    gravity(720)
    lives(4, "THE TRIALS ARE OVER")

    Level {
        Entity {
            id("hero")
            startPosition(60, 90)
            color("4ade80")
            Type { rectangle(34, 46) }
            Movement { speed(6) Control { movementControl(2) } }
            Events {
                OnCollision("door") { Actions {
                    sound("win2")
                    nextLevel()
                } }
            }
        }
        Entity { id("floor") startPosition(0, 0) color("475569")
            Type { rectangle(600, 70) } solid(true) }
        Entity { id("stepA") startPosition(200, 220) color("818cf8")
            Type { rectangle(180, 16) } oneWay(true) }
        Entity { id("stepB") startPosition(380, 360) color("818cf8")
            Type { rectangle(180, 16) } oneWay(true) }
        Entity { id("stepC") startPosition(160, 500) color("818cf8")
            Type { rectangle(180, 16) } oneWay(true) }
        Entity { id("door") startPosition(220, 516) color("fbbf24")
            Type { rectangle(50, 70) } }
        Entity { id("jumpButton") startPosition(380, 10) color("3b82f6")
            Type { rectangle(200, 50) }
            Events { OnTapped("jumpButton") { Actions {
                sound("jump")
                applyImpulse("hero", 0, 540)
            } } } }
        Entity { id("label") startPosition(300, 700) color("c7d2fe")
            Type { text("Trial 1: climb to the door", 26) } }
    }

    Level {
        Entity {
            id("hero")
            startPosition(60, 90)
            color("4ade80")
            Type { rectangle(34, 46) }
            Movement { speed(6) Control { movementControl(2) } }
            Events {
                OnCollision("spike1") { Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    setPosition("hero", 60, 90, 0)
                } }
                OnCollision("spike2") { Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    setPosition("hero", 60, 90, 0)
                } }
                OnCollision("spring") { Actions {
                    sound("blip")
                    applyImpulse(0, 800)
                } }
                OnCollision("door") { Actions {
                    sound("win2")
                    nextLevel()
                } }
            }
        }
        Entity { id("floor") startPosition(0, 0) color("475569")
            Type { rectangle(600, 70) } solid(true) }
        Entity { id("spike1") startPosition(220, 70) color("ef4444")
            Type { triangle(34, "up") } }
        Entity { id("spike2") startPosition(300, 70) color("ef4444")
            Type { triangle(34, "up") } }
        Entity { id("spring") startPosition(430, 70) color("22d3ee")
            Type { rectangle(60, 14) } }
        Entity { id("ledge") startPosition(300, 400) color("818cf8")
            Type { rectangle(240, 16) } oneWay(true) }
        Entity { id("ledge2") startPosition(40, 540) color("818cf8")
            Type { rectangle(240, 16) } oneWay(true) }
        Entity { id("door") startPosition(100, 556) color("fbbf24")
            Type { rectangle(50, 70) } }
        Entity { id("jumpButton") startPosition(380, 10) color("3b82f6")
            Type { rectangle(200, 50) }
            Events { OnTapped("jumpButton") { Actions {
                sound("jump")
                applyImpulse("hero", 0, 540)
            } } } }
    }

    Level {
        Entity {
            id("hero")
            startPosition(60, 90)
            color("4ade80")
            Type { rectangle(34, 46) }
            Movement { speed(6) Control { movementControl(2) } }
            Events {
                OnCollision("wisp") { Actions {
                    sound("hurt")
                    shake(5, 0.3)
                    updateBar("lives", -1)
                    setPosition("hero", 60, 90, 0)
                } }
            }
        }
        Entity { id("floor") startPosition(0, 0) color("475569")
            Type { rectangle(600, 70) } solid(true) }
        Entity { id("mover1") startPosition(150, 230) color("a5b4fc")
            Type { rectangle(140, 16) } oneWay(true)
            Movement { speed(1.2) Path {
                waypoint(80, 230)
                waypoint(420, 230)
            } } }
        Entity { id("mover2") startPosition(400, 400) color("a5b4fc")
            Type { rectangle(140, 16) } oneWay(true)
            Movement { speed(1.5) Path {
                waypoint(60, 400)
                waypoint(440, 400)
                dwell(0.5)
            } } }
        Entity { id("rest") startPosition(220, 560) color("818cf8")
            Type { rectangle(160, 16) } oneWay(true) }
        Entity { id("wisp") startPosition(100, 780) color("f0abfc")
            Type { circle(18) }
            Movement { speed(2) Path {
                waypoint(60, 780)
                waypoint(520, 800)
            } } }
        Entity { id("mid") startPosition(330, 700) color("818cf8")
            Type { rectangle(160, 16) } oneWay(true) }
        Entity { id("top") startPosition(80, 860) color("818cf8")
            Type { rectangle(200, 16) } oneWay(true) }
        goal("crown", 120, 880, 70, 50, "TOWER CONQUERED")
        Entity { id("jumpButton") startPosition(380, 10) color("3b82f6")
            Type { rectangle(200, 50) }
            Events { OnTapped("jumpButton") { Actions {
                sound("jump")
                applyImpulse("hero", 0, 560)
            } } } }
    }
}
