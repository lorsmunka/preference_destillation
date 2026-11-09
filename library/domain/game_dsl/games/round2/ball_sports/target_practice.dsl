Window {
    size(8, 12)
    background("1e293b")
    music("bg_6")

    Bar("hits") { initial(0) }
    Bar("shots") {
        initial(8)
        OnValue(0) { Actions { sound("lose") gameOver("OUT OF BALLS") } }
    }
    winAt("hits", 5, "ALL TARGETS DOWN", "win")

    Entity {
        id("redTarget")
        startPosition(100, 900)
        color("ef4444")
        Type { circle(45) }
        Movement { speed(2.5) Path { waypoint(80, 900) waypoint(620, 900) } }
        Events {
            OnCollision("ball") {
                Actions {
                    sound("glass")
                    updateBar("hits", 1)
                    burst(12, "ef4444")
                    setPosition("redTarget", 80, 900, 0)
                }
            }
        }
    }
    Entity {
        id("blueTarget")
        startPosition(600, 700)
        color("3b82f6")
        Type { rectangle(100, 60) }
        Movement { speed(1.8) Path { waypoint(650, 700) waypoint(100, 700) dwell(0.5) } }
        Events {
            OnCollision("ball") {
                Actions {
                    sound("glass")
                    updateBar("hits", 1)
                    burst(12, "3b82f6")
                    setPosition("blueTarget", 650, 700, 0)
                }
            }
        }
    }
    Entity {
        id("goldTarget")
        startPosition(380, 1050)
        color("fbbf24")
        Type { star(5, 40, 18) }
        Movement { speed(3.5) Path { waypoint(60, 1060) waypoint(700, 1060) } }
        Events {
            OnCollision("ball") {
                Actions {
                    sound("sparkle")
                    updateBar("hits", 2)
                    burst(16, "fbbf24")
                    setPosition("goldTarget", 60, 1060, 0)
                }
            }
        }
    }

    Entity {
        id("ball")
        startPosition(380, 60)
        color("f8fafc")
        Type { circle(20) }
        Movement { speed(0) AutoMove { direction(0, 0) } }
        Events {
            OnCollision {
                Actions {
                    sound("blip")
                    bounceBack()
                    delay(1.2) { Actions { setPosition("ball", 380, 60, 0) setVelocity(0, 0) } }
                }
            }
        }
    }

    Entity {
        id("launcher")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSwipe("up") {
                Actions {
                    sound("pop")
                    updateBar("shots", -1)
                    setVelocity("ball", 0, 700)
                    delay(1.6) { Actions { setPosition("ball", 380, 60, 0) setVelocity("ball", 0, 0) } }
                }
            }
            OnSwipe("left") {
                Actions {
                    sound("pop")
                    updateBar("shots", -1)
                    setVelocity("ball", -260, 680)
                    delay(1.6) { Actions { setPosition("ball", 380, 60, 0) setVelocity("ball", 0, 0) } }
                }
            }
            OnSwipe("right") {
                Actions {
                    sound("pop")
                    updateBar("shots", -1)
                    setVelocity("ball", 260, 680)
                    delay(1.6) { Actions { setPosition("ball", 380, 60, 0) setVelocity("ball", 0, 0) } }
                }
            }
        }
    }
}
