# A marble on an icy cliff: nothing slows it down. Each swipe gives it a push that way,
# a tap on the marble brakes it dead. Ledges catch it from above and bounce it from below.
Window {
    size(8, 12)
    background("dbeafe")
    music("bg_1")
    gravity(160)

    lives(5, "-- THE MARBLE SLIPPED AWAY --", "lose")

    Level {
        Entity {
            id("marble")
            startPosition(60, 30)
            color("1d4ed8")
            Type { circle(18) }
            Movement {
                speed(0)
                AutoMove { direction(1, 0) }
            }
            Events {
                OnSwipe("up") { Actions { sound("blip") applyImpulse(0, 320) } }
                OnSwipe("down") { Actions { sound("blip") applyImpulse(0, -200) } }
                OnSwipe("left") { Actions { sound("blip") applyImpulse(-160, 0) } }
                OnSwipe("right") { Actions { sound("blip") applyImpulse(160, 0) } }
                OnTapped("marble") { Actions { sound("click") setVelocity(0, 0) flash("ffffff", 0.2) } }
                OnCollision { Actions { sound("bump") bounceBack(0) } }
                OnCollision("ledge") { Actions { sound("bump") bounceBack(0) } }
                OnCollision("crack") {
                    Actions {
                        sound("glass")
                        burst(18, "93c5fd")
                        shake(5, 0.3)
                        updateBar("lives", -1)
                        setPosition("marble", 60, 30, 0)
                        setVelocity(0, 0)
                    }
                }
                OnCollision("cup") {
                    Actions {
                        sound("win2")
                        burst(24, "fbbf24")
                        nextLevel()
                    }
                }
            }
        }

        Entity { id("ledge") template(true) startPosition(0, 0) color("7dd3fc") oneWay(true) Type { rectangle(240, 20) } }
        Entity { id("crack") template(true) startPosition(0, 0) color("0f172a") Type { rectangle(60, 14) } }
        Entity { id("cup") startPosition(120, 800) color("fbbf24") Type { ring(26, 16) } }

        # the floor lays out the cliff: three rows of ledges with the way up on alternating sides
        Entity { id("floor") startPosition(0, 0) color("bfdbfe") oneWay(true) Type { rectangle(800, 20) }
            Events { OnElapsedTime { Actions {
                spawn("ledge", 0, 260) spawn("ledge", 240, 260) spawn("ledge", 480, 260)
                spawn("ledge", 80, 520) spawn("ledge", 320, 520) spawn("ledge", 560, 520)
                spawn("ledge", 0, 780) spawn("ledge", 240, 780) spawn("ledge", 480, 780)
                spawn("crack", 500, 534) spawn("crack", 400, 794) } } } }

        Entity { id("hint") startPosition(200, 120) color("1e3a8a") Type { text("swipe to push, tap the marble to brake", 22) } }
        Entity { id("lvl") startPosition(640, 1160) color("1e3a8a") Type { text("CLIFF 1/3", 24) } }
    }

    Level {
        Entity {
            id("marble")
            startPosition(60, 30)
            color("1d4ed8")
            Type { circle(18) }
            Movement {
                speed(0)
                AutoMove { direction(1, 0) }
            }
            Events {
                OnSwipe("up") { Actions { sound("blip") applyImpulse(0, 320) } }
                OnSwipe("down") { Actions { sound("blip") applyImpulse(0, -200) } }
                OnSwipe("left") { Actions { sound("blip") applyImpulse(-160, 0) } }
                OnSwipe("right") { Actions { sound("blip") applyImpulse(160, 0) } }
                OnTapped("marble") { Actions { sound("click") setVelocity(0, 0) flash("ffffff", 0.2) } }
                OnCollision { Actions { sound("bump") bounceBack(0) } }
                OnCollision("ledge") { Actions { sound("bump") bounceBack(0) } }
                OnCollision("post") { Actions { sound("bump") bounceBack(0) } }
                # the snowplough shoves a marble off at an angle
                OnCollision("plough") { Actions { sound("thud") bounceBack() shake(3, 0.2) } }
                OnCollision("crack") {
                    Actions {
                        sound("glass")
                        burst(18, "93c5fd")
                        shake(5, 0.3)
                        updateBar("lives", -1)
                        setPosition("marble", 60, 30, 0)
                        setVelocity(0, 0)
                    }
                }
                OnCollision("cup") {
                    Actions {
                        sound("win2")
                        burst(24, "fbbf24")
                        nextLevel()
                    }
                }
            }
        }

        Entity { id("ledge") template(true) startPosition(0, 0) color("7dd3fc") oneWay(true) Type { rectangle(240, 20) } }
        Entity { id("crack") template(true) startPosition(0, 0) color("0f172a") Type { rectangle(60, 14) } }
        Entity { id("post") template(true) startPosition(0, 0) color("60a5fa") Type { rectangle(20, 120) } }
        Entity { id("cup") startPosition(600, 980) color("fbbf24") Type { ring(26, 16) } }

        Entity { id("plough") startPosition(200, 20) color("ef4444") Type { rectangle(70, 40) }
            Movement { speed(1.6) Path { waypoint(200, 20) waypoint(720, 20) dwell(0.5) } } }

        Entity { id("floor") startPosition(0, 0) color("bfdbfe") oneWay(true) Type { rectangle(800, 20) }
            Events { OnElapsedTime { Actions {
                spawn("ledge", 0, 240) spawn("ledge", 240, 240) spawn("ledge", 400, 240)
                spawn("ledge", 160, 480) spawn("ledge", 400, 480) spawn("ledge", 560, 480)
                spawn("ledge", 0, 720) spawn("ledge", 240, 720) spawn("ledge", 400, 720)
                spawn("ledge", 160, 960) spawn("ledge", 400, 960) spawn("ledge", 560, 960)
                spawn("post", 520, 260) spawn("post", 300, 500)
                spawn("crack", 300, 254) spawn("crack", 600, 494) spawn("crack", 200, 734) } } } }

        Entity { id("lvl") startPosition(640, 1160) color("1e3a8a") Type { text("CLIFF 2/3", 24) } }
    }

    Level {
        Entity {
            id("marble")
            startPosition(60, 30)
            color("1d4ed8")
            Type { circle(18) }
            Movement {
                speed(0)
                AutoMove { direction(1, 0) }
            }
            Events {
                OnSwipe("up") { Actions { sound("blip") applyImpulse(0, 320) } }
                OnSwipe("down") { Actions { sound("blip") applyImpulse(0, -200) } }
                OnSwipe("left") { Actions { sound("blip") applyImpulse(-160, 0) } }
                OnSwipe("right") { Actions { sound("blip") applyImpulse(160, 0) } }
                OnTapped("marble") { Actions { sound("click") setVelocity(0, 0) flash("ffffff", 0.2) } }
                OnCollision { Actions { sound("bump") bounceBack(0) } }
                OnCollision("ledge") { Actions { sound("bump") bounceBack(0) } }
                OnCollision("post") { Actions { sound("bump") bounceBack(0) } }
                OnCollision("plough") { Actions { sound("thud") bounceBack() shake(3, 0.2) } }
                # a warm vent blows the marble up through the last gap
                OnCollision("vent") { Actions { sound("bubble") applyImpulse(0, 380) burst(8, "fecaca") } }
                OnCollision("crack") {
                    Actions {
                        sound("glass")
                        burst(18, "93c5fd")
                        shake(5, 0.3)
                        updateBar("lives", -1)
                        setPosition("marble", 60, 30, 0)
                        setVelocity(0, 0)
                    }
                }
                OnCollision("cup") {
                    Actions {
                        sound("win")
                        burst(40, "fbbf24")
                        shake(6, 0.5)
                        gameOver("-- THE MARBLE MADE THE SUMMIT --")
                    }
                }
            }
        }

        Entity { id("ledge") template(true) startPosition(0, 0) color("7dd3fc") oneWay(true) Type { rectangle(240, 20) } }
        Entity { id("crack") template(true) startPosition(0, 0) color("0f172a") Type { rectangle(60, 14) } }
        Entity { id("post") template(true) startPosition(0, 0) color("60a5fa") Type { rectangle(20, 120) } }
        Entity { id("cup") startPosition(60, 980) color("fbbf24") Type { ring(26, 16) } }
        Entity { id("vent") startPosition(660, 740) color("40f87171") Type { rectangle(120, 200) }
            Events { OnElapsedTime(0.3) { repeatCount(-1) Actions { burst(2, "fecaca") } } } }

        Entity { id("plough") startPosition(220, 260) color("ef4444") Type { rectangle(70, 40) }
            Movement { speed(1.8) Path { waypoint(220, 260) waypoint(720, 260) dwell(0.4) } } }

        Entity { id("floor") startPosition(0, 0) color("bfdbfe") oneWay(true) Type { rectangle(800, 20) }
            Events { OnElapsedTime { Actions {
                spawn("ledge", 160, 240) spawn("ledge", 400, 240) spawn("ledge", 560, 240)
                spawn("ledge", 0, 480) spawn("ledge", 240, 480) spawn("ledge", 400, 480)
                spawn("ledge", 160, 720) spawn("ledge", 400, 720) spawn("ledge", 560, 720)
                spawn("ledge", 0, 960) spawn("ledge", 240, 960) spawn("ledge", 400, 960)
                spawn("post", 260, 740)
                spawn("crack", 420, 254) spawn("crack", 660, 254) spawn("crack", 120, 494)
                spawn("crack", 460, 734) } } } }

        Entity { id("lvl") startPosition(640, 1160) color("1e3a8a") Type { text("CLIFF 3/3", 24) } }
    }
}
