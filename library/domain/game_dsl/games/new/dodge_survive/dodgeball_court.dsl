# Dodgeball Court: three timed rounds against a team that gets bigger and faster.
# The centre line is solid, so you stay in your half; the balls fly straight over it.
Window {
    size(8, 12)
    background("b45309")
    music("bg_4")
    gravity(0)

    lives(4, "-- YOU'RE OUT! --", "lose")
    Bar("clock") {
        initial(20)
        OnValue(0) { Actions { broadcast("whistle") } }
    }

    Level {
        Entity {
            id("title1")
            startPosition(20, 1150)
            color("fef3c7")
            Type { text("ROUND 1 / 3", 30) }
        }

        Entity {
            id("ref")
            startPosition(20, 620)
            color("00000000")
            Type { circle(1) }
            Events {
                OnElapsedTime { Actions { ramp("clock", 20, 0, 20) } }
                OnSignal("whistle") {
                    Actions {
                        sound("bell")
                        burst("you", 20, "fef3c7")
                        nextLevel()
                    }
                }
            }
        }

        Entity {
            id("line")
            startPosition(0, 600)
            color("fef3c7")
            Type { rectangle(800, 16) }
            solid(true)
        }

        Entity {
            id("you")
            startPosition(376, 200)
            color("22d3ee")
            Type { circle(24) }
            Movement {
                speed(6)
                Control { movementControl(0) }
            }
        }

        Entity {
            id("thrower1")
            startPosition(0, 0)
            color("dc2626")
            Type { rectangle(60, 60) }
            Movement {
                speed(1.6)
                Path {
                    waypoint(60, 1000)
                    waypoint(330, 1000)
                }
            }
            Events {
                OnElapsedTime(random(1.6, 2.4)) {
                    repeatCount(-1)
                    Actions {
                        setColor("ffffff")
                        delay(0.35) { Actions { sound("blip") spawnHere("ball") setColor("dc2626") } }
                    }
                }
            }
        }

        Entity {
            id("thrower2")
            startPosition(0, 0)
            color("dc2626")
            Type { rectangle(60, 60) }
            Movement {
                speed(1.6)
                Path {
                    waypoint(680, 1000)
                    waypoint(410, 1000)
                }
            }
            Events {
                OnElapsedTime(random(1.6, 2.4)) {
                    repeatCount(-1)
                    Actions {
                        setColor("ffffff")
                        delay(0.35) { Actions { sound("blip") spawnHere("ball") setColor("dc2626") } }
                    }
                }
            }
        }

        Entity {
            id("ball")
            template(true)
            startPosition(0, 0)
            color("f43f5e")
            Type { circle(20) }
            Movement { speed(4) AutoMove { direction(0, -1) } }
            var("bounces", 0)
            Events {
                OnElapsedTime { Actions { setVelocity(random(-240, 240), random(-460, -380)) } }
                OnCollision("you") {
                    Actions {
                        sound("hurt")
                        updateBar("lives", -1)
                        flash("you", "ff0000", 0.3)
                        shake(6, 0.3)
                        burst(12, "f43f5e")
                        deleteEntity()
                    }
                }
                OnCollision {
                    Actions {
                        sound("bump")
                        bounceBack(0)
                        changeVar("bounces", 1)
                        whenVar("bounces", ">=", 3) { Actions { burst(6, "fda4af") deleteEntity() } }
                    }
                }
            }
        }
    }

    Level {
        Entity { id("title2") startPosition(20, 1150) color("fef3c7") Type { text("ROUND 2 / 3", 30) } }
        Entity { id("ref") startPosition(20, 620) color("00000000") Type { circle(1) }
            Events {
                OnElapsedTime { Actions { ramp("clock", 20, 0, 20) } }
                OnElapsedTime(9) { repeatCount(2) Actions { spawn("bottle", random(40, 740), random(60, 520)) } }
                OnSignal("whistle") { Actions { sound("bell") burst("you", 20, "fef3c7") nextLevel() } }
            } }
        Entity { id("line") startPosition(0, 600) color("fef3c7") Type { rectangle(800, 16) } solid(true) }
        Entity { id("you") startPosition(376, 200) color("22d3ee") Type { circle(24) }
            Movement { speed(6) Control { movementControl(0) } } }

        Entity { id("thrower1") startPosition(0, 0) color("9333ea") Type { rectangle(60, 60) }
            Movement { speed(2.2) Path { waypoint(40, 1020) waypoint(300, 1020) } }
            Events { OnElapsedTime(random(1.2, 2.0)) { repeatCount(-1) Actions {
                setColor("ffffff") delay(0.3) { Actions { sound("blip") spawnHere("ball") setColor("9333ea") } } } } } }
        Entity { id("thrower2") startPosition(0, 0) color("9333ea") Type { rectangle(60, 60) }
            Movement { speed(2.2) Path { waypoint(370, 900) waypoint(370, 1100) } }
            Events { OnElapsedTime(random(1.2, 2.0)) { repeatCount(-1) Actions {
                setColor("ffffff") delay(0.3) { Actions { sound("blip") spawnHere("ball") setColor("9333ea") } } } } } }
        Entity { id("thrower3") startPosition(0, 0) color("9333ea") Type { rectangle(60, 60) }
            Movement { speed(2.2) Path { waypoint(700, 1020) waypoint(440, 1020) } }
            Events { OnElapsedTime(random(1.2, 2.0)) { repeatCount(-1) Actions {
                setColor("ffffff") delay(0.3) { Actions { sound("blip") spawnHere("ball") setColor("9333ea") } } } } } }

        Entity { id("ball") template(true) startPosition(0, 0) color("f43f5e") Type { circle(20) }
            Movement { speed(4) AutoMove { direction(0, -1) } }
            var("bounces", 0)
            Events {
                OnElapsedTime { Actions { setVelocity(random(-280, 280), random(-540, -440)) } }
                OnCollision("you") { Actions { sound("hurt") updateBar("lives", -1) flash("you", "ff0000", 0.3)
                    shake(6, 0.3) burst(12, "f43f5e") deleteEntity() } }
                OnCollision { Actions { sound("bump") bounceBack(0) changeVar("bounces", 1)
                    whenVar("bounces", ">=", 3) { Actions { burst(6, "fda4af") deleteEntity() } } } }
            } }
        Entity { id("bottle") template(true) startPosition(0, 0) color("60a5fa") Type { rectangle(20, 40) }
            Events {
                OnCollision("you") { Actions { sound("pickup") burst(10, "60a5fa")
                    whenBar("lives", "<", 4) { Actions { updateBar("lives", 1) } } deleteEntity() } }
                OnElapsedTime(5) { Actions { deleteEntity() } }
            } }
    }

    Level {
        Entity { id("title3") startPosition(20, 1150) color("fef3c7") Type { text("FINAL ROUND", 30) } }
        Entity { id("ref") startPosition(20, 620) color("00000000") Type { circle(1) }
            Events {
                OnElapsedTime { Actions { ramp("clock", 20, 0, 20) } }
                OnElapsedTime(10) { Actions { sound("buzz") broadcast("hurry") } }
                OnElapsedTime(8) { repeatCount(2) Actions { spawn("bottle", random(40, 740), random(60, 520)) } }
                OnSignal("whistle") { Actions {
                    sound("win") shake(6, 0.5) burst("you", 40, "fbbf24") gameOver("-- CHAMPIONS --") } }
            } }
        Entity { id("line") startPosition(0, 600) color("fef3c7") Type { rectangle(800, 16) } solid(true) }
        Entity { id("you") startPosition(376, 200) color("22d3ee") Type { circle(24) }
            Movement { speed(6) Control { movementControl(0) } } }

        Entity { id("thrower1") startPosition(0, 0) color("0f172a") Type { rectangle(60, 60) }
            Movement { speed(2.4) Path { waypoint(40, 1000) waypoint(700, 1000) } }
            Events {
                OnElapsedTime(random(1.0, 1.8)) { repeatCount(-1) Actions {
                    setColor("ffffff") delay(0.3) { Actions { sound("blip") spawnHere("ball") setColor("0f172a") } } } }
                OnSignal("hurry") { Actions { updateSpeed(3.6) flash("f97316", 0.4) } }
            } }
        Entity { id("thrower2") startPosition(0, 0) color("0f172a") Type { rectangle(60, 60) }
            Movement { speed(2.4) Path { waypoint(700, 1100) waypoint(40, 1100) } }
            Events {
                OnElapsedTime(random(1.0, 1.8)) { repeatCount(-1) Actions {
                    setColor("ffffff") delay(0.3) { Actions { sound("blip") spawnHere("ball") setColor("0f172a") } } } }
                OnSignal("hurry") { Actions { updateSpeed(3.6) flash("f97316", 0.4) } }
            } }
        Entity { id("captain") startPosition(0, 0) color("facc15") Type { star(5, 36, 16) }
            Movement { speed(3) Path { waypoint(80, 820) waypoint(640, 820) dwell(0.5) } }
            Events { OnElapsedTime(random(2.0, 3.0)) { repeatCount(-1) Actions {
                setColor("ef4444") delay(0.5) { Actions { sound("laser") spawnHere("fastball") setColor("facc15") } } } } } }

        Entity { id("ball") template(true) startPosition(0, 0) color("f43f5e") Type { circle(20) }
            Movement { speed(4) AutoMove { direction(0, -1) } }
            var("bounces", 0)
            Events {
                OnElapsedTime { Actions { setVelocity(random(-300, 300), random(-560, -460)) } }
                OnCollision("you") { Actions { sound("hurt") updateBar("lives", -1) flash("you", "ff0000", 0.3)
                    shake(6, 0.3) burst(12, "f43f5e") deleteEntity() } }
                OnCollision { Actions { sound("bump") bounceBack(0) changeVar("bounces", 1)
                    whenVar("bounces", ">=", 3) { Actions { burst(6, "fda4af") deleteEntity() } } } }
            } }
        Entity { id("fastball") template(true) startPosition(0, 0) color("facc15") Type { circle(16) }
            Movement { speed(8) AutoMove { direction(0, -1) } }
            Events {
                OnCollision("you") { Actions { sound("hurt") updateBar("lives", -1) flash("you", "ff0000", 0.3)
                    shake(8, 0.35) burst(16, "facc15") deleteEntity() } }
                OnCollision { Actions { burst(8, "facc15") deleteEntity() } }
            } }
        Entity { id("bottle") template(true) startPosition(0, 0) color("60a5fa") Type { rectangle(20, 40) }
            Events {
                OnCollision("you") { Actions { sound("pickup") burst(10, "60a5fa")
                    whenBar("lives", "<", 4) { Actions { updateBar("lives", 1) } } deleteEntity() } }
                OnElapsedTime(5) { Actions { deleteEntity() } }
            } }
    }
}
