# Firefly Jar: fireflies drift over the meadow and glow for a moment now and then.
# Tap one while it glows to catch it; grabbing a dark one scares the swarm. After midnight the glows get shorter.
Window {
    size(9, 9)
    background("05140f")
    music("bg_ambient")
    gravity(0)

    lives(3, "-- THE SWARM FLED --", "lose")
    Bar("jar") { initial(0) }
    Bar("night") { initial(1) }
    winAt("jar", 20, "-- A JAR FULL OF LIGHT --", "win2")

    Entity { id("meadow") startPosition(0, 0) color("14532d") Type { rectangle(900, 170) }
        Events {
            OnElapsedTime { Actions {
                spawn("fly", 120, 700) spawn("fly", 420, 780) spawn("fly", 720, 640)
                spawn("fly", 220, 380) spawn("fly", 560, 460) spawn("fly", 760, 300)
            } }
            OnSignal("midnight") { Actions { setColor("052e16") } }
        }
    }

    Entity {
        id("moon")
        startPosition(40, 780)
        color("fef9c3")
        Type { circle(36) }
        Events {
            OnElapsedTime { Actions { setPosition("moon", 790, 780, 55) } }
            Sequence {
                Phase(1, 25) { Actions { sound("bell") updateBar("night", 1) broadcast("midnight") } }
                Phase(1, 30) { Actions {
                    whenBar("jar", "<", 20) { Actions { sound("lose") shake(5, 0.4) gameOver("-- DAWN CAME TOO SOON --") } }
                } }
            }
            OnSignal("midnight") { Actions { setColor("fde68a") burst(10, "fef9c3") } }
        }
    }

    Entity { id("jarGlass") startPosition(400, 20) color("6094a3b8") Type { rectangle(100, 130) }
        Events { OnSignal("caught") { Actions { flash("fde047", 0.3) burst(6, "fef08a") } } } }

    Entity {
        id("fly")
        template(true)
        color("fde047")
        Type { circle(16) }
        var("glow", 0)
        Movement { speed(0.6) AutoMove { direction(1, 0) } }
        Events {
            OnElapsedTime { Actions { setOpacity(0.15) setVelocity(random(-70, 70), random(-70, 70)) } }
            OnCollision { Actions { bounceBack(0) } }
            OnElapsedTime(random(1.2, 3.4)) {
                repeatCount(-1)
                Actions { whenVar("glow", "==", 0) { Actions {
                    changeVar("glow", 1)
                    setOpacity(1)
                    whenBar("night", "==", 1) { Actions { delay(0.9) { Actions {
                        whenVar("glow", "==", 1) { Actions { changeVar("glow", -1) setOpacity(0.15) } } } } } }
                    whenBar("night", "==", 2) { Actions { delay(0.55) { Actions {
                        whenVar("glow", "==", 1) { Actions { changeVar("glow", -1) setOpacity(0.15) } } } } } }
                } } }
            }
            OnTapped { Actions {
                whenVar("glow", "==", 0) { Actions { sound("hurt") shake(4, 0.25) flash("ef4444", 0.2) updateBar("lives", -1) } }
                whenVar("glow", "==", 1) { Actions {
                    sound("sparkle") burst(12, "fde047") broadcast("caught") updateBar("jar", 1)
                    spawn("fly", random(60, 820), random(220, 840))
                    deleteEntity() } }
            } }
        }
    }
}
