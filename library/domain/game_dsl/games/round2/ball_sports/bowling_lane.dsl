# Roll the ball with a swipe and knock over all ten pins. You get three rolls.
Window {
    size(6, 12)
    background("1c1917")
    music("bg_ambient")

    Bar("pins") { initial(0) }
    Bar("rolls") {
        initial(3)
        OnValue(0) {
            Actions {
                whenBar("pins", "<", 10) { Actions { sound("lose") gameOver("PINS LEFT STANDING") } }
            }
        }
    }
    winAt("pins", 10, "STRIKE!", "win")

    Entity { id("gutterLeft") startPosition(60, 0) color("78350f") solid(true)
        Type { rectangle(30, 1200) } }
    Entity { id("gutterRight") startPosition(510, 0) color("78350f") solid(true)
        Type { rectangle(30, 1200) } }

    Entity {
        id("ball")
        startPosition(285, 100)
        color("2563eb")
        Type { circle(32) }
        Movement { speed(0) AutoMove { direction(0, 0) } }
        Events {
            OnCollision("gutterLeft") { Actions { sound("bump") setVelocity(0, 500) } }
            OnCollision("gutterRight") { Actions { sound("bump") setVelocity(0, 500) } }
            OnCollision { Actions { setVelocity(0, 0) } }
        }
    }

    Entity {
        id("lane")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSwipe("up") { Actions { sound("thud") updateBar("rolls", -1) setVelocity("ball", 0, 650)
                delay(2.6) { Actions { broadcast("newRoll") } } } }
            OnSwipe("left") { Actions { sound("thud") updateBar("rolls", -1) setVelocity("ball", -90, 650)
                delay(2.6) { Actions { broadcast("newRoll") } } } }
            OnSwipe("right") { Actions { sound("thud") updateBar("rolls", -1) setVelocity("ball", 90, 650)
                delay(2.6) { Actions { broadcast("newRoll") } } } }
        }
    }
    Entity {
        id("reset")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSignal("newRoll") {
                Actions {
                    setPosition("ball", 285, 100, 0)
                    setVelocity("ball", 0, 0)
                }
            }
        }
    }

    Entity {
        id("pin")
        template(true)
        startPosition(0, 0)
        color("f8fafc")
        Type { circle(20) }
        Events {
            OnCollision("ball") {
                Actions {
                    sound("glass")
                    updateBar("pins", 1)
                    burst(5, "f8fafc")
                    deleteEntity()
                }
            }
        }
    }
    Entity {
        id("pinSetter")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.05) {
                Actions {
                    spawn("pin", 265, 1000)
                    spawn("pin", 215, 1050)
                    spawn("pin", 315, 1050)
                    spawn("pin", 165, 1100)
                    spawn("pin", 265, 1100)
                    spawn("pin", 365, 1100)
                    spawn("pin", 215, 950)
                    spawn("pin", 315, 950)
                    spawn("pin", 265, 900)
                    spawn("pin", 265, 1150)
                }
            }
        }
    }
}
