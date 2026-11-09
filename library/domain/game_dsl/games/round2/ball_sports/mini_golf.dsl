Window {
    size(8, 10)
    background("14532d")
    music("bg_nature")

    Bar("strokes") {
        initial(0)
        OnValue(7) { Actions { sound("lose") gameOver("OUT OF STROKES") } }
    }

    goal("hole", 640, 840, 90, 90, "HOLE IN ONE... OK, HOLE!")

    Entity { id("wallLeft") startPosition(0, 0) color("a16207") solid(true)
        Type { rectangle(20, 1000) } }
    Entity { id("wallRight") startPosition(780, 0) color("a16207") solid(true)
        Type { rectangle(20, 1000) } }
    Entity { id("wallTop") startPosition(0, 980) color("a16207") solid(true)
        Type { rectangle(800, 20) } }
    Entity { id("wallBottom") startPosition(0, 0) color("a16207") solid(true)
        Type { rectangle(800, 20) } }

    Entity {
        id("bumper")
        startPosition(330, 450)
        color("f97316")
        solid(true)
        Type { circle(60) }
        Events { OnCollision("hero") { Actions { flash("ffffff", 0.15) } } }
    }
    Entity {
        id("fence")
        startPosition(120, 660)
        color("78350f")
        solid(true)
        Type { rectangle(380, 24) }
    }

    Entity {
        id("windmill")
        startPosition(520, 400)
        color("e2e8f0")
        Type { rectangle(150, 22) }
        Movement {
            speed(2)
            Path {
                waypoint(520, 400)
                waypoint(520, 700)
            }
        }
    }

    Entity {
        id("hero")
        startPosition(100, 100)
        color("ffffff")
        Type { circle(16) }
        Movement { speed(0) AutoMove { direction(0, 0) } }
        Events {
            OnCollision("bumper") { Actions { sound("blip") bounceBack() } }
            OnCollision("fence") { Actions { sound("bump") bounceBack() } }
            OnCollision("windmill") { Actions { sound("bump") bounceBack() } }
            OnCollision("wallLeft") { Actions { sound("bump") bounceBack() } }
            OnCollision("wallRight") { Actions { sound("bump") bounceBack() } }
            OnCollision("wallTop") { Actions { sound("bump") bounceBack() } }
            OnCollision("wallBottom") { Actions { sound("bump") bounceBack() } }
        }
    }

    Entity {
        id("putter")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSwipe("up") {
                Actions {
                    sound("click")
                    updateBar("strokes", 1)
                    setVelocity("hero", 0, 600)
                    delay(1.4) { Actions { setVelocity("hero", 0, 0) } }
                }
            }
            OnSwipe("right") {
                Actions {
                    sound("click")
                    updateBar("strokes", 1)
                    setVelocity("hero", 600, 0)
                    delay(1.4) { Actions { setVelocity("hero", 0, 0) } }
                }
            }
            OnSwipe("left") {
                Actions {
                    sound("click")
                    updateBar("strokes", 1)
                    setVelocity("hero", -600, 0)
                    delay(1.4) { Actions { setVelocity("hero", 0, 0) } }
                }
            }
            OnSwipe("down") {
                Actions {
                    sound("click")
                    updateBar("strokes", 1)
                    setVelocity("hero", 0, -600)
                    delay(1.4) { Actions { setVelocity("hero", 0, 0) } }
                }
            }
        }
    }
}
