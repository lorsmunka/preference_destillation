Window {
    size(7, 11)
    background("27272a")
    gravity(600)
    music("bg_3")

    lives(4, "OUT OF SHOTS", "lose")
    Bar("baskets") { initial(0) }
    winAt("baskets", 5, "5 BASKETS, YOU WIN", "win")

    Entity { id("backboard") startPosition(640, 640) color("e4e4e7") solid(true)
        Type { rectangle(20, 220) } }

    Entity {
        id("hoop")
        startPosition(420, 700)
        color("f97316")
        Type { rectangle(110, 14) }
        Movement { speed(1.2) Path { waypoint(380, 700) waypoint(380, 560) } }
        Events {
            OnCollision("ball") {
                Actions {
                    sound("coin")
                    updateBar("baskets", 1)
                    burst(14, "fb923c")
                    deleteCollider()
                }
            }
        }
    }

    Entity {
        id("floor")
        startPosition(0, 0)
        color("3f3f46")
        Type { rectangle(700, 30) }
        Events {
            OnCollision("ball") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    deleteCollider()
                }
            }
        }
    }

    Entity {
        id("ball")
        template(true)
        startPosition(0, 0)
        color("fb923c")
        Type { circle(24) }
        Movement { speed(0) AutoMove { direction(0, 0) } }
        Events {
            OnCollision("backboard") { Actions { sound("clang") bounceBack() } }
            OnCollision { Actions { bounceBack() } }
        }
    }

    Entity {
        id("player")
        startPosition(60, 60)
        color("38bdf8")
        Type { rectangle(50, 90) }
        Events {
            OnSwipe("up") {
                Actions {
                    sound("pop")
                    spawn("ball", 100, 160)
                    setVelocity("ball", 260, 850)
                }
            }
            OnSwipe("right") {
                Actions {
                    sound("pop")
                    spawn("ball", 100, 160)
                    setVelocity("ball", 400, 720)
                }
            }
            OnSwipe("down") {
                Actions {
                    sound("pop")
                    spawn("ball", 100, 160)
                    setVelocity("ball", 200, 900)
                }
            }
        }
    }
}
