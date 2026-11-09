Window {
    size(8, 10)
    background("1e1b4b")
    winAt("score", 8, "-- QUICK FINGERS --", "win")
    lives(3, "-- TOO SLOW --", "lose")
    Bar("score") { initial(0) }

    Entity {
        id("title")
        startPosition(180, 940)
        color("c7d2fe")
        Type { text("Tap before it shrinks", 36) }
    }

    Entity {
        id("decoy")
        startPosition(60, 80)
        color("64748b")
        Type { rectangle(90, 90) }
        Events { OnTapped { Actions { sound("buzz") updateBar("lives", -1) } } }
    }

    Entity {
        id("dot")
        startPosition(380, 480)
        color("f472b6")
        Type { circle(60) }
        Events {
            OnTapped {
                Actions {
                    sound("pop")
                    burst(10, "f472b6")
                    updateBar("score", 1)
                    setScale(1)
                    setPosition("dot", random(80, 640), random(100, 780), 0)
                }
            }
            OnElapsedTime(1.6) {
                repeatCount(-1)
                Actions {
                    sound("buzz")
                    updateBar("lives", -1)
                    setScale(1)
                    setPosition("dot", random(80, 640), random(100, 780), 0)
                    delay(0.5) { Actions { setScale(0.75) } }
                    delay(1.0) { Actions { setScale(0.5) } }
                }
            }
        }
    }
}
