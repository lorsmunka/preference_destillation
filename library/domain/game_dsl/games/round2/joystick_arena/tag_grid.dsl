# Tag on a grid: grab six flags before the two taggers catch you.
Window {
    size(8, 12)
    background("1e1b4b")
    music("bg_2")
    gravity(0)

    lives(3, "-- TAGGED OUT --", "lose")
    winAt("flags", 6, "-- ALL SIX FLAGS --", "win")
    dpad("runner")
    wallBox(0, 360, 800, 840, "4338ca", 20)

    Entity {
        id("runner")
        startPosition(400, 760)
        color("fde047")
        Type { circle(30) }
        Movement { GridMove(80, 0.2) { direction(0, 0) } }
    }

    Entity { id("flag1") startPosition(160, 440) color("f472b6") Type { star(5, 22, 10) }
        Events { OnCollision("runner") { Actions { sound("pickup") updateBar("flags", 1) burst(8, "f472b6") deleteEntity() } } } }
    Entity { id("flag2") startPosition(640, 440) color("f472b6") Type { star(5, 22, 10) }
        Events { OnCollision("runner") { Actions { sound("pickup") updateBar("flags", 1) burst(8, "f472b6") deleteEntity() } } } }
    Entity { id("flag3") startPosition(80, 760) color("f472b6") Type { star(5, 22, 10) }
        Events { OnCollision("runner") { Actions { sound("pickup") updateBar("flags", 1) burst(8, "f472b6") deleteEntity() } } } }
    Entity { id("flag4") startPosition(720, 760) color("f472b6") Type { star(5, 22, 10) }
        Events { OnCollision("runner") { Actions { sound("pickup") updateBar("flags", 1) burst(8, "f472b6") deleteEntity() } } } }
    Entity { id("flag5") startPosition(240, 1080) color("f472b6") Type { star(5, 22, 10) }
        Events { OnCollision("runner") { Actions { sound("pickup") updateBar("flags", 1) burst(8, "f472b6") deleteEntity() } } } }
    Entity { id("flag6") startPosition(560, 1080) color("f472b6") Type { star(5, 22, 10) }
        Events { OnCollision("runner") { Actions { sound("pickup") updateBar("flags", 1) burst(8, "f472b6") deleteEntity() } } } }

    Entity {
        id("pillar")
        startPosition(380, 920)
        color("312e81")
        solid(true)
        Type { rectangle(80, 80) }
    }

    Entity {
        id("hunter")
        startPosition(80, 1100)
        color("ef4444")
        Type { circle(28) }
        Movement { speed(1.5) chase("runner") }
        Events {
            OnCollision("runner") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    shake(5, 0.3)
                    burst(10, "ef4444")
                    setPosition("hunter", 80, 1100, 0)
                }
            }
        }
    }

    Entity {
        id("decoy")
        startPosition(700, 500)
        color("94a3b8")
        Type { circle(18) }
        Movement { speed(2) Path { waypoint(700, 500) waypoint(700, 1100) waypoint(100, 1100) waypoint(100, 500) } }
    }

    Entity {
        id("sneak")
        startPosition(700, 1100)
        color("fb923c")
        Type { circle(28) }
        Movement { speed(1.8) chase("decoy") }
        Events {
            OnElapsedTime(8.0) {
                Actions {
                    sound("buzz")
                    flash("ffffff", 0.4)
                    retarget("runner")
                }
            }
            OnCollision("runner") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    shake(5, 0.3)
                    burst(10, "fb923c")
                    setPosition("sneak", 700, 1100, 0)
                }
            }
        }
    }
}
