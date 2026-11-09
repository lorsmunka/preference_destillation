# Hit a drum right after the metronome pulses. Tempo gets faster in three steps.
Window {
    size(8, 12)
    background("1c1917")
    music("bg_1")
    lives(4, "OFF BEAT", "lose")
    winAt("score", 24, "THE CROWD GOES WILD", "win")
    Bar("window") { initial(0) }

    Entity {
        id("conductor")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(8, 1.0) { Actions { broadcast("beat") } }
                Phase(8, 0.8) { Actions { broadcast("beat") } }
                Phase(-1, 0.6) { Actions { broadcast("beat") } }
            }
        }
    }

    Entity {
        id("metronome")
        startPosition(340, 900)
        color("fbbf24")
        Type { circle(60) }
        Events {
            OnSignal("beat") {
                Actions {
                    updateBar("window", 1)
                    setScale(1.4)
                    sound("click")
                    delay(0.28) { Actions {
                        setScale(1)
                        whenBar("window", "==", 1) { Actions { updateBar("window", -1) } }
                    } }
                }
            }
        }
    }

    Entity {
        id("drumA")
        startPosition(60, 300)
        color("dc2626")
        Type { circle(110) }
        Events {
            OnTapped("drumA") { Actions {
                whenBar("window", "==", 0) { Actions { updateBar("lives", -1) sound("buzz") } }
                whenBar("window", "==", 1) { Actions { updateBar("score", 1) updateBar("window", -1) sound("thud") flash("fecaca", 0.15) } }
            } }
        }
    }
    Entity {
        id("drumB")
        startPosition(290, 150)
        color("2563eb")
        Type { circle(110) }
        Events {
            OnTapped("drumB") { Actions {
                whenBar("window", "==", 0) { Actions { updateBar("lives", -1) sound("buzz") } }
                whenBar("window", "==", 1) { Actions { updateBar("score", 1) updateBar("window", -1) sound("clang") flash("bfdbfe", 0.15) } }
            } }
        }
    }
    Entity {
        id("drumC")
        startPosition(520, 300)
        color("16a34a")
        Type { circle(110) }
        Events {
            OnTapped("drumC") { Actions {
                whenBar("window", "==", 0) { Actions { updateBar("lives", -1) sound("buzz") } }
                whenBar("window", "==", 1) { Actions { updateBar("score", 1) updateBar("window", -1) sound("bell") flash("bbf7d0", 0.15) } }
            } }
        }
    }
}
