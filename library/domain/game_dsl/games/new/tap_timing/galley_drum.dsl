# Galley Drum: the drummer keeps the beat and your crew rows on it. Tap the oars just
# as the drum booms: a stroke on the beat moves the boat ahead, a stroke between beats
# makes the crew stumble and lose ground. Beat the rival boat to the flag in two races.
Window {
    size(10, 6)
    background("0c4a6e")
    music("bg_5")

    Bar("progress") {
        initial(0)
        OnValue(0) { Actions { setPosition("boat", 40, 250, 0.25) } }
        OnValue(1) { Actions { setPosition("boat", 95, 250, 0.25) } }
        OnValue(2) { Actions { setPosition("boat", 150, 250, 0.25) } }
        OnValue(3) { Actions { setPosition("boat", 205, 250, 0.25) } }
        OnValue(4) { Actions { setPosition("boat", 260, 250, 0.25) } }
        OnValue(5) { Actions { setPosition("boat", 315, 250, 0.25) } }
        OnValue(6) { Actions { setPosition("boat", 370, 250, 0.25) } }
        OnValue(7) { Actions { setPosition("boat", 425, 250, 0.25) } }
        OnValue(8) { Actions { setPosition("boat", 480, 250, 0.25) } }
        OnValue(9) { Actions { setPosition("boat", 535, 250, 0.25) } }
        OnValue(10) { Actions { setPosition("boat", 590, 250, 0.25) } }
        OnValue(11) { Actions { setPosition("boat", 645, 250, 0.25) } }
        OnValue(12) { Actions { setPosition("boat", 700, 250, 0.25) } }
        OnValue(13) { Actions { setPosition("boat", 755, 250, 0.25) } }
        OnValue(14) { Actions { setPosition("boat", 810, 250, 0.25) } }
        OnValue(15) { Actions { setPosition("boat", 865, 250, 0.25) } }
        OnValue(16) { Actions { setPosition("boat", 920, 250, 0.25) broadcast("crossed") } }
    }
    Bar("window") { initial(0) }

    # Race 1: a steady beat and a slow rival
    Level {
        Entity { id("flag") startPosition(940, 180) color("f8fafc") Type { rectangle(12, 340) } }
        Entity { id("rival") startPosition(40, 420) color("ef4444") Type { rectangle(70, 30) }
            Movement { speed(0.52) AutoMove { direction(1, 0) } }
            Events { OnCollision("flag") { Actions { sound("lose") shake(6, 0.4) gameOver("-- THE RIVAL CREW WON --") } } } }
        Entity { id("boat") startPosition(40, 250) color("facc15") Type { rectangle(70, 30) } }

        Entity { id("drum") startPosition(460, 40) color("b45309") Type { circle(40) }
            Events {
                Sequence {
                    Phase(12, 0.8) { Actions { broadcast("boom") } }
                    Phase(-1, 0.7) { Actions { broadcast("boom") } }
                }
                OnSignal("boom") { Actions {
                    sound("thud") flash("fde68a", 0.12) setScale(1.2)
                    whenBar("window", "==", 0) { Actions { updateBar("window", 1) } }
                    delay(0.28) { Actions {
                        setScale(1)
                        whenBar("window", "==", 1) { Actions { updateBar("window", -1) } }
                    } }
                } }
            } }

        Entity { id("oars") startPosition(40, 20) color("1e3a8a") Type { rectangle(300, 120) }
            Events { OnTapped("oars") { Actions {
                whenBar("window", "==", 0) { Actions { whenBar("progress", ">", 0) { Actions {
                    updateBar("progress", -1) sound("buzz") flash("boat", "ef4444", 0.2) setText("cox", "Off the beat!")
                } } } }
                whenBar("window", "==", 1) { Actions {
                    updateBar("window", -1) updateBar("progress", 1)
                    sound("blip") burst("boat", 6, "bae6fd") setText("cox", "Stroke!")
                } }
            } } } }
        Entity { id("cox") startPosition(620, 60) color("e0f2fe") Type { text("Row on the drum!", 36) }
            Events { OnSignal("crossed") { Actions {
                setEnable("rival", false)
                setEnable("oars", false)
                sound("win2") setText("cox", "Race 1 won!") burst("boat", 20, "facc15")
                delay(1.5) { Actions { updateBar("progress", -16) nextLevel() } }
            } } } }
    }

    # Race 2: a faster beat, a faster rival and a river current
    Level {
        Entity { id("flag") startPosition(940, 180) color("f8fafc") Type { rectangle(12, 340) } }
        Entity { id("rival") startPosition(40, 420) color("a855f7") Type { rectangle(70, 30) }
            Movement { speed(0.66) AutoMove { direction(1, 0) } }
            Events { OnCollision("flag") { Actions { sound("lose") shake(6, 0.4) gameOver("-- THE RIVAL CREW WON --") } } } }
        Entity { id("boat") startPosition(40, 250) color("facc15") Type { rectangle(70, 30) } }

        Entity { id("drum") startPosition(460, 40) color("b45309") Type { circle(40) }
            Events {
                Sequence {
                    Phase(10, 0.65) { Actions { broadcast("boom") } }
                    Phase(-1, 0.55) { Actions { broadcast("boom") } }
                }
                OnSignal("boom") { Actions {
                    sound("thud") flash("fde68a", 0.12) setScale(1.2)
                    whenBar("window", "==", 0) { Actions { updateBar("window", 1) } }
                    delay(0.24) { Actions {
                        setScale(1)
                        whenBar("window", "==", 1) { Actions { updateBar("window", -1) } }
                    } }
                } }
            } }

        Entity { id("current") startPosition(600, 330) color("38bdf8") Type { triangle(40, "left") }
            Events { OnElapsedTime(random(3, 5)) { repeatCount(-1) Actions { randomAction {
                Actions { }
                Actions { setOpacity(0.4) }
                Actions { setOpacity(1) shake(3, 0.2) whenBar("progress", ">", 0) { Actions {
                    updateBar("progress", -1) setText("cox", "The current pulls us back!") } } }
            } } } } }

        Entity { id("oars") startPosition(40, 20) color("1e3a8a") Type { rectangle(300, 120) }
            Events { OnTapped("oars") { Actions {
                whenBar("window", "==", 0) { Actions { whenBar("progress", ">", 0) { Actions {
                    updateBar("progress", -1) sound("buzz") flash("boat", "ef4444", 0.2) setText("cox", "Off the beat!")
                } } } }
                whenBar("window", "==", 1) { Actions {
                    updateBar("window", -1) updateBar("progress", 1)
                    sound("blip") burst("boat", 6, "bae6fd") setText("cox", "Stroke!")
                } }
            } } } }
        Entity { id("cox") startPosition(620, 60) color("e0f2fe") Type { text("Final race!", 36) }
            Events { OnSignal("crossed") { Actions {
                sound("win") shake(5, 0.5) burst("boat", 30, "facc15")
                gameOver("-- CHAMPIONS OF THE RIVER --")
            } } } }
    }
}
