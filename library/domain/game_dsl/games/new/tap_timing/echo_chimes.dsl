# Echo Chimes: the wind plays a tune on four chimes; tap them back in the same order.
# Three songs, each one longer and faster. A wrong chime costs a life and the wind plays again.
Window {
    size(8, 10)
    background("1e1b4b")
    music("bg_ambient")

    lives(3, "-- THE CHIMES FELL SILENT --", "lose")
    Bar("step") { initial(0) }
    Bar("listen") { initial(0) }

    # Song 1: 2 4 1
    Level {
        Entity { id("title") startPosition(250, 920) color("c7d2fe") Type { text("Song 1 of 3", 44) } }
        Entity { id("prompt") startPosition(250, 80) color("e0e7ff") Type { text("Listen...", 40) }
            Events {
                OnElapsedTime(1.2) { Actions { broadcast("play") } }
                OnSignal("play") { Actions {
                    whenBar("listen", "==", 1) { Actions { updateBar("listen", -1) } }
                    whenBar("step", "==", 1) { Actions { updateBar("step", -1) } }
                    whenBar("step", "==", 2) { Actions { updateBar("step", -2) } }
                    setText("prompt", "Listen...")
                    delay(0.6) { Actions { broadcast("ring2")
                        delay(0.8) { Actions { broadcast("ring4")
                            delay(0.8) { Actions { broadcast("ring1")
                                delay(0.6) { Actions { setText("prompt", "Your turn") updateBar("listen", 1) } }
                            } }
                        } }
                    } }
                } }
                OnSignal("wrong") { Actions {
                    whenBar("listen", "==", 1) { Actions { updateBar("listen", -1) } }
                    sound("buzz") shake(5, 0.3) updateBar("lives", -1)
                    setText("prompt", "Wrong chime!")
                    delay(1.2) { Actions { broadcast("play") } }
                } }
                OnSignal("solved") { Actions {
                    updateBar("listen", -1)
                    updateBar("step", -3)
                    sound("win2") burst(20, "fde68a") setText("prompt", "Lovely!")
                    delay(1.2) { Actions { nextLevel() } }
                } }
            } }
        Entity { id("c1") startPosition(110, 420) color("f472b6") Type { rectangle(80, 320) }
            Events { OnSignal("ring1") { Actions { sound("bell") flash("ffffff", 0.25) setScale(1.1) delay(0.25) { Actions { setScale(1) } } } }
                OnTapped("c1") { Actions { whenBar("listen", "==", 1) { Actions { broadcast("ring1")
                    whenBar("step", "!=", 2) { Actions { broadcast("wrong") } }
                    whenBar("step", "==", 2) { Actions { updateBar("step", 1) broadcast("solved") } } } } } } } }
        Entity { id("c2") startPosition(280, 360) color("60a5fa") Type { rectangle(80, 380) }
            Events { OnSignal("ring2") { Actions { sound("glass") flash("ffffff", 0.25) setScale(1.1) delay(0.25) { Actions { setScale(1) } } } }
                OnTapped("c2") { Actions { whenBar("listen", "==", 1) { Actions { broadcast("ring2")
                    whenBar("step", "!=", 0) { Actions { broadcast("wrong") } }
                    whenBar("step", "==", 0) { Actions { updateBar("step", 1) } } } } } } } }
        Entity { id("c3") startPosition(450, 450) color("34d399") Type { rectangle(80, 290) }
            Events { OnSignal("ring3") { Actions { sound("beep") flash("ffffff", 0.25) setScale(1.1) delay(0.25) { Actions { setScale(1) } } } }
                OnTapped("c3") { Actions { whenBar("listen", "==", 1) { Actions { broadcast("ring3") broadcast("wrong") } } } } } }
        Entity { id("c4") startPosition(620, 390) color("fbbf24") Type { rectangle(80, 350) }
            Events { OnSignal("ring4") { Actions { sound("blip") flash("ffffff", 0.25) setScale(1.1) delay(0.25) { Actions { setScale(1) } } } }
                OnTapped("c4") { Actions { whenBar("listen", "==", 1) { Actions { broadcast("ring4")
                    whenBar("step", "!=", 1) { Actions { broadcast("wrong") } }
                    whenBar("step", "==", 1) { Actions { updateBar("step", 1) } } } } } } } }
    }

    # Song 2: 3 1 4 2
    Level {
        Entity { id("title") startPosition(250, 920) color("c7d2fe") Type { text("Song 2 of 3", 44) } }
        Entity { id("prompt") startPosition(250, 80) color("e0e7ff") Type { text("Listen...", 40) }
            Events {
                OnElapsedTime(1.2) { Actions { broadcast("play") } }
                OnSignal("play") { Actions {
                    whenBar("listen", "==", 1) { Actions { updateBar("listen", -1) } }
                    whenBar("step", "==", 1) { Actions { updateBar("step", -1) } }
                    whenBar("step", "==", 2) { Actions { updateBar("step", -2) } }
                    whenBar("step", "==", 3) { Actions { updateBar("step", -3) } }
                    setText("prompt", "Listen...")
                    delay(0.6) { Actions { broadcast("ring3")
                        delay(0.7) { Actions { broadcast("ring1")
                            delay(0.7) { Actions { broadcast("ring4")
                                delay(0.7) { Actions { broadcast("ring2")
                                    delay(0.6) { Actions { setText("prompt", "Your turn") updateBar("listen", 1) } }
                                } }
                            } }
                        } }
                    } }
                } }
                OnSignal("wrong") { Actions {
                    whenBar("listen", "==", 1) { Actions { updateBar("listen", -1) } }
                    sound("buzz") shake(5, 0.3) updateBar("lives", -1)
                    setText("prompt", "Wrong chime!")
                    delay(1.2) { Actions { broadcast("play") } }
                } }
                OnSignal("solved") { Actions {
                    updateBar("listen", -1)
                    updateBar("step", -4)
                    sound("win2") burst(20, "fde68a") setText("prompt", "Beautiful!")
                    delay(1.2) { Actions { nextLevel() } }
                } }
            } }
        Entity { id("c1") startPosition(110, 420) color("f472b6") Type { rectangle(80, 320) }
            Events { OnSignal("ring1") { Actions { sound("bell") flash("ffffff", 0.25) setScale(1.1) delay(0.25) { Actions { setScale(1) } } } }
                OnTapped("c1") { Actions { whenBar("listen", "==", 1) { Actions { broadcast("ring1")
                    whenBar("step", "!=", 1) { Actions { broadcast("wrong") } }
                    whenBar("step", "==", 1) { Actions { updateBar("step", 1) } } } } } } } }
        Entity { id("c2") startPosition(280, 360) color("60a5fa") Type { rectangle(80, 380) }
            Events { OnSignal("ring2") { Actions { sound("glass") flash("ffffff", 0.25) setScale(1.1) delay(0.25) { Actions { setScale(1) } } } }
                OnTapped("c2") { Actions { whenBar("listen", "==", 1) { Actions { broadcast("ring2")
                    whenBar("step", "!=", 3) { Actions { broadcast("wrong") } }
                    whenBar("step", "==", 3) { Actions { updateBar("step", 1) broadcast("solved") } } } } } } } }
        Entity { id("c3") startPosition(450, 450) color("34d399") Type { rectangle(80, 290) }
            Events { OnSignal("ring3") { Actions { sound("beep") flash("ffffff", 0.25) setScale(1.1) delay(0.25) { Actions { setScale(1) } } } }
                OnTapped("c3") { Actions { whenBar("listen", "==", 1) { Actions { broadcast("ring3")
                    whenBar("step", "!=", 0) { Actions { broadcast("wrong") } }
                    whenBar("step", "==", 0) { Actions { updateBar("step", 1) } } } } } } } }
        Entity { id("c4") startPosition(620, 390) color("fbbf24") Type { rectangle(80, 350) }
            Events { OnSignal("ring4") { Actions { sound("blip") flash("ffffff", 0.25) setScale(1.1) delay(0.25) { Actions { setScale(1) } } } }
                OnTapped("c4") { Actions { whenBar("listen", "==", 1) { Actions { broadcast("ring4")
                    whenBar("step", "!=", 2) { Actions { broadcast("wrong") } }
                    whenBar("step", "==", 2) { Actions { updateBar("step", 1) } } } } } } } }
    }

    # Song 3: 1 3 2 4 3, played quicker
    Level {
        Entity { id("title") startPosition(250, 920) color("c7d2fe") Type { text("Song 3 of 3", 44) } }
        Entity { id("prompt") startPosition(250, 80) color("e0e7ff") Type { text("Listen...", 40) }
            Events {
                OnElapsedTime(1.2) { Actions { broadcast("play") } }
                OnSignal("play") { Actions {
                    whenBar("listen", "==", 1) { Actions { updateBar("listen", -1) } }
                    whenBar("step", "==", 1) { Actions { updateBar("step", -1) } }
                    whenBar("step", "==", 2) { Actions { updateBar("step", -2) } }
                    whenBar("step", "==", 3) { Actions { updateBar("step", -3) } }
                    whenBar("step", "==", 4) { Actions { updateBar("step", -4) } }
                    setText("prompt", "Listen...")
                    delay(0.6) { Actions { broadcast("ring1")
                        delay(0.55) { Actions { broadcast("ring3")
                            delay(0.55) { Actions { broadcast("ring2")
                                delay(0.55) { Actions { broadcast("ring4")
                                    delay(0.55) { Actions { broadcast("ring3")
                                        delay(0.5) { Actions { setText("prompt", "Your turn") updateBar("listen", 1) } }
                                    } }
                                } }
                            } }
                        } }
                    } }
                } }
                OnSignal("wrong") { Actions {
                    whenBar("listen", "==", 1) { Actions { updateBar("listen", -1) } }
                    sound("buzz") shake(5, 0.3) updateBar("lives", -1)
                    setText("prompt", "Wrong chime!")
                    delay(1.2) { Actions { broadcast("play") } }
                } }
                OnSignal("solved") { Actions {
                    updateBar("listen", -1)
                    sound("win") burst(30, "fde68a") shake(4, 0.4)
                    gameOver("-- THE GARDEN SINGS --")
                } }
            } }
        Entity { id("c1") startPosition(110, 420) color("f472b6") Type { rectangle(80, 320) }
            Events { OnSignal("ring1") { Actions { sound("bell") flash("ffffff", 0.25) setScale(1.1) delay(0.25) { Actions { setScale(1) } } } }
                OnTapped("c1") { Actions { whenBar("listen", "==", 1) { Actions { broadcast("ring1")
                    whenBar("step", "!=", 0) { Actions { broadcast("wrong") } }
                    whenBar("step", "==", 0) { Actions { updateBar("step", 1) } } } } } } } }
        Entity { id("c2") startPosition(280, 360) color("60a5fa") Type { rectangle(80, 380) }
            Events { OnSignal("ring2") { Actions { sound("glass") flash("ffffff", 0.25) setScale(1.1) delay(0.25) { Actions { setScale(1) } } } }
                OnTapped("c2") { Actions { whenBar("listen", "==", 1) { Actions { broadcast("ring2")
                    whenBar("step", "!=", 2) { Actions { broadcast("wrong") } }
                    whenBar("step", "==", 2) { Actions { updateBar("step", 1) } } } } } } } }
        Entity { id("c3") startPosition(450, 450) color("34d399") Type { rectangle(80, 290) }
            Events { OnSignal("ring3") { Actions { sound("beep") flash("ffffff", 0.25) setScale(1.1) delay(0.25) { Actions { setScale(1) } } } }
                OnTapped("c3") { Actions { whenBar("listen", "==", 1) { Actions { broadcast("ring3")
                    whenBar("step", "!=", 1) { Actions { whenBar("step", "!=", 4) { Actions { broadcast("wrong") } } } }
                    whenBar("step", "==", 4) { Actions { updateBar("step", 1) broadcast("solved") } }
                    whenBar("step", "==", 1) { Actions { updateBar("step", 1) } } } } } } } }
        Entity { id("c4") startPosition(620, 390) color("fbbf24") Type { rectangle(80, 350) }
            Events { OnSignal("ring4") { Actions { sound("blip") flash("ffffff", 0.25) setScale(1.1) delay(0.25) { Actions { setScale(1) } } } }
                OnTapped("c4") { Actions { whenBar("listen", "==", 1) { Actions { broadcast("ring4")
                    whenBar("step", "!=", 3) { Actions { broadcast("wrong") } }
                    whenBar("step", "==", 3) { Actions { updateBar("step", 1) } } } } } } } }
    }
}
