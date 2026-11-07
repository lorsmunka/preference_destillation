# Lantern Ring: a light races around eight paper lanterns. Each round the lanterns
# take new colours: gold is worth 3, silver 1, red costs a life. Tap STOP to land on one.
Window {
    size(7, 10)
    background("18181b")
    music("bg_6")

    lives(3, "-- THE LANTERNS WENT DARK --", "lose")
    Bar("score") {
        initial(0)
        OnValue(6) { Actions { updateBar("gear", 1) setText("msg", "The light speeds up!") } }
        OnValue(13) { Actions { updateBar("gear", 1) setText("msg", "Faster still!") } }
    }
    winAt("score", 20, "-- FESTIVAL CHAMPION --", "win")
    Bar("pos") { initial(1) }
    Bar("running") { initial(0) }
    Bar("gear") { initial(0) }

    Entity { id("spinner") startPosition(330, 580) color("fbbf24") Type { star(8, 20, 8) }
        Events {
            OnElapsedTime(0.16) { repeatCount(-1) Actions { whenBar("gear", "==", 0) { Actions { broadcast("step") } } } }
            OnElapsedTime(0.11) { repeatCount(-1) Actions { whenBar("gear", "==", 1) { Actions { broadcast("step") } } } }
            OnElapsedTime(0.07) { repeatCount(-1) Actions { whenBar("gear", "==", 2) { Actions { broadcast("step") } } } }
            OnSignal("step") { Actions { whenBar("running", "==", 1) { Actions {
                updateBar("pos", 1)
                whenBar("pos", "==", 9) { Actions { updateBar("pos", -8) } }
                broadcast("tick")
            } } } }
        } }

    Entity { id("lamp1") startPosition(320, 790) color("78716c") Type { circle(30) } var("kind", 0)
        Events {
            OnSignal("shuffle") { Actions {
                whenVar("kind", "==", 1) { Actions { changeVar("kind", -1) } }
                whenVar("kind", "==", 2) { Actions { changeVar("kind", -2) } }
                whenVar("kind", "==", 3) { Actions { changeVar("kind", -3) } }
                randomAction {
                    Actions { changeVar("kind", 1) setColor("facc15") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 3) setColor("ef4444") }
                    Actions { setColor("78716c") }
                    Actions { setColor("78716c") }
                } } }
            OnSignal("tick") { Actions {
                whenBar("pos", "==", 1) { Actions { setOpacity(1) setScale(1.3) } }
                whenBar("pos", "!=", 1) { Actions { setOpacity(0.35) setScale(1) } } } }
            OnSignal("judge") { Actions { whenBar("pos", "==", 1) { Actions {
                burst(14, "fde68a")
                whenVar("kind", "==", 0) { Actions { broadcast("blank") } }
                whenVar("kind", "==", 1) { Actions { broadcast("gold") } }
                whenVar("kind", "==", 2) { Actions { broadcast("silver") } }
                whenVar("kind", "==", 3) { Actions { broadcast("red") } } } } } }
        } }

    Entity { id("lamp2") startPosition(476, 726) color("78716c") Type { circle(30) } var("kind", 0)
        Events {
            OnSignal("shuffle") { Actions {
                whenVar("kind", "==", 1) { Actions { changeVar("kind", -1) } }
                whenVar("kind", "==", 2) { Actions { changeVar("kind", -2) } }
                whenVar("kind", "==", 3) { Actions { changeVar("kind", -3) } }
                randomAction {
                    Actions { changeVar("kind", 1) setColor("facc15") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 3) setColor("ef4444") }
                    Actions { setColor("78716c") }
                    Actions { setColor("78716c") }
                } } }
            OnSignal("tick") { Actions {
                whenBar("pos", "==", 2) { Actions { setOpacity(1) setScale(1.3) } }
                whenBar("pos", "!=", 2) { Actions { setOpacity(0.35) setScale(1) } } } }
            OnSignal("judge") { Actions { whenBar("pos", "==", 2) { Actions {
                burst(14, "fde68a")
                whenVar("kind", "==", 0) { Actions { broadcast("blank") } }
                whenVar("kind", "==", 1) { Actions { broadcast("gold") } }
                whenVar("kind", "==", 2) { Actions { broadcast("silver") } }
                whenVar("kind", "==", 3) { Actions { broadcast("red") } } } } } }
        } }

    Entity { id("lamp3") startPosition(540, 570) color("78716c") Type { circle(30) } var("kind", 0)
        Events {
            OnSignal("shuffle") { Actions {
                whenVar("kind", "==", 1) { Actions { changeVar("kind", -1) } }
                whenVar("kind", "==", 2) { Actions { changeVar("kind", -2) } }
                whenVar("kind", "==", 3) { Actions { changeVar("kind", -3) } }
                randomAction {
                    Actions { changeVar("kind", 1) setColor("facc15") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 3) setColor("ef4444") }
                    Actions { setColor("78716c") }
                    Actions { setColor("78716c") }
                } } }
            OnSignal("tick") { Actions {
                whenBar("pos", "==", 3) { Actions { setOpacity(1) setScale(1.3) } }
                whenBar("pos", "!=", 3) { Actions { setOpacity(0.35) setScale(1) } } } }
            OnSignal("judge") { Actions { whenBar("pos", "==", 3) { Actions {
                burst(14, "fde68a")
                whenVar("kind", "==", 0) { Actions { broadcast("blank") } }
                whenVar("kind", "==", 1) { Actions { broadcast("gold") } }
                whenVar("kind", "==", 2) { Actions { broadcast("silver") } }
                whenVar("kind", "==", 3) { Actions { broadcast("red") } } } } } }
        } }

    Entity { id("lamp4") startPosition(476, 414) color("78716c") Type { circle(30) } var("kind", 0)
        Events {
            OnSignal("shuffle") { Actions {
                whenVar("kind", "==", 1) { Actions { changeVar("kind", -1) } }
                whenVar("kind", "==", 2) { Actions { changeVar("kind", -2) } }
                whenVar("kind", "==", 3) { Actions { changeVar("kind", -3) } }
                randomAction {
                    Actions { changeVar("kind", 1) setColor("facc15") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 3) setColor("ef4444") }
                    Actions { setColor("78716c") }
                    Actions { setColor("78716c") }
                } } }
            OnSignal("tick") { Actions {
                whenBar("pos", "==", 4) { Actions { setOpacity(1) setScale(1.3) } }
                whenBar("pos", "!=", 4) { Actions { setOpacity(0.35) setScale(1) } } } }
            OnSignal("judge") { Actions { whenBar("pos", "==", 4) { Actions {
                burst(14, "fde68a")
                whenVar("kind", "==", 0) { Actions { broadcast("blank") } }
                whenVar("kind", "==", 1) { Actions { broadcast("gold") } }
                whenVar("kind", "==", 2) { Actions { broadcast("silver") } }
                whenVar("kind", "==", 3) { Actions { broadcast("red") } } } } } }
        } }

    Entity { id("lamp5") startPosition(320, 350) color("78716c") Type { circle(30) } var("kind", 0)
        Events {
            OnSignal("shuffle") { Actions {
                whenVar("kind", "==", 1) { Actions { changeVar("kind", -1) } }
                whenVar("kind", "==", 2) { Actions { changeVar("kind", -2) } }
                whenVar("kind", "==", 3) { Actions { changeVar("kind", -3) } }
                randomAction {
                    Actions { changeVar("kind", 1) setColor("facc15") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 3) setColor("ef4444") }
                    Actions { setColor("78716c") }
                    Actions { setColor("78716c") }
                } } }
            OnSignal("tick") { Actions {
                whenBar("pos", "==", 5) { Actions { setOpacity(1) setScale(1.3) } }
                whenBar("pos", "!=", 5) { Actions { setOpacity(0.35) setScale(1) } } } }
            OnSignal("judge") { Actions { whenBar("pos", "==", 5) { Actions {
                burst(14, "fde68a")
                whenVar("kind", "==", 0) { Actions { broadcast("blank") } }
                whenVar("kind", "==", 1) { Actions { broadcast("gold") } }
                whenVar("kind", "==", 2) { Actions { broadcast("silver") } }
                whenVar("kind", "==", 3) { Actions { broadcast("red") } } } } } }
        } }

    Entity { id("lamp6") startPosition(164, 414) color("78716c") Type { circle(30) } var("kind", 0)
        Events {
            OnSignal("shuffle") { Actions {
                whenVar("kind", "==", 1) { Actions { changeVar("kind", -1) } }
                whenVar("kind", "==", 2) { Actions { changeVar("kind", -2) } }
                whenVar("kind", "==", 3) { Actions { changeVar("kind", -3) } }
                randomAction {
                    Actions { changeVar("kind", 1) setColor("facc15") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 3) setColor("ef4444") }
                    Actions { setColor("78716c") }
                    Actions { setColor("78716c") }
                } } }
            OnSignal("tick") { Actions {
                whenBar("pos", "==", 6) { Actions { setOpacity(1) setScale(1.3) } }
                whenBar("pos", "!=", 6) { Actions { setOpacity(0.35) setScale(1) } } } }
            OnSignal("judge") { Actions { whenBar("pos", "==", 6) { Actions {
                burst(14, "fde68a")
                whenVar("kind", "==", 0) { Actions { broadcast("blank") } }
                whenVar("kind", "==", 1) { Actions { broadcast("gold") } }
                whenVar("kind", "==", 2) { Actions { broadcast("silver") } }
                whenVar("kind", "==", 3) { Actions { broadcast("red") } } } } } }
        } }

    Entity { id("lamp7") startPosition(100, 570) color("78716c") Type { circle(30) } var("kind", 0)
        Events {
            OnSignal("shuffle") { Actions {
                whenVar("kind", "==", 1) { Actions { changeVar("kind", -1) } }
                whenVar("kind", "==", 2) { Actions { changeVar("kind", -2) } }
                whenVar("kind", "==", 3) { Actions { changeVar("kind", -3) } }
                randomAction {
                    Actions { changeVar("kind", 1) setColor("facc15") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 3) setColor("ef4444") }
                    Actions { setColor("78716c") }
                    Actions { setColor("78716c") }
                } } }
            OnSignal("tick") { Actions {
                whenBar("pos", "==", 7) { Actions { setOpacity(1) setScale(1.3) } }
                whenBar("pos", "!=", 7) { Actions { setOpacity(0.35) setScale(1) } } } }
            OnSignal("judge") { Actions { whenBar("pos", "==", 7) { Actions {
                burst(14, "fde68a")
                whenVar("kind", "==", 0) { Actions { broadcast("blank") } }
                whenVar("kind", "==", 1) { Actions { broadcast("gold") } }
                whenVar("kind", "==", 2) { Actions { broadcast("silver") } }
                whenVar("kind", "==", 3) { Actions { broadcast("red") } } } } } }
        } }

    Entity { id("lamp8") startPosition(164, 726) color("78716c") Type { circle(30) } var("kind", 0)
        Events {
            OnSignal("shuffle") { Actions {
                whenVar("kind", "==", 1) { Actions { changeVar("kind", -1) } }
                whenVar("kind", "==", 2) { Actions { changeVar("kind", -2) } }
                whenVar("kind", "==", 3) { Actions { changeVar("kind", -3) } }
                randomAction {
                    Actions { changeVar("kind", 1) setColor("facc15") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 2) setColor("cbd5e1") }
                    Actions { changeVar("kind", 3) setColor("ef4444") }
                    Actions { setColor("78716c") }
                    Actions { setColor("78716c") }
                } } }
            OnSignal("tick") { Actions {
                whenBar("pos", "==", 8) { Actions { setOpacity(1) setScale(1.3) } }
                whenBar("pos", "!=", 8) { Actions { setOpacity(0.35) setScale(1) } } } }
            OnSignal("judge") { Actions { whenBar("pos", "==", 8) { Actions {
                burst(14, "fde68a")
                whenVar("kind", "==", 0) { Actions { broadcast("blank") } }
                whenVar("kind", "==", 1) { Actions { broadcast("gold") } }
                whenVar("kind", "==", 2) { Actions { broadcast("silver") } }
                whenVar("kind", "==", 3) { Actions { broadcast("red") } } } } } }
        } }

    Entity { id("msg") startPosition(40, 930) color("fef3c7") Type { text("Stop the light on gold", 36) } }

    Entity { id("stop") startPosition(220, 70) color("dc2626") Type { rectangle(260, 110) }
        Events {
            OnElapsedTime(0.8) { Actions { broadcast("shuffle") updateBar("running", 1) } }
            OnTapped("stop") { Actions { whenBar("running", "==", 1) { Actions {
                sound("click")
                updateBar("running", -1)
                flash("ffffff", 0.15)
                broadcast("judge")
                delay(1.3) { Actions { broadcast("shuffle") updateBar("running", 1) } }
            } } } }
            OnSignal("gold") { Actions { sound("win2") shake(4, 0.3) setText("msg", "GOLD! +3") loopCount(3) updateBar("score", 1) } }
            OnSignal("silver") { Actions { sound("coin") setText("msg", "Silver, +1") updateBar("score", 1) } }
            OnSignal("red") { Actions { sound("hurt") shake(7, 0.4) setText("msg", "Red lantern! A life lost") updateBar("lives", -1) } }
            OnSignal("blank") { Actions { sound("thud") setText("msg", "An empty lantern") } }
        } }
    Entity { id("stopLabel") startPosition(290, 100) color("ffffff") Type { text("STOP", 48) } }
}
