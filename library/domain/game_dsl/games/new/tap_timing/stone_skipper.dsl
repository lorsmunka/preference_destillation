# Stone Skipper: the power needle swings back and forth. Tap the lake to throw
# while it sits in the green zone; the wind moves the zone after every throw.
Window {
    size(12, 7)
    background("0c4a6e")
    music("bg_1")

    Bar("power") {
        initial(0)
        OnValue(0) { Actions { setPosition("needle", 100, 45, 0) } }
        OnValue(1) { Actions { setPosition("needle", 150, 45, 0) } }
        OnValue(2) { Actions { setPosition("needle", 200, 45, 0) } }
        OnValue(3) { Actions { setPosition("needle", 250, 45, 0) } }
        OnValue(4) { Actions { setPosition("needle", 300, 45, 0) } }
        OnValue(5) { Actions { setPosition("needle", 350, 45, 0) } }
        OnValue(6) { Actions { setPosition("needle", 400, 45, 0) } }
        OnValue(7) { Actions { setPosition("needle", 450, 45, 0) } }
        OnValue(8) { Actions { setPosition("needle", 500, 45, 0) } }
        OnValue(9) { Actions { setPosition("needle", 550, 45, 0) } }
        OnValue(10) { Actions { setPosition("needle", 600, 45, 0) } }
    }
    Bar("skips") { initial(0) }
    winAt("skips", 15, "-- KING OF THE LAKE --", "win")
    Bar("stones") {
        initial(10)
        OnValue(1) { Actions { setText("windLabel", "Last stone!") } }
        OnValue(0) { Actions { delay(1.6) { Actions {
            whenBar("skips", "<", 15) { Actions { sound("lose") gameOver("-- OUT OF STONES --") } }
        } } } }
    }
    Bar("ready") { initial(1) }
    Bar("target") { initial(0) }

    Entity { id("lake") startPosition(0, 150) color("0369a1") Type { rectangle(1200, 450) }
        Events {
            OnTapped("lake") { Actions { whenBar("ready", "==", 1) { Actions {
                updateBar("ready", -1)
                updateBar("stones", -1)
                whenBar("target", "==", 1) { Actions {
                    whenBar("power", "<", 1) { Actions { broadcast("plop") } }
                    whenBar("power", "==", 1) { Actions { broadcast("okay") } }
                    whenBar("power", ">=", 2) { Actions { whenBar("power", "<=", 4) { Actions { broadcast("great") } } } }
                    whenBar("power", "==", 5) { Actions { broadcast("okay") } }
                    whenBar("power", ">", 5) { Actions { broadcast("plop") } }
                } }
                whenBar("target", "==", 2) { Actions {
                    whenBar("power", "<", 3) { Actions { broadcast("plop") } }
                    whenBar("power", "==", 3) { Actions { broadcast("okay") } }
                    whenBar("power", ">=", 4) { Actions { whenBar("power", "<=", 6) { Actions { broadcast("great") } } } }
                    whenBar("power", "==", 7) { Actions { broadcast("okay") } }
                    whenBar("power", ">", 7) { Actions { broadcast("plop") } }
                } }
                whenBar("target", "==", 3) { Actions {
                    whenBar("power", "<", 5) { Actions { broadcast("plop") } }
                    whenBar("power", "==", 5) { Actions { broadcast("okay") } }
                    whenBar("power", ">=", 6) { Actions { whenBar("power", "<=", 8) { Actions { broadcast("great") } } } }
                    whenBar("power", "==", 9) { Actions { broadcast("okay") } }
                    whenBar("power", ">", 9) { Actions { broadcast("plop") } }
                } }
                delay(1.5) { Actions { updateBar("ready", 1) broadcast("wind") } }
            } } } }
            OnSignal("great") { Actions { sound("sparkle") spawn("skimmer", 120, 320) loopCount(3) updateBar("skips", 1) } }
            OnSignal("okay") { Actions { sound("pop") spawn("hopper", 120, 320) updateBar("skips", 1) } }
            OnSignal("plop") { Actions { sound("thud") spawn("sinker", 120, 320) } }
        } }

    Entity { id("skimmer") template(true) startPosition(0, 0) color("d6d3d1") Type { circle(12) }
        Movement { speed(7) AutoMove { direction(1, 0) } }
        Events {
            OnElapsedTime(0.35) { repeatCount(3) Actions { sound("bubble") burst(8, "bae6fd") } }
            OnElapsedTime(1.4) { Actions { burst(14, "38bdf8") deleteEntity() } }
        } }
    Entity { id("hopper") template(true) startPosition(0, 0) color("d6d3d1") Type { circle(12) }
        Movement { speed(4) AutoMove { direction(1, 0) } }
        Events {
            OnElapsedTime(0.4) { Actions { sound("bubble") burst(8, "bae6fd") } }
            OnElapsedTime(0.8) { Actions { burst(14, "38bdf8") deleteEntity() } }
        } }
    Entity { id("sinker") template(true) startPosition(0, 0) color("a8a29e") Type { circle(12) }
        Movement { speed(2) AutoMove { direction(1, 0) } }
        Events {
            OnElapsedTime(0.3) { Actions { burst(18, "0ea5e9") shake(3, 0.2) deleteEntity() } }
        } }

    Entity { id("meter") startPosition(90, 60) color("1e293b") Type { rectangle(530, 30) } }
    Entity { id("zone") startPosition(285, 60) color("22c55e") Type { rectangle(140, 30) }
        Events {
            OnElapsedTime { Actions { broadcast("wind") } }
            OnSignal("wind") { Actions {
                whenBar("target", "==", 1) { Actions { updateBar("target", -1) } }
                whenBar("target", "==", 2) { Actions { updateBar("target", -2) } }
                whenBar("target", "==", 3) { Actions { updateBar("target", -3) } }
                randomAction {
                    Actions { updateBar("target", 1) setPosition("zone", 185, 60, 0.3) setText("windLabel", "Tailwind: throw soft") }
                    Actions { updateBar("target", 2) setPosition("zone", 285, 60, 0.3) setText("windLabel", "Calm water") }
                    Actions { updateBar("target", 3) setPosition("zone", 385, 60, 0.3) setText("windLabel", "Headwind: throw hard") }
                }
            } }
            OnSignal("great") { Actions { flash("fde047", 0.3) } }
        } }
    Entity { id("needle") startPosition(100, 45) color("ffffff") Type { rectangle(10, 60) }
        Events { Sequence {
            Phase(-1, 2.6) { Actions {
                ramp("power", 0, 10, 1.2)
                delay(1.3) { Actions { ramp("power", 10, 0, 1.2) } }
            } }
        } } }
    Entity { id("windLabel") startPosition(680, 70) color("e0f2fe") Type { text("Calm water", 36) } }
}
