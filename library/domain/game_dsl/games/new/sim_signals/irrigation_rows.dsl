# Irrigation Rows: four plots dry out at different speeds, and a crop only grows while its soil is between 3 and 7.
# Swipe left / right to move the sprinkler, swipe up to water the plot below it, swipe down to pump water back out.
# A plot that dries to 0 or floods to 10 loses its crop (and a life) and is replanted.
Window {
    size(12, 8)
    background("292017")
    music("bg_3")
    gravity(0)

    lives(3, "-- THE CROP FAILED --", "lose")
    Bar("harvest") { initial(0) }
    winAt("harvest", 100, "-- THE BARNS ARE FULL --", "win")

    Bar("pos") { initial(0)
        OnValue(0) { Actions { setPosition("nozzle", 150, 560, 0.2) } }
        OnValue(1) { Actions { setPosition("nozzle", 430, 560, 0.2) } }
        OnValue(2) { Actions { setPosition("nozzle", 710, 560, 0.2) } }
        OnValue(3) { Actions { setPosition("nozzle", 990, 560, 0.2) } } }

    Bar("waterA") { initial(5)
        OnValue(0) { Actions { broadcast("lostA") updateBar("lives", -1) updateBar("waterA", 5) } }
        OnValue(2) { Actions { broadcast("dryA") } }
        OnValue(3) { Actions { broadcast("okA") } }
        OnValue(7) { Actions { broadcast("okA") } }
        OnValue(8) { Actions { broadcast("wetA") } }
        OnValue(10) { Actions { broadcast("lostA") updateBar("lives", -1) updateBar("waterA", -5) } } }
    Bar("waterB") { initial(5)
        OnValue(0) { Actions { broadcast("lostB") updateBar("lives", -1) updateBar("waterB", 5) } }
        OnValue(2) { Actions { broadcast("dryB") } }
        OnValue(3) { Actions { broadcast("okB") } }
        OnValue(7) { Actions { broadcast("okB") } }
        OnValue(8) { Actions { broadcast("wetB") } }
        OnValue(10) { Actions { broadcast("lostB") updateBar("lives", -1) updateBar("waterB", -5) } } }
    Bar("waterC") { initial(5)
        OnValue(0) { Actions { broadcast("lostC") updateBar("lives", -1) updateBar("waterC", 5) } }
        OnValue(2) { Actions { broadcast("dryC") } }
        OnValue(3) { Actions { broadcast("okC") } }
        OnValue(7) { Actions { broadcast("okC") } }
        OnValue(8) { Actions { broadcast("wetC") } }
        OnValue(10) { Actions { broadcast("lostC") updateBar("lives", -1) updateBar("waterC", -5) } } }
    Bar("waterD") { initial(5)
        OnValue(0) { Actions { broadcast("lostD") updateBar("lives", -1) updateBar("waterD", 5) } }
        OnValue(2) { Actions { broadcast("dryD") } }
        OnValue(3) { Actions { broadcast("okD") } }
        OnValue(7) { Actions { broadcast("okD") } }
        OnValue(8) { Actions { broadcast("wetD") } }
        OnValue(10) { Actions { broadcast("lostD") updateBar("lives", -1) updateBar("waterD", -5) } } }

    Entity { id("plotA") startPosition(60, 180) color("15803d") Type { rectangle(240, 300) }
        Events {
            OnElapsedTime(3) { repeatCount(-1) Actions { updateBar("waterA", -1) } }
            OnElapsedTime(2) { repeatCount(-1) Actions { whenBar("waterA", ">=", 3) { Actions { whenBar("waterA", "<=", 7) { Actions { updateBar("harvest", 1) } } } } } }
            OnSignal("dryA") { Actions { setColor("a16207") } }
            OnSignal("okA") { Actions { setColor("15803d") } }
            OnSignal("wetA") { Actions { setColor("1e3a8a") } }
            OnSignal("lostA") { Actions { setColor("15803d") sound("hurt") shake(5, 0.3) burst(20, "78350f") } } } }
    Entity { id("plotB") startPosition(340, 180) color("15803d") Type { rectangle(240, 300) }
        Events {
            OnElapsedTime(3.8) { repeatCount(-1) Actions { updateBar("waterB", -1) } }
            OnElapsedTime(2) { repeatCount(-1) Actions { whenBar("waterB", ">=", 3) { Actions { whenBar("waterB", "<=", 7) { Actions { updateBar("harvest", 1) } } } } } }
            OnSignal("dryB") { Actions { setColor("a16207") } }
            OnSignal("okB") { Actions { setColor("15803d") } }
            OnSignal("wetB") { Actions { setColor("1e3a8a") } }
            OnSignal("lostB") { Actions { setColor("15803d") sound("hurt") shake(5, 0.3) burst(20, "78350f") } } } }
    Entity { id("plotC") startPosition(620, 180) color("15803d") Type { rectangle(240, 300) }
        Events {
            OnElapsedTime(4.6) { repeatCount(-1) Actions { updateBar("waterC", -1) } }
            OnElapsedTime(2) { repeatCount(-1) Actions { whenBar("waterC", ">=", 3) { Actions { whenBar("waterC", "<=", 7) { Actions { updateBar("harvest", 1) } } } } } }
            OnSignal("dryC") { Actions { setColor("a16207") } }
            OnSignal("okC") { Actions { setColor("15803d") } }
            OnSignal("wetC") { Actions { setColor("1e3a8a") } }
            OnSignal("lostC") { Actions { setColor("15803d") sound("hurt") shake(5, 0.3) burst(20, "78350f") } } } }
    Entity { id("plotD") startPosition(900, 180) color("15803d") Type { rectangle(240, 300) }
        Events {
            OnElapsedTime(2.6) { repeatCount(-1) Actions { updateBar("waterD", -1) } }
            OnElapsedTime(2) { repeatCount(-1) Actions { whenBar("waterD", ">=", 3) { Actions { whenBar("waterD", "<=", 7) { Actions { updateBar("harvest", 1) } } } } } }
            OnSignal("dryD") { Actions { setColor("a16207") } }
            OnSignal("okD") { Actions { setColor("15803d") } }
            OnSignal("wetD") { Actions { setColor("1e3a8a") } }
            OnSignal("lostD") { Actions { setColor("15803d") sound("hurt") shake(5, 0.3) burst(20, "78350f") } } } }

    Entity { id("nozzle") startPosition(150, 560) color("38bdf8") Type { triangle(60, "down") } var("busy", 0)
        Events {
            OnSwipe("left") { Actions { whenBar("pos", ">", 0) { Actions { updateBar("pos", -1) sound("click") } } } }
            OnSwipe("right") { Actions { whenBar("pos", "<", 3) { Actions { updateBar("pos", 1) sound("click") } } } }
            OnSwipe("up") { Actions { whenVar("busy", "==", 0) { Actions {
                changeVar("busy", 1) sound("bubble") burst(10, "60a5fa")
                whenBar("pos", "==", 0) { Actions { updateBar("waterA", 1) } }
                whenBar("pos", "==", 1) { Actions { updateBar("waterB", 1) } }
                whenBar("pos", "==", 2) { Actions { updateBar("waterC", 1) } }
                whenBar("pos", "==", 3) { Actions { updateBar("waterD", 1) } }
                delay(0.3) { Actions { changeVar("busy", -1) } } } } } }
            OnSwipe("down") { Actions { whenVar("busy", "==", 0) { Actions {
                changeVar("busy", 1) sound("pop") burst(6, "a8a29e")
                whenBar("pos", "==", 0) { Actions { updateBar("waterA", -1) } }
                whenBar("pos", "==", 1) { Actions { updateBar("waterB", -1) } }
                whenBar("pos", "==", 2) { Actions { updateBar("waterC", -1) } }
                whenBar("pos", "==", 3) { Actions { updateBar("waterD", -1) } }
                delay(0.3) { Actions { changeVar("busy", -1) } } } } } }
        } }

    Entity { id("sun") startPosition(1080, 680) color("fbbf24") Type { star(12, 50, 34) }
        Events { OnElapsedTime(9) { repeatCount(-1) Actions { sound("buzz") flash("fef08a", 0.5) randomAction {
            Actions { flash("plotA", "fde047", 0.4) updateBar("waterA", -1) updateBar("waterA", -1) }
            Actions { flash("plotB", "fde047", 0.4) updateBar("waterB", -1) updateBar("waterB", -1) }
            Actions { flash("plotC", "fde047", 0.4) updateBar("waterC", -1) updateBar("waterC", -1) }
            Actions { flash("plotD", "fde047", 0.4) updateBar("waterD", -1) updateBar("waterD", -1) } } } } } }
}
