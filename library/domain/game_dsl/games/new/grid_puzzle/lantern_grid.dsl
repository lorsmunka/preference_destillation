# Festival lanterns: tapping a lantern flips it and its four neighbours.
Window {
    size(7, 9)
    background("0b1026")
    music("bg_ambient")

    Bar("glow") { initial(4) }
    Bar("moves") { initial(8)
        OnValue(0) { Actions { sound("lose") gameOver("-- OUT OF MATCHES --") } } }

    # The judge looks at the square only after every flip of a tap is done.
    Entity { id("title") startPosition(150, 800) color("fde68a") Type { text("LIGHT EVERY LANTERN", 34) }
        Events { OnSignal("judge") { Actions {
            whenBar("glow", "==", 9) { Actions { sound("win") shake(5, 0.5) burst(30, "fbbf24") gameOver("-- THE SQUARE IS GLOWING --") } }
            whenBar("glow", "<", 9) { Actions { updateBar("moves", -1) } } } } } }
    Entity { id("hint") startPosition(170, 120) color("64748b") Type { text("a tap flips the cross", 28) } }

    Entity { id("a1") startPosition(120, 600) color("44403c") Type { circle(60) } var("lit", 0)
        Events { OnTapped("a1") { Actions { sound("click") broadcast("flip_a1") broadcast("flip_a2") broadcast("flip_b1") broadcast("judge") } }
            OnSignal("flip_a1") { Actions {
                whenVar("lit", "==", 0) { Actions { changeVar("lit", 2) setColor("f59e0b") updateBar("glow", 1) burst(6, "fde68a") } }
                whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("44403c") updateBar("glow", -1) } }
                whenVar("lit", "==", 2) { Actions { changeVar("lit", -1) } } } } } }
    Entity { id("a2") startPosition(290, 600) color("f59e0b") Type { circle(60) } var("lit", 1)
        Events { OnTapped("a2") { Actions { sound("click") broadcast("flip_a1") broadcast("flip_a2") broadcast("flip_a3") broadcast("flip_b2") broadcast("judge") } }
            OnSignal("flip_a2") { Actions {
                whenVar("lit", "==", 0) { Actions { changeVar("lit", 2) setColor("f59e0b") updateBar("glow", 1) burst(6, "fde68a") } }
                whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("44403c") updateBar("glow", -1) } }
                whenVar("lit", "==", 2) { Actions { changeVar("lit", -1) } } } } } }
    Entity { id("a3") startPosition(460, 600) color("44403c") Type { circle(60) } var("lit", 0)
        Events { OnTapped("a3") { Actions { sound("click") broadcast("flip_a2") broadcast("flip_a3") broadcast("flip_b3") broadcast("judge") } }
            OnSignal("flip_a3") { Actions {
                whenVar("lit", "==", 0) { Actions { changeVar("lit", 2) setColor("f59e0b") updateBar("glow", 1) burst(6, "fde68a") } }
                whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("44403c") updateBar("glow", -1) } }
                whenVar("lit", "==", 2) { Actions { changeVar("lit", -1) } } } } } }
    Entity { id("b1") startPosition(120, 430) color("f59e0b") Type { circle(60) } var("lit", 1)
        Events { OnTapped("b1") { Actions { sound("click") broadcast("flip_a1") broadcast("flip_b1") broadcast("flip_b2") broadcast("flip_c1") broadcast("judge") } }
            OnSignal("flip_b1") { Actions {
                whenVar("lit", "==", 0) { Actions { changeVar("lit", 2) setColor("f59e0b") updateBar("glow", 1) burst(6, "fde68a") } }
                whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("44403c") updateBar("glow", -1) } }
                whenVar("lit", "==", 2) { Actions { changeVar("lit", -1) } } } } } }
    Entity { id("b2") startPosition(290, 430) color("44403c") Type { circle(60) } var("lit", 0)
        Events { OnTapped("b2") { Actions { sound("click") broadcast("flip_a2") broadcast("flip_b1") broadcast("flip_b2") broadcast("flip_b3") broadcast("flip_c2") broadcast("judge") } }
            OnSignal("flip_b2") { Actions {
                whenVar("lit", "==", 0) { Actions { changeVar("lit", 2) setColor("f59e0b") updateBar("glow", 1) burst(6, "fde68a") } }
                whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("44403c") updateBar("glow", -1) } }
                whenVar("lit", "==", 2) { Actions { changeVar("lit", -1) } } } } } }
    Entity { id("b3") startPosition(460, 430) color("f59e0b") Type { circle(60) } var("lit", 1)
        Events { OnTapped("b3") { Actions { sound("click") broadcast("flip_a3") broadcast("flip_b2") broadcast("flip_b3") broadcast("flip_c3") broadcast("judge") } }
            OnSignal("flip_b3") { Actions {
                whenVar("lit", "==", 0) { Actions { changeVar("lit", 2) setColor("f59e0b") updateBar("glow", 1) burst(6, "fde68a") } }
                whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("44403c") updateBar("glow", -1) } }
                whenVar("lit", "==", 2) { Actions { changeVar("lit", -1) } } } } } }
    Entity { id("c1") startPosition(120, 260) color("44403c") Type { circle(60) } var("lit", 0)
        Events { OnTapped("c1") { Actions { sound("click") broadcast("flip_b1") broadcast("flip_c1") broadcast("flip_c2") broadcast("judge") } }
            OnSignal("flip_c1") { Actions {
                whenVar("lit", "==", 0) { Actions { changeVar("lit", 2) setColor("f59e0b") updateBar("glow", 1) burst(6, "fde68a") } }
                whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("44403c") updateBar("glow", -1) } }
                whenVar("lit", "==", 2) { Actions { changeVar("lit", -1) } } } } } }
    Entity { id("c2") startPosition(290, 260) color("44403c") Type { circle(60) } var("lit", 0)
        Events { OnTapped("c2") { Actions { sound("click") broadcast("flip_b2") broadcast("flip_c1") broadcast("flip_c2") broadcast("flip_c3") broadcast("judge") } }
            OnSignal("flip_c2") { Actions {
                whenVar("lit", "==", 0) { Actions { changeVar("lit", 2) setColor("f59e0b") updateBar("glow", 1) burst(6, "fde68a") } }
                whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("44403c") updateBar("glow", -1) } }
                whenVar("lit", "==", 2) { Actions { changeVar("lit", -1) } } } } } }
    Entity { id("c3") startPosition(460, 260) color("f59e0b") Type { circle(60) } var("lit", 1)
        Events { OnTapped("c3") { Actions { sound("click") broadcast("flip_b3") broadcast("flip_c2") broadcast("flip_c3") broadcast("judge") } }
            OnSignal("flip_c3") { Actions {
                whenVar("lit", "==", 0) { Actions { changeVar("lit", 2) setColor("f59e0b") updateBar("glow", 1) burst(6, "fde68a") } }
                whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("44403c") updateBar("glow", -1) } }
                whenVar("lit", "==", 2) { Actions { changeVar("lit", -1) } } } } } }
}
