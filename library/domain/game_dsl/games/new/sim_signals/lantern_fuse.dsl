# Lantern Fuse: a chain-reaction puzzle. A lit lantern sets fire to the fuses that lead away from it,
# and the next lantern catches half a second later. You only have a few sparks: tap the right lanterns
# so the chains light every one. If the sparks run out with lanterns still dark, the board resets.
Window {
    size(8, 10)
    background("0f0a1e")
    music("bg_6")
    gravity(0)

    Bar("glow") { initial(0)
        OnValue(6) { Actions { sound("win2") shake(4, 0.4) nextLevel() } }
        OnValue(16) { Actions { sound("win") shake(6, 0.6) gameOver("-- EVERY LANTERN IS LIT --") } } }
    Bar("sparks") { initial(1)
        OnValue(0) { Actions { broadcast("spent") } } }

    Level {
        Entity { id("title1") startPosition(40, 920) color("e9d5ff") Type { text("One spark. Find where the fuse begins.", 30) }
            Events { OnSignal("spent") { Actions { delay(3) { Actions { whenBar("glow", "<", 6) { Actions {
                setText("title1", "The chain broke. Again!") sound("lose") shake(5, 0.3)
                delay(1.2) { Actions { restart() } } } } } } } } } }

        Entity { id("wA") startPosition(240, 497) color("44403c") Type { rectangle(140, 6) }
            Events { OnSignal("litA1") { Actions { setColor("f59e0b") burst(6, "fb923c") } } } }
        Entity { id("wB") startPosition(407, 530) color("44403c") Type { rectangle(6, 140) }
            Events { OnSignal("litA2") { Actions { setColor("f59e0b") burst(6, "fb923c") } } } }
        Entity { id("wC") startPosition(407, 330) color("44403c") Type { rectangle(6, 140) }
            Events { OnSignal("litA2") { Actions { setColor("f59e0b") burst(6, "fb923c") } } } }
        Entity { id("wD") startPosition(440, 697) color("44403c") Type { rectangle(140, 6) }
            Events { OnSignal("litA3") { Actions { setColor("f59e0b") burst(6, "fb923c") } } } }
        Entity { id("wE") startPosition(440, 297) color("44403c") Type { rectangle(140, 6) }
            Events { OnSignal("litA4") { Actions { setColor("f59e0b") burst(6, "fb923c") } } } }

        Entity { id("a1") startPosition(180, 470) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events { OnTapped("a1") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1)
                delay(0.5) { Actions { broadcast("litA1") } } } } } } } } } }
        Entity { id("a2") startPosition(380, 470) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events {
                OnTapped("a2") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                    updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litA2") } } } } } } } }
                OnSignal("litA1") { Actions { whenVar("lit", "==", 0) { Actions {
                    changeVar("lit", 1) setColor("fbbf24") sound("pop") burst(10, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litA2") } } } } } } } }
        Entity { id("a3") startPosition(380, 670) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events {
                OnTapped("a3") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                    updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litA3") } } } } } } } }
                OnSignal("litA2") { Actions { whenVar("lit", "==", 0) { Actions {
                    changeVar("lit", 1) setColor("fbbf24") sound("pop") burst(10, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litA3") } } } } } } } }
        Entity { id("a4") startPosition(380, 270) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events {
                OnTapped("a4") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                    updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litA4") } } } } } } } }
                OnSignal("litA2") { Actions { whenVar("lit", "==", 0) { Actions {
                    changeVar("lit", 1) setColor("fbbf24") sound("pop") burst(10, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litA4") } } } } } } } }
        Entity { id("a5") startPosition(580, 270) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events {
                OnTapped("a5") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                    updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1) } } } } } }
                OnSignal("litA4") { Actions { whenVar("lit", "==", 0) { Actions {
                    changeVar("lit", 1) setColor("fbbf24") sound("pop") burst(10, "fde68a") updateBar("glow", 1) } } } } } }
        Entity { id("a6") startPosition(580, 670) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events {
                OnTapped("a6") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                    updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1) } } } } } }
                OnSignal("litA3") { Actions { whenVar("lit", "==", 0) { Actions {
                    changeVar("lit", 1) setColor("fbbf24") sound("pop") burst(10, "fde68a") updateBar("glow", 1) } } } } } }
    }

    Level {
        Entity { id("title2") startPosition(40, 920) color("e9d5ff") Type { text("Two sparks. The blue lantern is damp.", 30) }
            Events {
                OnElapsedTime { Actions { ramp("sparks", 2, 2, 0.1) } }
                OnSignal("spent") { Actions { delay(4) { Actions { whenBar("glow", "<", 16) { Actions {
                    setText("title2", "Still dark somewhere. Again!") sound("lose") shake(5, 0.3)
                    delay(1.2) { Actions { restart() } } } } } } } } } }

        Entity { id("v12") startPosition(160, 827) color("44403c") Type { rectangle(140, 6) }
            Events { OnSignal("litB1") { Actions { setColor("f59e0b") } } } }
        Entity { id("v23") startPosition(360, 827) color("44403c") Type { rectangle(140, 6) }
            Events { OnSignal("litB2") { Actions { setColor("f59e0b") } } } }
        Entity { id("v34") startPosition(560, 827) color("44403c") Type { rectangle(140, 6) }
            Events { OnSignal("litB3") { Actions { setColor("f59e0b") } } } }
        Entity { id("v101") startPosition(127, 660) color("44403c") Type { rectangle(6, 140) }
            Events { OnSignal("litB10") { Actions { setColor("f59e0b") } } } }
        Entity { id("v910") startPosition(160, 627) color("44403c") Type { rectangle(140, 6) }
            Events { OnSignal("litB9") { Actions { setColor("f59e0b") } } } }
        Entity { id("v89") startPosition(327, 460) color("44403c") Type { rectangle(6, 140) }
            Events { OnSignal("litB8") { Actions { setColor("f59e0b") } } } }
        Entity { id("v97") startPosition(360, 627) color("44403c") Type { rectangle(140, 6) }
            Events { OnSignal("litB9") { Actions { setColor("f59e0b") } } } }
        Entity { id("v67") startPosition(527, 460) color("44403c") Type { rectangle(6, 140) }
            Events { OnSignal("litB6") { Actions { setColor("f59e0b") } } } }
        Entity { id("v56") startPosition(560, 427) color("44403c") Type { rectangle(140, 6) }
            Events { OnSignal("litB5") { Actions { setColor("f59e0b") } } } }

        Entity { id("b1") startPosition(100, 800) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events {
                OnTapped("b1") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                    updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litB1") } } } } } } } }
                OnSignal("litB10") { Actions { whenVar("lit", "==", 0) { Actions {
                    changeVar("lit", 1) setColor("fbbf24") sound("pop") burst(10, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litB1") } } } } } } } }
        Entity { id("b2") startPosition(300, 800) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events {
                OnTapped("b2") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                    updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litB2") } } } } } } } }
                OnSignal("litB1") { Actions { whenVar("lit", "==", 0) { Actions {
                    changeVar("lit", 1) setColor("fbbf24") sound("pop") burst(10, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litB2") } } } } } } } }
        Entity { id("b3") startPosition(500, 800) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events {
                OnTapped("b3") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                    updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litB3") } } } } } } } }
                OnSignal("litB2") { Actions { whenVar("lit", "==", 0) { Actions {
                    changeVar("lit", 1) setColor("fbbf24") sound("pop") burst(10, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litB3") } } } } } } } }
        Entity { id("b4") startPosition(700, 800) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events {
                OnTapped("b4") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                    updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1) } } } } } }
                OnSignal("litB3") { Actions { whenVar("lit", "==", 0) { Actions {
                    changeVar("lit", 1) setColor("fbbf24") sound("pop") burst(10, "fde68a") updateBar("glow", 1) } } } } } }
        Entity { id("b5") startPosition(700, 400) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events { OnTapped("b5") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1)
                delay(0.5) { Actions { broadcast("litB5") } } } } } } } } } }
        Entity { id("b6") startPosition(500, 400) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events {
                OnTapped("b6") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                    updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litB6") } } } } } } } }
                OnSignal("litB5") { Actions { whenVar("lit", "==", 0) { Actions {
                    changeVar("lit", 1) setColor("fbbf24") sound("pop") burst(10, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litB6") } } } } } } } }
        Entity { id("b7") startPosition(500, 600) color("1e3a5f") Type { polygon(8, 30) } var("lit", 0) var("damp", 2)
            Events {
                OnTapped("b7") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                    updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1) } } } } } }
                OnSignal("litB6") { Actions { whenVar("lit", "==", 0) { Actions {
                    changeVar("damp", -1) setColor("64748b") burst(6, "e2e8f0")
                    whenVar("damp", "==", 0) { Actions {
                        changeVar("lit", 1) setColor("fbbf24") sound("pop") burst(10, "fde68a") updateBar("glow", 1) } } } } } }
                OnSignal("litB9") { Actions { whenVar("lit", "==", 0) { Actions {
                    changeVar("damp", -1) setColor("64748b") burst(6, "e2e8f0")
                    whenVar("damp", "==", 0) { Actions {
                        changeVar("lit", 1) setColor("fbbf24") sound("pop") burst(10, "fde68a") updateBar("glow", 1) } } } } } } } }
        Entity { id("b8") startPosition(300, 400) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events { OnTapped("b8") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1)
                delay(0.5) { Actions { broadcast("litB8") } } } } } } } } } }
        Entity { id("b9") startPosition(300, 600) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events {
                OnTapped("b9") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                    updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litB9") } } } } } } } }
                OnSignal("litB8") { Actions { whenVar("lit", "==", 0) { Actions {
                    changeVar("lit", 1) setColor("fbbf24") sound("pop") burst(10, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litB9") } } } } } } } }
        Entity { id("b10") startPosition(100, 600) color("3f3f46") Type { polygon(8, 30) } var("lit", 0)
            Events {
                OnTapped("b10") { Actions { whenVar("lit", "==", 0) { Actions { whenBar("sparks", ">", 0) { Actions {
                    updateBar("sparks", -1) changeVar("lit", 1) setColor("fbbf24") sound("sparkle") burst(14, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litB10") } } } } } } } }
                OnSignal("litB9") { Actions { whenVar("lit", "==", 0) { Actions {
                    changeVar("lit", 1) setColor("fbbf24") sound("pop") burst(10, "fde68a") updateBar("glow", 1)
                    delay(0.5) { Actions { broadcast("litB10") } } } } } } } }
    }
}
