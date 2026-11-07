# Night Bakery: three nights in a bakery, each with more ovens. Tap an empty oven to load dough;
# when it turns golden, tap it again to take the loaf out before it burns. Every loaf takes a scoop of flour:
# tap the sack to refill. Three burnt loaves in one night fill the kitchen with smoke and the night starts over;
# if dawn comes before the quota, the bakery has failed. On the second night the shop bell sometimes rings for a
# rush order (the next loaf counts twice); on the third the fuses blow and an oven goes dark for a moment.
Window {
    size(10, 12)
    background("2a1a10")
    music("bg_2")
    gravity(0)

    Bar("loaves") { initial(0)
        OnValue(6) { Actions { sound("win2") shake(4, 0.4) nextLevel() } }
        OnValue(16) { Actions { sound("win2") shake(4, 0.4) nextLevel() } }
        OnValue(30) { Actions { sound("win") shake(6, 0.6) gameOver("-- THREE NIGHTS, NOT ONE EMPTY SHELF --") } } }
    Bar("smoke") { initial(0)
        OnValue(3) { Actions { broadcast("smokedOut") sound("lose") shake(8, 0.6) delay(1.5) { Actions { restart() } } } } }
    Bar("flour") { initial(6)
        OnValue(0) { Actions { broadcast("noFlour") } } }
    Bar("rush") { initial(0)
        OnValue(1) { Actions { broadcast("rushOn") } }
        OnValue(0) { Actions { broadcast("rushOff") } } }
    Bar("clock") { initial(0)
        OnValue(45) { Actions { broadcast("dawnSoon") sound("bell") } }
        OnValue(60) { Actions { sound("lose") gameOver("-- DAWN CAME AND THE SHELVES WERE EMPTY --") } } }

    Level {
        Entity { id("night1") startPosition(40, 1120) color("fde68a") Type { text("NIGHT 1: bake 6 loaves before dawn", 34) }
            Events {
                OnElapsedTime { Actions { ramp("clock", 0, 60, 75) loopCount(3) whenBar("smoke", ">", 0) { Actions { updateBar("smoke", -1) } } } }
                OnSignal("dawnSoon") { Actions { setText("night1", "Dawn is close - keep the ovens busy!") setColor("fb923c") } }
                OnSignal("smokedOut") { Actions { setText("night1", "Too much smoke! The night starts over.") setColor("ef4444") } }
            } }
        Entity { id("sack1") startPosition(40, 120) color("e7e5e4") Type { rectangle(110, 130) } var("cool", 0)
            Events {
                OnTapped("sack1") { Actions { whenVar("cool", "==", 0) { Actions {
                    changeVar("cool", 1) sound("thud") burst(10, "f5f5f4") setOpacity(0.5)
                    delay(2.5) { Actions { changeVar("cool", -1) setOpacity(1) } }
                    loopCount(3) updateBar("flour", 1) } } } }
                OnSignal("noFlour") { Actions { flash("ef4444", 0.5) sound("beep") } }
            } }

        Entity { id("ovenA") startPosition(150, 500) color("44403c") Type { rectangle(200, 150) } var("stage", 0)
            Events { OnTapped("ovenA") { Actions {
                whenVar("stage", "==", 0) { Actions {
                    whenBar("flour", "==", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                    whenBar("flour", ">", 0) { Actions { updateBar("flour", -1) changeVar("stage", 1) setColor("fde68a") sound("click")
                        delay(4) { Actions { whenVar("stage", "==", 1) { Actions {
                            changeVar("stage", 1) setColor("d97706") sound("bell") burst(8, "fbbf24")
                            delay(3) { Actions { whenVar("stage", "==", 2) { Actions {
                                changeVar("stage", 1) setColor("1c1917") sound("hurt") burst(16, "78716c") shake(4, 0.3) updateBar("smoke", 1)
                                delay(1.5) { Actions { whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") } } } }
                            } } } }
                        } } } }
                    } }
                } }
                whenVar("stage", "==", 2) { Actions { changeVar("stage", -2) setColor("44403c") sound("coin") burst(12, "fbbf24") updateBar("loaves", 1)
                    whenBar("rush", "==", 1) { Actions { updateBar("rush", -1) updateBar("loaves", 1) burst(20, "fde047") } } } }
                whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") sound("pop") } }
            } } } }
        Entity { id("ovenB") startPosition(650, 500) color("44403c") Type { rectangle(200, 150) } var("stage", 0)
            Events { OnTapped("ovenB") { Actions {
                whenVar("stage", "==", 0) { Actions {
                    whenBar("flour", "==", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                    whenBar("flour", ">", 0) { Actions { updateBar("flour", -1) changeVar("stage", 1) setColor("fde68a") sound("click")
                        delay(4) { Actions { whenVar("stage", "==", 1) { Actions {
                            changeVar("stage", 1) setColor("d97706") sound("bell") burst(8, "fbbf24")
                            delay(3) { Actions { whenVar("stage", "==", 2) { Actions {
                                changeVar("stage", 1) setColor("1c1917") sound("hurt") burst(16, "78716c") shake(4, 0.3) updateBar("smoke", 1)
                                delay(1.5) { Actions { whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") } } } }
                            } } } }
                        } } } }
                    } }
                } }
                whenVar("stage", "==", 2) { Actions { changeVar("stage", -2) setColor("44403c") sound("coin") burst(12, "fbbf24") updateBar("loaves", 1)
                    whenBar("rush", "==", 1) { Actions { updateBar("rush", -1) updateBar("loaves", 1) burst(20, "fde047") } } } }
                whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") sound("pop") } }
            } } } }
    }

    Level {
        Entity { id("night2") startPosition(40, 1120) color("fde68a") Type { text("NIGHT 2: 10 more, and the old oven is slow", 34) }
            Events {
                OnElapsedTime { Actions { ramp("clock", 0, 60, 75) loopCount(3) whenBar("smoke", ">", 0) { Actions { updateBar("smoke", -1) } } } }
                OnSignal("dawnSoon") { Actions { setText("night2", "Dawn is close - keep the ovens busy!") setColor("fb923c") } }
                OnSignal("smokedOut") { Actions { setText("night2", "Too much smoke! The night starts over.") setColor("ef4444") } }
            } }
        Entity { id("sack2") startPosition(40, 120) color("e7e5e4") Type { rectangle(110, 130) } var("cool", 0)
            Events {
                OnTapped("sack2") { Actions { whenVar("cool", "==", 0) { Actions {
                    changeVar("cool", 1) sound("thud") burst(10, "f5f5f4") setOpacity(0.5)
                    delay(2.5) { Actions { changeVar("cool", -1) setOpacity(1) } }
                    loopCount(3) updateBar("flour", 1) } } } }
                OnSignal("noFlour") { Actions { flash("ef4444", 0.5) sound("beep") } }
            } }
        Entity { id("shopBell") startPosition(460, 900) color("78716c") Type { star(6, 40, 22) }
            Events {
                Sequence { Phase(-1, 12) { Actions { whenBar("rush", "==", 0) { Actions { randomAction {
                    Actions { updateBar("rush", 1) sound("bell") }
                    Actions { } } } } } } }
                OnSignal("rushOn") { Actions { setColor("fde047") flash("ffffff", 0.4) burst(12, "fde047") } }
                OnSignal("rushOff") { Actions { setColor("78716c") } }
            } }

        Entity { id("ovenA") startPosition(60, 500) color("44403c") Type { rectangle(200, 150) } var("stage", 0)
            Events { OnTapped("ovenA") { Actions {
                whenVar("stage", "==", 0) { Actions {
                    whenBar("flour", "==", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                    whenBar("flour", ">", 0) { Actions { updateBar("flour", -1) changeVar("stage", 1) setColor("fde68a") sound("click")
                        delay(5) { Actions { whenVar("stage", "==", 1) { Actions {
                            changeVar("stage", 1) setColor("d97706") sound("bell") burst(8, "fbbf24")
                            delay(3) { Actions { whenVar("stage", "==", 2) { Actions {
                                changeVar("stage", 1) setColor("1c1917") sound("hurt") burst(16, "78716c") shake(4, 0.3) updateBar("smoke", 1)
                                delay(1.5) { Actions { whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") } } } }
                            } } } }
                        } } } }
                    } }
                } }
                whenVar("stage", "==", 2) { Actions { changeVar("stage", -2) setColor("44403c") sound("coin") burst(12, "fbbf24") updateBar("loaves", 1)
                    whenBar("rush", "==", 1) { Actions { updateBar("rush", -1) updateBar("loaves", 1) burst(20, "fde047") } } } }
                whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") sound("pop") } }
            } } } }
        Entity { id("ovenB") startPosition(400, 500) color("44403c") Type { rectangle(200, 150) } var("stage", 0)
            Events { OnTapped("ovenB") { Actions {
                whenVar("stage", "==", 0) { Actions {
                    whenBar("flour", "==", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                    whenBar("flour", ">", 0) { Actions { updateBar("flour", -1) changeVar("stage", 1) setColor("fde68a") sound("click")
                        delay(3) { Actions { whenVar("stage", "==", 1) { Actions {
                            changeVar("stage", 1) setColor("d97706") sound("bell") burst(8, "fbbf24")
                            delay(2.5) { Actions { whenVar("stage", "==", 2) { Actions {
                                changeVar("stage", 1) setColor("1c1917") sound("hurt") burst(16, "78716c") shake(4, 0.3) updateBar("smoke", 1)
                                delay(1.5) { Actions { whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") } } } }
                            } } } }
                        } } } }
                    } }
                } }
                whenVar("stage", "==", 2) { Actions { changeVar("stage", -2) setColor("44403c") sound("coin") burst(12, "fbbf24") updateBar("loaves", 1)
                    whenBar("rush", "==", 1) { Actions { updateBar("rush", -1) updateBar("loaves", 1) burst(20, "fde047") } } } }
                whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") sound("pop") } }
            } } } }
        Entity { id("ovenC") startPosition(740, 500) color("44403c") Type { rectangle(200, 150) } var("stage", 0)
            Events { OnTapped("ovenC") { Actions {
                whenVar("stage", "==", 0) { Actions {
                    whenBar("flour", "==", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                    whenBar("flour", ">", 0) { Actions { updateBar("flour", -1) changeVar("stage", 1) setColor("fde68a") sound("click")
                        delay(4) { Actions { whenVar("stage", "==", 1) { Actions {
                            changeVar("stage", 1) setColor("d97706") sound("bell") burst(8, "fbbf24")
                            delay(3) { Actions { whenVar("stage", "==", 2) { Actions {
                                changeVar("stage", 1) setColor("1c1917") sound("hurt") burst(16, "78716c") shake(4, 0.3) updateBar("smoke", 1)
                                delay(1.5) { Actions { whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") } } } }
                            } } } }
                        } } } }
                    } }
                } }
                whenVar("stage", "==", 2) { Actions { changeVar("stage", -2) setColor("44403c") sound("coin") burst(12, "fbbf24") updateBar("loaves", 1)
                    whenBar("rush", "==", 1) { Actions { updateBar("rush", -1) updateBar("loaves", 1) burst(20, "fde047") } } } }
                whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") sound("pop") } }
            } } } }
    }

    Level {
        Entity { id("night3") startPosition(40, 1120) color("fde68a") Type { text("NIGHT 3: 14 more, mind the fuses", 34) }
            Events {
                OnElapsedTime { Actions { ramp("clock", 0, 60, 75) loopCount(3) whenBar("smoke", ">", 0) { Actions { updateBar("smoke", -1) } } } }
                OnSignal("dawnSoon") { Actions { setText("night3", "Dawn is close - keep the ovens busy!") setColor("fb923c") } }
                OnSignal("smokedOut") { Actions { setText("night3", "Too much smoke! The night starts over.") setColor("ef4444") } }
            } }
        Entity { id("sack3") startPosition(40, 120) color("e7e5e4") Type { rectangle(110, 130) } var("cool", 0)
            Events {
                OnTapped("sack3") { Actions { whenVar("cool", "==", 0) { Actions {
                    changeVar("cool", 1) sound("thud") burst(10, "f5f5f4") setOpacity(0.5)
                    delay(2.5) { Actions { changeVar("cool", -1) setOpacity(1) } }
                    loopCount(3) updateBar("flour", 1) } } } }
                OnSignal("noFlour") { Actions { flash("ef4444", 0.5) sound("beep") } }
            } }
        Entity { id("hearthA") startPosition(140, 690) color("57534e") Type { rectangle(220, 170) } }
        Entity { id("hearthB") startPosition(640, 690) color("57534e") Type { rectangle(220, 170) } }
        Entity { id("hearthC") startPosition(140, 340) color("57534e") Type { rectangle(220, 170) } }
        Entity { id("hearthD") startPosition(640, 340) color("57534e") Type { rectangle(220, 170) } }
        Entity { id("fusebox") startPosition(440, 560) color("a1a1aa") Type { rectangle(120, 80) }
            Events { OnElapsedTime(random(7, 11)) { repeatCount(-1) Actions { sound("buzz") flash("fde047", 0.3) randomAction {
                Actions { setEnable("ovenA", false) flash("hearthA", "ef4444", 1.5) delay(1.5) { Actions { setEnable("ovenA", true) } } }
                Actions { setEnable("ovenB", false) flash("hearthB", "ef4444", 1.5) delay(1.5) { Actions { setEnable("ovenB", true) } } }
                Actions { setEnable("ovenC", false) flash("hearthC", "ef4444", 1.5) delay(1.5) { Actions { setEnable("ovenC", true) } } }
                Actions { setEnable("ovenD", false) flash("hearthD", "ef4444", 1.5) delay(1.5) { Actions { setEnable("ovenD", true) } } }
            } } } } }

        Entity { id("ovenA") startPosition(150, 700) color("44403c") Type { rectangle(200, 150) } var("stage", 0)
            Events { OnTapped("ovenA") { Actions {
                whenVar("stage", "==", 0) { Actions {
                    whenBar("flour", "==", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                    whenBar("flour", ">", 0) { Actions { updateBar("flour", -1) changeVar("stage", 1) setColor("fde68a") sound("click")
                        delay(3) { Actions { whenVar("stage", "==", 1) { Actions {
                            changeVar("stage", 1) setColor("d97706") sound("bell") burst(8, "fbbf24")
                            delay(2) { Actions { whenVar("stage", "==", 2) { Actions {
                                changeVar("stage", 1) setColor("1c1917") sound("hurt") burst(16, "78716c") shake(4, 0.3) updateBar("smoke", 1)
                                delay(1.5) { Actions { whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") } } } }
                            } } } }
                        } } } }
                    } }
                } }
                whenVar("stage", "==", 2) { Actions { changeVar("stage", -2) setColor("44403c") sound("coin") burst(12, "fbbf24") updateBar("loaves", 1)
                    whenBar("rush", "==", 1) { Actions { updateBar("rush", -1) updateBar("loaves", 1) burst(20, "fde047") } } } }
                whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") sound("pop") } }
            } } } }
        Entity { id("ovenB") startPosition(650, 700) color("44403c") Type { rectangle(200, 150) } var("stage", 0)
            Events { OnTapped("ovenB") { Actions {
                whenVar("stage", "==", 0) { Actions {
                    whenBar("flour", "==", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                    whenBar("flour", ">", 0) { Actions { updateBar("flour", -1) changeVar("stage", 1) setColor("fde68a") sound("click")
                        delay(3) { Actions { whenVar("stage", "==", 1) { Actions {
                            changeVar("stage", 1) setColor("d97706") sound("bell") burst(8, "fbbf24")
                            delay(2) { Actions { whenVar("stage", "==", 2) { Actions {
                                changeVar("stage", 1) setColor("1c1917") sound("hurt") burst(16, "78716c") shake(4, 0.3) updateBar("smoke", 1)
                                delay(1.5) { Actions { whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") } } } }
                            } } } }
                        } } } }
                    } }
                } }
                whenVar("stage", "==", 2) { Actions { changeVar("stage", -2) setColor("44403c") sound("coin") burst(12, "fbbf24") updateBar("loaves", 1)
                    whenBar("rush", "==", 1) { Actions { updateBar("rush", -1) updateBar("loaves", 1) burst(20, "fde047") } } } }
                whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") sound("pop") } }
            } } } }
        Entity { id("ovenC") startPosition(150, 350) color("44403c") Type { rectangle(200, 150) } var("stage", 0)
            Events { OnTapped("ovenC") { Actions {
                whenVar("stage", "==", 0) { Actions {
                    whenBar("flour", "==", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                    whenBar("flour", ">", 0) { Actions { updateBar("flour", -1) changeVar("stage", 1) setColor("fde68a") sound("click")
                        delay(3) { Actions { whenVar("stage", "==", 1) { Actions {
                            changeVar("stage", 1) setColor("d97706") sound("bell") burst(8, "fbbf24")
                            delay(2) { Actions { whenVar("stage", "==", 2) { Actions {
                                changeVar("stage", 1) setColor("1c1917") sound("hurt") burst(16, "78716c") shake(4, 0.3) updateBar("smoke", 1)
                                delay(1.5) { Actions { whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") } } } }
                            } } } }
                        } } } }
                    } }
                } }
                whenVar("stage", "==", 2) { Actions { changeVar("stage", -2) setColor("44403c") sound("coin") burst(12, "fbbf24") updateBar("loaves", 1)
                    whenBar("rush", "==", 1) { Actions { updateBar("rush", -1) updateBar("loaves", 1) burst(20, "fde047") } } } }
                whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") sound("pop") } }
            } } } }
        Entity { id("ovenD") startPosition(650, 350) color("44403c") Type { rectangle(200, 150) } var("stage", 0)
            Events { OnTapped("ovenD") { Actions {
                whenVar("stage", "==", 0) { Actions {
                    whenBar("flour", "==", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                    whenBar("flour", ">", 0) { Actions { updateBar("flour", -1) changeVar("stage", 1) setColor("fde68a") sound("click")
                        delay(3) { Actions { whenVar("stage", "==", 1) { Actions {
                            changeVar("stage", 1) setColor("d97706") sound("bell") burst(8, "fbbf24")
                            delay(2) { Actions { whenVar("stage", "==", 2) { Actions {
                                changeVar("stage", 1) setColor("1c1917") sound("hurt") burst(16, "78716c") shake(4, 0.3) updateBar("smoke", 1)
                                delay(1.5) { Actions { whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") } } } }
                            } } } }
                        } } } }
                    } }
                } }
                whenVar("stage", "==", 2) { Actions { changeVar("stage", -2) setColor("44403c") sound("coin") burst(12, "fbbf24") updateBar("loaves", 1)
                    whenBar("rush", "==", 1) { Actions { updateBar("rush", -1) updateBar("loaves", 1) burst(20, "fde047") } } } }
                whenVar("stage", "==", 3) { Actions { changeVar("stage", -3) setColor("44403c") sound("pop") } }
            } } } }
    }
}
