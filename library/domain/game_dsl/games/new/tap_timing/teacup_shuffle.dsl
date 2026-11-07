# Teacup Shuffle: the sugar cube goes under one of three teacups, then the cups swap places,
# one arcing over and one under. Tap the cup that hides the sugar. Each round swaps more, faster.
Window {
    size(8, 9)
    background("292524")
    music("bg_4")

    lives(3, "-- OUT OF TEA AND PATIENCE --", "lose")
    Bar("guess") { initial(0) }

    Level {
        Entity { id("cube") startPosition(385, 430) color("fafaf9") Type { rectangle(40, 40) } }
        Entity { id("cupA") startPosition(120, 420) color("e0f2fe") Type { rectangle(110, 130) } var("sugar", 0)
            Events {
                OnSignal("clear") { Actions { whenVar("sugar", "==", 1) { Actions { changeVar("sugar", -1) } } } }
                OnSignal("sugarA") { Actions { changeVar("sugar", 1) } }
                OnSignal("reveal") { Actions { whenVar("sugar", "==", 1) { Actions { flash("fde047", 0.8) burst(14, "ffffff") } } } }
                OnTapped("cupA") { Actions { whenBar("guess", "==", 1) { Actions {
                    updateBar("guess", -1)
                    whenVar("sugar", "==", 1) { Actions { burst(20, "fde047") broadcast("right") } }
                    whenVar("sugar", "==", 0) { Actions { flash("ef4444", 0.4) broadcast("wrong") broadcast("reveal") } }
                } } } }
            } }
        Entity { id("cupB") startPosition(350, 420) color("e0f2fe") Type { rectangle(110, 130) } var("sugar", 0)
            Events {
                OnSignal("clear") { Actions { whenVar("sugar", "==", 1) { Actions { changeVar("sugar", -1) } } } }
                OnSignal("sugarB") { Actions { changeVar("sugar", 1) } }
                OnSignal("reveal") { Actions { whenVar("sugar", "==", 1) { Actions { flash("fde047", 0.8) burst(14, "ffffff") } } } }
                OnTapped("cupB") { Actions { whenBar("guess", "==", 1) { Actions {
                    updateBar("guess", -1)
                    whenVar("sugar", "==", 1) { Actions { burst(20, "fde047") broadcast("right") } }
                    whenVar("sugar", "==", 0) { Actions { flash("ef4444", 0.4) broadcast("wrong") broadcast("reveal") } }
                } } } }
            } }
        Entity { id("cupC") startPosition(580, 420) color("e0f2fe") Type { rectangle(110, 130) } var("sugar", 0)
            Events {
                OnSignal("clear") { Actions { whenVar("sugar", "==", 1) { Actions { changeVar("sugar", -1) } } } }
                OnSignal("sugarC") { Actions { changeVar("sugar", 1) } }
                OnSignal("reveal") { Actions { whenVar("sugar", "==", 1) { Actions { flash("fde047", 0.8) burst(14, "ffffff") } } } }
                OnTapped("cupC") { Actions { whenBar("guess", "==", 1) { Actions {
                    updateBar("guess", -1)
                    whenVar("sugar", "==", 1) { Actions { burst(20, "fde047") broadcast("right") } }
                    whenVar("sugar", "==", 0) { Actions { flash("ef4444", 0.4) broadcast("wrong") broadcast("reveal") } }
                } } } }
            } }
        Entity { id("prompt") startPosition(40, 120) color("fef3c7") Type { text("Watch the sugar...", 40) } }
        Entity { id("host") startPosition(40, 820) color("fde68a") Type { text("Round 1 of 3", 44) }
            Events {
                OnElapsedTime(0.8) { Actions { broadcast("round") } }
                OnSignal("round") { Actions {
                    broadcast("clear")
                    setText("prompt", "Watch the sugar...")
                    setPosition("cupA", 120, 570, 0.4) setPosition("cupB", 350, 570, 0.4) setPosition("cupC", 580, 570, 0.4)
                    setVisible("cube", true)
                    randomAction {
                        Actions { broadcast("sugarA") setPosition("cube", 155, 430, 0) }
                        Actions { broadcast("sugarB") setPosition("cube", 385, 430, 0) }
                        Actions { broadcast("sugarC") setPosition("cube", 615, 430, 0) }
                    }
                    delay(1.6) { Actions { setPosition("cupA", 120, 420, 0.3) setPosition("cupB", 350, 420, 0.3) setPosition("cupC", 580, 420, 0.3) } }
                    delay(2) { Actions { setVisible("cube", false) setText("prompt", "Keep your eye on it!") } }
                    randomAction {
                        Actions {
                            delay(2.2) { Actions { setPosition("cupA", 235, 560, 0.35) setPosition("cupB", 235, 280, 0.35) } }
                            delay(2.55) { Actions { setPosition("cupA", 350, 420, 0.35) setPosition("cupB", 120, 420, 0.35) } }
                            delay(2.95) { Actions { setPosition("cupA", 465, 560, 0.35) setPosition("cupC", 465, 280, 0.35) } }
                            delay(3.3) { Actions { setPosition("cupA", 580, 420, 0.35) setPosition("cupC", 350, 420, 0.35) } }
                            delay(3.7) { Actions { setPosition("cupB", 350, 600, 0.35) setPosition("cupA", 350, 240, 0.35) } }
                            delay(4.05) { Actions { setPosition("cupB", 580, 420, 0.35) setPosition("cupA", 120, 420, 0.35) } }
                            delay(4.45) { Actions { setPosition("cupA", 235, 560, 0.35) setPosition("cupC", 235, 280, 0.35) } }
                            delay(4.8) { Actions { setPosition("cupA", 350, 420, 0.35) setPosition("cupC", 120, 420, 0.35) } }
                        }
                        Actions {
                            delay(2.2) { Actions { setPosition("cupB", 465, 560, 0.35) setPosition("cupC", 465, 280, 0.35) } }
                            delay(2.55) { Actions { setPosition("cupB", 580, 420, 0.35) setPosition("cupC", 350, 420, 0.35) } }
                            delay(2.95) { Actions { setPosition("cupA", 350, 600, 0.35) setPosition("cupB", 350, 240, 0.35) } }
                            delay(3.3) { Actions { setPosition("cupA", 580, 420, 0.35) setPosition("cupB", 120, 420, 0.35) } }
                            delay(3.7) { Actions { setPosition("cupB", 235, 560, 0.35) setPosition("cupC", 235, 280, 0.35) } }
                            delay(4.05) { Actions { setPosition("cupB", 350, 420, 0.35) setPosition("cupC", 120, 420, 0.35) } }
                            delay(4.45) { Actions { setPosition("cupB", 465, 560, 0.35) setPosition("cupA", 465, 280, 0.35) } }
                            delay(4.8) { Actions { setPosition("cupB", 580, 420, 0.35) setPosition("cupA", 350, 420, 0.35) } }
                        }
                    }
                    delay(5.4) { Actions { updateBar("guess", 1) sound("bell") setText("prompt", "Which cup hides the sugar?") } }
                } }
                OnSignal("right") { Actions { sound("win2") setText("prompt", "Sweet! Next round...") delay(1.5) { Actions { nextLevel() } } } }
                OnSignal("wrong") { Actions {
                    sound("buzz") shake(6, 0.4) updateBar("lives", -1)
                    setText("prompt", "Not that one! Again...")
                    delay(2) { Actions { broadcast("round") } }
                } }
            } }
    }

    Level {
        Entity { id("cube") startPosition(385, 430) color("fafaf9") Type { rectangle(40, 40) } }
        Entity { id("cupA") startPosition(120, 420) color("e0f2fe") Type { rectangle(110, 130) } var("sugar", 0)
            Events {
                OnSignal("clear") { Actions { whenVar("sugar", "==", 1) { Actions { changeVar("sugar", -1) } } } }
                OnSignal("sugarA") { Actions { changeVar("sugar", 1) } }
                OnSignal("reveal") { Actions { whenVar("sugar", "==", 1) { Actions { flash("fde047", 0.8) burst(14, "ffffff") } } } }
                OnTapped("cupA") { Actions { whenBar("guess", "==", 1) { Actions {
                    updateBar("guess", -1)
                    whenVar("sugar", "==", 1) { Actions { burst(20, "fde047") broadcast("right") } }
                    whenVar("sugar", "==", 0) { Actions { flash("ef4444", 0.4) broadcast("wrong") broadcast("reveal") } }
                } } } }
            } }
        Entity { id("cupB") startPosition(350, 420) color("e0f2fe") Type { rectangle(110, 130) } var("sugar", 0)
            Events {
                OnSignal("clear") { Actions { whenVar("sugar", "==", 1) { Actions { changeVar("sugar", -1) } } } }
                OnSignal("sugarB") { Actions { changeVar("sugar", 1) } }
                OnSignal("reveal") { Actions { whenVar("sugar", "==", 1) { Actions { flash("fde047", 0.8) burst(14, "ffffff") } } } }
                OnTapped("cupB") { Actions { whenBar("guess", "==", 1) { Actions {
                    updateBar("guess", -1)
                    whenVar("sugar", "==", 1) { Actions { burst(20, "fde047") broadcast("right") } }
                    whenVar("sugar", "==", 0) { Actions { flash("ef4444", 0.4) broadcast("wrong") broadcast("reveal") } }
                } } } }
            } }
        Entity { id("cupC") startPosition(580, 420) color("e0f2fe") Type { rectangle(110, 130) } var("sugar", 0)
            Events {
                OnSignal("clear") { Actions { whenVar("sugar", "==", 1) { Actions { changeVar("sugar", -1) } } } }
                OnSignal("sugarC") { Actions { changeVar("sugar", 1) } }
                OnSignal("reveal") { Actions { whenVar("sugar", "==", 1) { Actions { flash("fde047", 0.8) burst(14, "ffffff") } } } }
                OnTapped("cupC") { Actions { whenBar("guess", "==", 1) { Actions {
                    updateBar("guess", -1)
                    whenVar("sugar", "==", 1) { Actions { burst(20, "fde047") broadcast("right") } }
                    whenVar("sugar", "==", 0) { Actions { flash("ef4444", 0.4) broadcast("wrong") broadcast("reveal") } }
                } } } }
            } }
        Entity { id("prompt") startPosition(40, 120) color("fef3c7") Type { text("Watch the sugar...", 40) } }
        Entity { id("host") startPosition(40, 820) color("fde68a") Type { text("Round 2 of 3", 44) }
            Events {
                OnElapsedTime(0.8) { Actions { broadcast("round") } }
                OnSignal("round") { Actions {
                    broadcast("clear")
                    setText("prompt", "Watch the sugar...")
                    setPosition("cupA", 120, 570, 0.4) setPosition("cupB", 350, 570, 0.4) setPosition("cupC", 580, 570, 0.4)
                    setVisible("cube", true)
                    randomAction {
                        Actions { broadcast("sugarA") setPosition("cube", 155, 430, 0) }
                        Actions { broadcast("sugarB") setPosition("cube", 385, 430, 0) }
                        Actions { broadcast("sugarC") setPosition("cube", 615, 430, 0) }
                    }
                    delay(1.6) { Actions { setPosition("cupA", 120, 420, 0.3) setPosition("cupB", 350, 420, 0.3) setPosition("cupC", 580, 420, 0.3) } }
                    delay(2) { Actions { setVisible("cube", false) setText("prompt", "Keep your eye on it!") } }
                    randomAction {
                        Actions {
                            delay(2.2) { Actions { setPosition("cupA", 350, 600, 0.25) setPosition("cupC", 350, 240, 0.25) } }
                            delay(2.45) { Actions { setPosition("cupA", 580, 420, 0.25) setPosition("cupC", 120, 420, 0.25) } }
                            delay(2.75) { Actions { setPosition("cupC", 235, 560, 0.25) setPosition("cupB", 235, 280, 0.25) } }
                            delay(3) { Actions { setPosition("cupC", 350, 420, 0.25) setPosition("cupB", 120, 420, 0.25) } }
                            delay(3.3) { Actions { setPosition("cupC", 465, 560, 0.25) setPosition("cupA", 465, 280, 0.25) } }
                            delay(3.55) { Actions { setPosition("cupC", 580, 420, 0.25) setPosition("cupA", 350, 420, 0.25) } }
                            delay(3.85) { Actions { setPosition("cupB", 235, 560, 0.25) setPosition("cupA", 235, 280, 0.25) } }
                            delay(4.1) { Actions { setPosition("cupB", 350, 420, 0.25) setPosition("cupA", 120, 420, 0.25) } }
                            delay(4.4) { Actions { setPosition("cupA", 350, 600, 0.25) setPosition("cupC", 350, 240, 0.25) } }
                            delay(4.65) { Actions { setPosition("cupA", 580, 420, 0.25) setPosition("cupC", 120, 420, 0.25) } }
                            delay(4.95) { Actions { setPosition("cupB", 465, 560, 0.25) setPosition("cupA", 465, 280, 0.25) } }
                            delay(5.2) { Actions { setPosition("cupB", 580, 420, 0.25) setPosition("cupA", 350, 420, 0.25) } }
                        }
                        Actions {
                            delay(2.2) { Actions { setPosition("cupB", 465, 560, 0.25) setPosition("cupC", 465, 280, 0.25) } }
                            delay(2.45) { Actions { setPosition("cupB", 580, 420, 0.25) setPosition("cupC", 350, 420, 0.25) } }
                            delay(2.75) { Actions { setPosition("cupA", 235, 560, 0.25) setPosition("cupC", 235, 280, 0.25) } }
                            delay(3) { Actions { setPosition("cupA", 350, 420, 0.25) setPosition("cupC", 120, 420, 0.25) } }
                            delay(3.3) { Actions { setPosition("cupC", 350, 600, 0.25) setPosition("cupB", 350, 240, 0.25) } }
                            delay(3.55) { Actions { setPosition("cupC", 580, 420, 0.25) setPosition("cupB", 120, 420, 0.25) } }
                            delay(3.85) { Actions { setPosition("cupA", 465, 560, 0.25) setPosition("cupC", 465, 280, 0.25) } }
                            delay(4.1) { Actions { setPosition("cupA", 580, 420, 0.25) setPosition("cupC", 350, 420, 0.25) } }
                            delay(4.4) { Actions { setPosition("cupB", 235, 560, 0.25) setPosition("cupC", 235, 280, 0.25) } }
                            delay(4.65) { Actions { setPosition("cupB", 350, 420, 0.25) setPosition("cupC", 120, 420, 0.25) } }
                            delay(4.95) { Actions { setPosition("cupC", 350, 600, 0.25) setPosition("cupA", 350, 240, 0.25) } }
                            delay(5.2) { Actions { setPosition("cupC", 580, 420, 0.25) setPosition("cupA", 120, 420, 0.25) } }
                        }
                    }
                    delay(5.7) { Actions { updateBar("guess", 1) sound("bell") setText("prompt", "Which cup hides the sugar?") } }
                } }
                OnSignal("right") { Actions { sound("win2") setText("prompt", "Sweet! Next round...") delay(1.5) { Actions { nextLevel() } } } }
                OnSignal("wrong") { Actions {
                    sound("buzz") shake(6, 0.4) updateBar("lives", -1)
                    setText("prompt", "Not that one! Again...")
                    delay(2) { Actions { broadcast("round") } }
                } }
            } }
    }

    Level {
        Entity { id("cube") startPosition(385, 430) color("fafaf9") Type { rectangle(40, 40) } }
        Entity { id("cupA") startPosition(120, 420) color("e0f2fe") Type { rectangle(110, 130) } var("sugar", 0)
            Events {
                OnSignal("clear") { Actions { whenVar("sugar", "==", 1) { Actions { changeVar("sugar", -1) } } } }
                OnSignal("sugarA") { Actions { changeVar("sugar", 1) } }
                OnSignal("reveal") { Actions { whenVar("sugar", "==", 1) { Actions { flash("fde047", 0.8) burst(14, "ffffff") } } } }
                OnTapped("cupA") { Actions { whenBar("guess", "==", 1) { Actions {
                    updateBar("guess", -1)
                    whenVar("sugar", "==", 1) { Actions { burst(20, "fde047") broadcast("right") } }
                    whenVar("sugar", "==", 0) { Actions { flash("ef4444", 0.4) broadcast("wrong") broadcast("reveal") } }
                } } } }
            } }
        Entity { id("cupB") startPosition(350, 420) color("e0f2fe") Type { rectangle(110, 130) } var("sugar", 0)
            Events {
                OnSignal("clear") { Actions { whenVar("sugar", "==", 1) { Actions { changeVar("sugar", -1) } } } }
                OnSignal("sugarB") { Actions { changeVar("sugar", 1) } }
                OnSignal("reveal") { Actions { whenVar("sugar", "==", 1) { Actions { flash("fde047", 0.8) burst(14, "ffffff") } } } }
                OnTapped("cupB") { Actions { whenBar("guess", "==", 1) { Actions {
                    updateBar("guess", -1)
                    whenVar("sugar", "==", 1) { Actions { burst(20, "fde047") broadcast("right") } }
                    whenVar("sugar", "==", 0) { Actions { flash("ef4444", 0.4) broadcast("wrong") broadcast("reveal") } }
                } } } }
            } }
        Entity { id("cupC") startPosition(580, 420) color("e0f2fe") Type { rectangle(110, 130) } var("sugar", 0)
            Events {
                OnSignal("clear") { Actions { whenVar("sugar", "==", 1) { Actions { changeVar("sugar", -1) } } } }
                OnSignal("sugarC") { Actions { changeVar("sugar", 1) } }
                OnSignal("reveal") { Actions { whenVar("sugar", "==", 1) { Actions { flash("fde047", 0.8) burst(14, "ffffff") } } } }
                OnTapped("cupC") { Actions { whenBar("guess", "==", 1) { Actions {
                    updateBar("guess", -1)
                    whenVar("sugar", "==", 1) { Actions { burst(20, "fde047") broadcast("right") } }
                    whenVar("sugar", "==", 0) { Actions { flash("ef4444", 0.4) broadcast("wrong") broadcast("reveal") } }
                } } } }
            } }
        Entity { id("prompt") startPosition(40, 120) color("fef3c7") Type { text("Watch the sugar...", 40) } }
        Entity { id("host") startPosition(40, 820) color("fde68a") Type { text("Final round", 44) }
            Events {
                OnElapsedTime(0.8) { Actions { broadcast("round") } }
                OnSignal("round") { Actions {
                    broadcast("clear")
                    setText("prompt", "Watch the sugar...")
                    setPosition("cupA", 120, 570, 0.4) setPosition("cupB", 350, 570, 0.4) setPosition("cupC", 580, 570, 0.4)
                    setVisible("cube", true)
                    randomAction {
                        Actions { broadcast("sugarA") setPosition("cube", 155, 430, 0) }
                        Actions { broadcast("sugarB") setPosition("cube", 385, 430, 0) }
                        Actions { broadcast("sugarC") setPosition("cube", 615, 430, 0) }
                    }
                    delay(1.6) { Actions { setPosition("cupA", 120, 420, 0.3) setPosition("cupB", 350, 420, 0.3) setPosition("cupC", 580, 420, 0.3) } }
                    delay(2) { Actions { setVisible("cube", false) setText("prompt", "Keep your eye on it!") } }
                    randomAction {
                        Actions {
                            delay(2.2) { Actions { setPosition("cupA", 235, 560, 0.18) setPosition("cupB", 235, 280, 0.18) } }
                            delay(2.38) { Actions { setPosition("cupA", 350, 420, 0.18) setPosition("cupB", 120, 420, 0.18) } }
                            delay(2.61) { Actions { setPosition("cupB", 350, 600, 0.18) setPosition("cupC", 350, 240, 0.18) } }
                            delay(2.79) { Actions { setPosition("cupB", 580, 420, 0.18) setPosition("cupC", 120, 420, 0.18) } }
                            delay(3.02) { Actions { setPosition("cupA", 465, 560, 0.18) setPosition("cupB", 465, 280, 0.18) } }
                            delay(3.2) { Actions { setPosition("cupA", 580, 420, 0.18) setPosition("cupB", 350, 420, 0.18) } }
                            delay(3.43) { Actions { setPosition("cupC", 235, 560, 0.18) setPosition("cupB", 235, 280, 0.18) } }
                            delay(3.61) { Actions { setPosition("cupC", 350, 420, 0.18) setPosition("cupB", 120, 420, 0.18) } }
                            delay(3.84) { Actions { setPosition("cupC", 465, 560, 0.18) setPosition("cupA", 465, 280, 0.18) } }
                            delay(4.02) { Actions { setPosition("cupC", 580, 420, 0.18) setPosition("cupA", 350, 420, 0.18) } }
                            delay(4.25) { Actions { setPosition("cupB", 350, 600, 0.18) setPosition("cupC", 350, 240, 0.18) } }
                            delay(4.43) { Actions { setPosition("cupB", 580, 420, 0.18) setPosition("cupC", 120, 420, 0.18) } }
                            delay(4.66) { Actions { setPosition("cupC", 235, 560, 0.18) setPosition("cupA", 235, 280, 0.18) } }
                            delay(4.84) { Actions { setPosition("cupC", 350, 420, 0.18) setPosition("cupA", 120, 420, 0.18) } }
                            delay(5.07) { Actions { setPosition("cupC", 465, 560, 0.18) setPosition("cupB", 465, 280, 0.18) } }
                            delay(5.25) { Actions { setPosition("cupC", 580, 420, 0.18) setPosition("cupB", 350, 420, 0.18) } }
                        }
                        Actions {
                            delay(2.2) { Actions { setPosition("cupA", 350, 600, 0.18) setPosition("cupC", 350, 240, 0.18) } }
                            delay(2.38) { Actions { setPosition("cupA", 580, 420, 0.18) setPosition("cupC", 120, 420, 0.18) } }
                            delay(2.61) { Actions { setPosition("cupB", 465, 560, 0.18) setPosition("cupA", 465, 280, 0.18) } }
                            delay(2.79) { Actions { setPosition("cupB", 580, 420, 0.18) setPosition("cupA", 350, 420, 0.18) } }
                            delay(3.02) { Actions { setPosition("cupC", 235, 560, 0.18) setPosition("cupA", 235, 280, 0.18) } }
                            delay(3.2) { Actions { setPosition("cupC", 350, 420, 0.18) setPosition("cupA", 120, 420, 0.18) } }
                            delay(3.43) { Actions { setPosition("cupA", 350, 600, 0.18) setPosition("cupB", 350, 240, 0.18) } }
                            delay(3.61) { Actions { setPosition("cupA", 580, 420, 0.18) setPosition("cupB", 120, 420, 0.18) } }
                            delay(3.84) { Actions { setPosition("cupC", 465, 560, 0.18) setPosition("cupA", 465, 280, 0.18) } }
                            delay(4.02) { Actions { setPosition("cupC", 580, 420, 0.18) setPosition("cupA", 350, 420, 0.18) } }
                            delay(4.25) { Actions { setPosition("cupB", 235, 560, 0.18) setPosition("cupA", 235, 280, 0.18) } }
                            delay(4.43) { Actions { setPosition("cupB", 350, 420, 0.18) setPosition("cupA", 120, 420, 0.18) } }
                            delay(4.66) { Actions { setPosition("cupA", 350, 600, 0.18) setPosition("cupC", 350, 240, 0.18) } }
                            delay(4.84) { Actions { setPosition("cupA", 580, 420, 0.18) setPosition("cupC", 120, 420, 0.18) } }
                            delay(5.07) { Actions { setPosition("cupC", 235, 560, 0.18) setPosition("cupB", 235, 280, 0.18) } }
                            delay(5.25) { Actions { setPosition("cupC", 350, 420, 0.18) setPosition("cupB", 120, 420, 0.18) } }
                        }
                    }
                    delay(5.68) { Actions { updateBar("guess", 1) sound("bell") setText("prompt", "Which cup hides the sugar?") } }
                } }
                OnSignal("right") { Actions { sound("win") shake(5, 0.5) gameOver("-- THE SHARPEST EYES IN THE TEA HOUSE --") } }
                OnSignal("wrong") { Actions {
                    sound("buzz") shake(6, 0.4) updateBar("lives", -1)
                    setText("prompt", "Not that one! Again...")
                    delay(2) { Actions { broadcast("round") } }
                } }
            } }
    }
}
