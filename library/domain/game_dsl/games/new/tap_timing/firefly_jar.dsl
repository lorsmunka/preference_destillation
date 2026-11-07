# Firefly Jar: fireflies light up somewhere in the meadow and fade within a second.
# Tap the glowing ones before they go dark. Grey moths flutter up too: leave them be.
Window {
    size(10, 8)
    background("0b1026")
    music("bg_nature")

    Bar("caught") {
        initial(0)
        OnValue(5) { Actions { setOpacity("jar", 0.5) } }
        OnValue(10) { Actions { setOpacity("jar", 0.75) setText("sky", "Halfway there") } }
        OnValue(15) { Actions { setOpacity("jar", 1) } }
    }
    winAt("caught", 20, "-- A JAR FULL OF LIGHT --", "win")
    Bar("scares") {
        initial(0)
        OnValue(2) { Actions { setText("sky", "Careful, one more moth!") } }
        OnValue(3) { Actions { sound("lose") gameOver("-- THE MOTHS SCARED THEM OFF --") } }
    }

    Entity { id("jar") startPosition(900, 20) color("fde047") Type { rectangle(70, 100) }
        Events { OnElapsedTime { Actions { setOpacity(0.2) } } } }

    Entity { id("sky") startPosition(40, 730) color("94a3b8") Type { text("Catch 20 fireflies", 36) } }

    Entity { id("moon") startPosition(860, 640) color("e2e8f0") Type { circle(40) }
        Events { Sequence {
            Phase(1, 15) { Actions { setText("sky", "Midnight") } }
            Phase(1, 15) { Actions { setText("sky", "Almost dawn!") setColor("fdba74") sound("beep") } }
            Phase(1, 15) { Actions { sound("lose") gameOver("-- DAWN BROKE --") } }
        } } }

    Entity { id("fly1") startPosition(150, 560) color("fde047") Type { circle(14) } var("lit", 0)
        Events {
            OnElapsedTime { Actions { setVisible(false) } }
            OnElapsedTime(random(2, 5)) { repeatCount(-1) Actions { whenVar("lit", "==", 0) { Actions {
                setPosition("fly1", random(40, 300), random(450, 680), 0)
                setVisible(true) setOpacity(1)
                randomAction {
                    Actions { changeVar("lit", 1) setColor("fde047") }
                    Actions { changeVar("lit", 1) setColor("bef264") }
                    Actions { changeVar("lit", 1) setColor("fde047") }
                    Actions { changeVar("lit", 2) setColor("a8a29e") }
                }
                delay(0.5) { Actions { whenVar("lit", ">", 0) { Actions { setOpacity(0.6) } } } }
                delay(0.9) { Actions { whenVar("lit", ">", 0) { Actions { setOpacity(0.3) } } } }
                delay(1.3) { Actions {
                    whenVar("lit", "==", 2) { Actions { changeVar("lit", -2) setVisible(false) } }
                    whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setVisible(false) } }
                } }
            } } } }
            OnTapped("fly1") { Actions {
                whenVar("lit", "==", 2) { Actions { changeVar("lit", -2) setVisible(false) sound("hurt") shake(5, 0.3) updateBar("scares", 1) } }
                whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setVisible(false) sound("sparkle") burst(10, "fde047") updateBar("caught", 1) } }
            } }
        } }

    Entity { id("fly2") startPosition(480, 560) color("fde047") Type { circle(14) } var("lit", 0)
        Events {
            OnElapsedTime { Actions { setVisible(false) } }
            OnElapsedTime(random(2, 5)) { repeatCount(-1) Actions { whenVar("lit", "==", 0) { Actions {
                setPosition("fly2", random(360, 640), random(450, 680), 0)
                setVisible(true) setOpacity(1)
                randomAction {
                    Actions { changeVar("lit", 1) setColor("fde047") }
                    Actions { changeVar("lit", 1) setColor("bef264") }
                    Actions { changeVar("lit", 1) setColor("fde047") }
                    Actions { changeVar("lit", 2) setColor("a8a29e") }
                }
                delay(0.5) { Actions { whenVar("lit", ">", 0) { Actions { setOpacity(0.6) } } } }
                delay(0.9) { Actions { whenVar("lit", ">", 0) { Actions { setOpacity(0.3) } } } }
                delay(1.3) { Actions {
                    whenVar("lit", "==", 2) { Actions { changeVar("lit", -2) setVisible(false) } }
                    whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setVisible(false) } }
                } }
            } } } }
            OnTapped("fly2") { Actions {
                whenVar("lit", "==", 2) { Actions { changeVar("lit", -2) setVisible(false) sound("hurt") shake(5, 0.3) updateBar("scares", 1) } }
                whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setVisible(false) sound("sparkle") burst(10, "fde047") updateBar("caught", 1) } }
            } }
        } }

    Entity { id("fly3") startPosition(800, 560) color("fde047") Type { circle(14) } var("lit", 0)
        Events {
            OnElapsedTime { Actions { setVisible(false) } }
            OnElapsedTime(random(2, 5)) { repeatCount(-1) Actions { whenVar("lit", "==", 0) { Actions {
                setPosition("fly3", random(700, 820), random(420, 600), 0)
                setVisible(true) setOpacity(1)
                randomAction {
                    Actions { changeVar("lit", 1) setColor("fde047") }
                    Actions { changeVar("lit", 1) setColor("bef264") }
                    Actions { changeVar("lit", 1) setColor("fde047") }
                    Actions { changeVar("lit", 2) setColor("a8a29e") }
                }
                delay(0.5) { Actions { whenVar("lit", ">", 0) { Actions { setOpacity(0.6) } } } }
                delay(0.9) { Actions { whenVar("lit", ">", 0) { Actions { setOpacity(0.3) } } } }
                delay(1.3) { Actions {
                    whenVar("lit", "==", 2) { Actions { changeVar("lit", -2) setVisible(false) } }
                    whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setVisible(false) } }
                } }
            } } } }
            OnTapped("fly3") { Actions {
                whenVar("lit", "==", 2) { Actions { changeVar("lit", -2) setVisible(false) sound("hurt") shake(5, 0.3) updateBar("scares", 1) } }
                whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setVisible(false) sound("sparkle") burst(10, "fde047") updateBar("caught", 1) } }
            } }
        } }

    Entity { id("fly4") startPosition(250, 250) color("fde047") Type { circle(14) } var("lit", 0)
        Events {
            OnElapsedTime { Actions { setVisible(false) } }
            OnElapsedTime(random(2, 5)) { repeatCount(-1) Actions { whenVar("lit", "==", 0) { Actions {
                setPosition("fly4", random(60, 440), random(160, 380), 0)
                setVisible(true) setOpacity(1)
                randomAction {
                    Actions { changeVar("lit", 1) setColor("fde047") }
                    Actions { changeVar("lit", 1) setColor("bef264") }
                    Actions { changeVar("lit", 1) setColor("fde047") }
                    Actions { changeVar("lit", 2) setColor("a8a29e") }
                }
                delay(0.5) { Actions { whenVar("lit", ">", 0) { Actions { setOpacity(0.6) } } } }
                delay(0.9) { Actions { whenVar("lit", ">", 0) { Actions { setOpacity(0.3) } } } }
                delay(1.3) { Actions {
                    whenVar("lit", "==", 2) { Actions { changeVar("lit", -2) setVisible(false) } }
                    whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setVisible(false) } }
                } }
            } } } }
            OnTapped("fly4") { Actions {
                whenVar("lit", "==", 2) { Actions { changeVar("lit", -2) setVisible(false) sound("hurt") shake(5, 0.3) updateBar("scares", 1) } }
                whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setVisible(false) sound("sparkle") burst(10, "fde047") updateBar("caught", 1) } }
            } }
        } }

    Entity { id("fly5") startPosition(700, 250) color("fde047") Type { circle(14) } var("lit", 0)
        Events {
            OnElapsedTime { Actions { setVisible(false) } }
            OnElapsedTime(random(2, 5)) { repeatCount(-1) Actions { whenVar("lit", "==", 0) { Actions {
                setPosition("fly5", random(520, 840), random(160, 380), 0)
                setVisible(true) setOpacity(1)
                randomAction {
                    Actions { changeVar("lit", 1) setColor("fde047") }
                    Actions { changeVar("lit", 1) setColor("bef264") }
                    Actions { changeVar("lit", 1) setColor("fde047") }
                    Actions { changeVar("lit", 2) setColor("a8a29e") }
                }
                delay(0.5) { Actions { whenVar("lit", ">", 0) { Actions { setOpacity(0.6) } } } }
                delay(0.9) { Actions { whenVar("lit", ">", 0) { Actions { setOpacity(0.3) } } } }
                delay(1.3) { Actions {
                    whenVar("lit", "==", 2) { Actions { changeVar("lit", -2) setVisible(false) } }
                    whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setVisible(false) } }
                } }
            } } } }
            OnTapped("fly5") { Actions {
                whenVar("lit", "==", 2) { Actions { changeVar("lit", -2) setVisible(false) sound("hurt") shake(5, 0.3) updateBar("scares", 1) } }
                whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setVisible(false) sound("sparkle") burst(10, "fde047") updateBar("caught", 1) } }
            } }
        } }
}
