# Circuit board. Tap GO and the spark runs along the traces; each switch turns it the way its
# arrow points (tap a switch to turn the arrow a quarter clockwise). The spark keeps running
# while you re-point switches. Touching a resistor or the board edge shorts it: a life is
# lost and every LED goes dark again. Light all six LEDs in a single run.
Window {
    size(8, 10)
    background("064e3b")
    music("bg_6")
    gravity(0)

    Bar("leds") { initial(0)
        OnValue(6) { Actions { sound("win") shake(5, 0.5) burst("title", 30, "fde047") gameOver("-- THE BOARD LIGHTS UP --") } } }
    Bar("running") { initial(0) }
    lives(3, "-- BURNT OUT --", "lose")

    Entity { id("edgeS") startPosition(0, 0) color("022c22") Type { rectangle(800, 100) } solid(true) }
    Entity { id("edgeN") startPosition(0, 900) color("022c22") Type { rectangle(800, 100) } solid(true) }
    Entity { id("title") startPosition(20, 930) color("a7f3d0") Type { text("LIGHT ALL SIX LEDS", 36) } }

    Entity { id("go") startPosition(360, 25) color("fde047") Type { text("GO", 50) }
        Events { OnTapped("go") { Actions { whenBar("running", "==", 0) { Actions {
            sound("beep") updateBar("running", 1) setDirection("spark", 1, 0) } } } } } }

    # Resistors
    Entity { id("res1") startPosition(410, 110) color("b45309") Type { rectangle(80, 80) } solid(true) }
    Entity { id("res2") startPosition(210, 310) color("b45309") Type { rectangle(80, 80) } solid(true) }
    Entity { id("res3") startPosition(210, 510) color("b45309") Type { rectangle(80, 80) } solid(true) }
    Entity { id("res4") startPosition(710, 510) color("b45309") Type { rectangle(80, 80) } solid(true) }
    Entity { id("res5") startPosition(610, 210) color("b45309") Type { rectangle(80, 80) } solid(true) }
    Entity { id("res6") startPosition(410, 810) color("b45309") Type { rectangle(80, 80) } solid(true) }

    # LEDs
    Entity { id("led1") startPosition(132, 132) color("365314") Type { circle(18) } var("lit", 0)
        Events { OnCollision("spark") { Actions { whenVar("lit", "==", 0) { Actions { changeVar("lit", 1) setColor("fde047") sound("pickup") burst(8, "fde047") updateBar("leds", 1) } } } }
            OnSignal("blackout") { Actions { whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("365314") updateBar("leds", -1) } } } } } }
    Entity { id("led2") startPosition(332, 432) color("365314") Type { circle(18) } var("lit", 0)
        Events { OnCollision("spark") { Actions { whenVar("lit", "==", 0) { Actions { changeVar("lit", 1) setColor("fde047") sound("pickup") burst(8, "fde047") updateBar("leds", 1) } } } }
            OnSignal("blackout") { Actions { whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("365314") updateBar("leds", -1) } } } } } }
    Entity { id("led3") startPosition(532, 532) color("365314") Type { circle(18) } var("lit", 0)
        Events { OnCollision("spark") { Actions { whenVar("lit", "==", 0) { Actions { changeVar("lit", 1) setColor("fde047") sound("pickup") burst(8, "fde047") updateBar("leds", 1) } } } }
            OnSignal("blackout") { Actions { whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("365314") updateBar("leds", -1) } } } } } }
    Entity { id("led4") startPosition(632, 432) color("365314") Type { circle(18) } var("lit", 0)
        Events { OnCollision("spark") { Actions { whenVar("lit", "==", 0) { Actions { changeVar("lit", 1) setColor("fde047") sound("pickup") burst(8, "fde047") updateBar("leds", 1) } } } }
            OnSignal("blackout") { Actions { whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("365314") updateBar("leds", -1) } } } } } }
    Entity { id("led5") startPosition(432, 332) color("365314") Type { circle(18) } var("lit", 0)
        Events { OnCollision("spark") { Actions { whenVar("lit", "==", 0) { Actions { changeVar("lit", 1) setColor("fde047") sound("pickup") burst(8, "fde047") updateBar("leds", 1) } } } }
            OnSignal("blackout") { Actions { whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("365314") updateBar("leds", -1) } } } } } }
    Entity { id("led6") startPosition(132, 832) color("365314") Type { circle(18) } var("lit", 0)
        Events { OnCollision("spark") { Actions { whenVar("lit", "==", 0) { Actions { changeVar("lit", 1) setColor("fde047") sound("pickup") burst(8, "fde047") updateBar("leds", 1) } } } }
            OnSignal("blackout") { Actions { whenVar("lit", "==", 1) { Actions { changeVar("lit", -1) setColor("365314") updateBar("leds", -1) } } } } } }

    # Switches: way 0 = up, 1 = right, 2 = down, 3 = left.
    Entity { id("swA") startPosition(330, 115) color("67e8f9") Type { text("→", 60) } var("way", 1)
        Events {
            OnTapped("swA") { Actions { sound("click") changeVar("way", 1) whenVar("way", "==", 4) { Actions { changeVar("way", -4) } }
                whenVar("way", "==", 0) { Actions { setText("swA", "↑") } } whenVar("way", "==", 1) { Actions { setText("swA", "→") } }
                whenVar("way", "==", 2) { Actions { setText("swA", "↓") } } whenVar("way", "==", 3) { Actions { setText("swA", "←") } } } }
            OnCollision("spark") { Actions { sound("blip")
                whenVar("way", "==", 0) { Actions { setDirection("spark", 0, 1) } } whenVar("way", "==", 1) { Actions { setDirection("spark", 1, 0) } }
                whenVar("way", "==", 2) { Actions { setDirection("spark", 0, -1) } } whenVar("way", "==", 3) { Actions { setDirection("spark", -1, 0) } } } } } }
    Entity { id("swE") startPosition(330, 315) color("67e8f9") Type { text("←", 60) } var("way", 3)
        Events {
            OnTapped("swE") { Actions { sound("click") changeVar("way", 1) whenVar("way", "==", 4) { Actions { changeVar("way", -4) } }
                whenVar("way", "==", 0) { Actions { setText("swE", "↑") } } whenVar("way", "==", 1) { Actions { setText("swE", "→") } }
                whenVar("way", "==", 2) { Actions { setText("swE", "↓") } } whenVar("way", "==", 3) { Actions { setText("swE", "←") } } } }
            OnCollision("spark") { Actions { sound("blip")
                whenVar("way", "==", 0) { Actions { setDirection("spark", 0, 1) } } whenVar("way", "==", 1) { Actions { setDirection("spark", 1, 0) } }
                whenVar("way", "==", 2) { Actions { setDirection("spark", 0, -1) } } whenVar("way", "==", 3) { Actions { setDirection("spark", -1, 0) } } } } } }
    Entity { id("swB") startPosition(330, 515) color("67e8f9") Type { text("↑", 60) } var("way", 0)
        Events {
            OnTapped("swB") { Actions { sound("click") changeVar("way", 1) whenVar("way", "==", 4) { Actions { changeVar("way", -4) } }
                whenVar("way", "==", 0) { Actions { setText("swB", "↑") } } whenVar("way", "==", 1) { Actions { setText("swB", "→") } }
                whenVar("way", "==", 2) { Actions { setText("swB", "↓") } } whenVar("way", "==", 3) { Actions { setText("swB", "←") } } } }
            OnCollision("spark") { Actions { sound("blip")
                whenVar("way", "==", 0) { Actions { setDirection("spark", 0, 1) } } whenVar("way", "==", 1) { Actions { setDirection("spark", 1, 0) } }
                whenVar("way", "==", 2) { Actions { setDirection("spark", 0, -1) } } whenVar("way", "==", 3) { Actions { setDirection("spark", -1, 0) } } } } } }
    Entity { id("swC") startPosition(630, 515) color("67e8f9") Type { text("↑", 60) } var("way", 0)
        Events {
            OnTapped("swC") { Actions { sound("click") changeVar("way", 1) whenVar("way", "==", 4) { Actions { changeVar("way", -4) } }
                whenVar("way", "==", 0) { Actions { setText("swC", "↑") } } whenVar("way", "==", 1) { Actions { setText("swC", "→") } }
                whenVar("way", "==", 2) { Actions { setText("swC", "↓") } } whenVar("way", "==", 3) { Actions { setText("swC", "←") } } } }
            OnCollision("spark") { Actions { sound("blip")
                whenVar("way", "==", 0) { Actions { setDirection("spark", 0, 1) } } whenVar("way", "==", 1) { Actions { setDirection("spark", 1, 0) } }
                whenVar("way", "==", 2) { Actions { setDirection("spark", 0, -1) } } whenVar("way", "==", 3) { Actions { setDirection("spark", -1, 0) } } } } } }
    Entity { id("swD") startPosition(630, 315) color("67e8f9") Type { text("↓", 60) } var("way", 2)
        Events {
            OnTapped("swD") { Actions { sound("click") changeVar("way", 1) whenVar("way", "==", 4) { Actions { changeVar("way", -4) } }
                whenVar("way", "==", 0) { Actions { setText("swD", "↑") } } whenVar("way", "==", 1) { Actions { setText("swD", "→") } }
                whenVar("way", "==", 2) { Actions { setText("swD", "↓") } } whenVar("way", "==", 3) { Actions { setText("swD", "←") } } } }
            OnCollision("spark") { Actions { sound("blip")
                whenVar("way", "==", 0) { Actions { setDirection("spark", 0, 1) } } whenVar("way", "==", 1) { Actions { setDirection("spark", 1, 0) } }
                whenVar("way", "==", 2) { Actions { setDirection("spark", 0, -1) } } whenVar("way", "==", 3) { Actions { setDirection("spark", -1, 0) } } } } } }
    Entity { id("swF") startPosition(330, 815) color("67e8f9") Type { text("→", 60) } var("way", 1)
        Events {
            OnTapped("swF") { Actions { sound("click") changeVar("way", 1) whenVar("way", "==", 4) { Actions { changeVar("way", -4) } }
                whenVar("way", "==", 0) { Actions { setText("swF", "↑") } } whenVar("way", "==", 1) { Actions { setText("swF", "→") } }
                whenVar("way", "==", 2) { Actions { setText("swF", "↓") } } whenVar("way", "==", 3) { Actions { setText("swF", "←") } } } }
            OnCollision("spark") { Actions { sound("blip")
                whenVar("way", "==", 0) { Actions { setDirection("spark", 0, 1) } } whenVar("way", "==", 1) { Actions { setDirection("spark", 1, 0) } }
                whenVar("way", "==", 2) { Actions { setDirection("spark", 0, -1) } } whenVar("way", "==", 3) { Actions { setDirection("spark", -1, 0) } } } } } }

    Entity { id("spark") startPosition(20, 120) color("fef08a") Type { circle(30) }
        Movement { GridMove(100, 0.45) { direction(0, 0) } }
        Events {
            OnGridStep { Actions { burst(3, "fde047") } }
            OnGridBlocked { Actions {
                sound("buzz") shake(6, 0.4) flash("ef4444", 0.3) burst(16, "f97316")
                updateBar("lives", -1) updateBar("running", -1) broadcast("blackout")
                setPosition("spark", 20, 120, 0) setDirection("spark", 0, 0) } } } }
}
