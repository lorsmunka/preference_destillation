# Potion Bench: the order card names a recipe. Tap the red, blue and green valves to pour exactly that much;
# when all three are measured out the potion is bottled. Pour one drop too many, or let the candle burn down,
# and the brew boils over.
Window {
    size(9, 10)
    background("1e1028")
    music("bg_1")
    gravity(0)

    lives(3, "-- THE WORKSHOP BLEW UP --", "lose")
    Bar("potions") { initial(0) }
    winAt("potions", 8, "-- THE SHELF IS STOCKED --", "win")
    Bar("brewing") { initial(0) }
    Bar("red") { initial(0) OnValue(0) { Actions { broadcast("checkBrew") } } }
    Bar("blue") { initial(0) OnValue(0) { Actions { broadcast("checkBrew") } } }
    Bar("green") { initial(0) OnValue(0) { Actions { broadcast("checkBrew") } } }
    Bar("candle") { initial(10)
        OnValue(3) { Actions { broadcast("hurry") } }
        OnValue(10) { Actions { broadcast("fresh") } }
        OnValue(0) { Actions { broadcast("spoil") } } }

    Entity { id("card") startPosition(40, 930) color("f5d0fe") Type { text("Lighting the burner...", 36) }
        Events {
            OnElapsedTime(1) { Actions { broadcast("newOrder") } }
            OnSignal("newOrder") { Actions {
                randomAction {
                    Actions { updateBar("red", 2) updateBar("blue", 1) setText("card", "ORDER: 2 red + 1 blue") }
                    Actions { updateBar("red", 1) updateBar("green", 2) setText("card", "ORDER: 1 red + 2 green") }
                    Actions { updateBar("blue", 3) setText("card", "ORDER: 3 blue") }
                    Actions { updateBar("red", 1) updateBar("blue", 1) updateBar("green", 1) setText("card", "ORDER: 1 of each") }
                    Actions { updateBar("green", 2) updateBar("blue", 2) setText("card", "ORDER: 2 green + 2 blue") }
                    Actions { updateBar("red", 3) updateBar("green", 1) setText("card", "ORDER: 3 red + 1 green") }
                }
                updateBar("brewing", 1)
                sound("bell")
                loopCount(10)
                whenBar("candle", "<", 10) { Actions { updateBar("candle", 1) } } } }
            OnSignal("hurry") { Actions { setColor("f87171") } }
            OnSignal("fresh") { Actions { setColor("f5d0fe") } } } }

    Entity { id("candle") startPosition(790, 560) color("fef3c7") Type { rectangle(40, 160) }
        Events {
            OnElapsedTime(1.2) { repeatCount(-1) Actions { whenBar("brewing", "==", 1) { Actions { updateBar("candle", -1) } } } }
            OnSignal("hurry") { Actions { flash("f97316", 0.4) sound("beep") } } } }

    Entity { id("cauldron") startPosition(330, 260) color("334155") Type { circle(120) }
        Events {
            OnSignal("checkBrew") { Actions { whenBar("brewing", "==", 1) { Actions {
                whenBar("red", "==", 0) { Actions { whenBar("blue", "==", 0) { Actions { whenBar("green", "==", 0) { Actions {
                    updateBar("brewing", -1) updateBar("potions", 1)
                    spawnAtEmpty("bottle", 100, 50, 820, 8, 1)
                    sound("win2") burst(24, "c084fc") setColor("7e22ce")
                    delay(1.2) { Actions { setColor("334155") broadcast("newOrder") } } } } } } } } } } } }
            OnSignal("spoil") { Actions { whenBar("brewing", "==", 1) { Actions {
                updateBar("brewing", -1) updateBar("lives", -1)
                sound("death") shake(8, 0.5) burst(30, "57534e") setColor("57534e")
                delay(1.5) { Actions { setColor("334155") broadcast("newOrder") } }
                loopCount(4)
                whenBar("red", ">", 0) { Actions { updateBar("red", -1) } }
                whenBar("blue", ">", 0) { Actions { updateBar("blue", -1) } }
                whenBar("green", ">", 0) { Actions { updateBar("green", -1) } } } } } } } }

    Entity { id("valveRed") startPosition(80, 640) color("dc2626") Type { rectangle(130, 60) }
        Events { OnTapped("valveRed") { Actions { whenBar("brewing", "==", 1) { Actions {
            whenBar("red", "==", 0) { Actions { broadcast("spoil") } }
            whenBar("red", ">", 0) { Actions { sound("blip") spawnHere("dropRed") updateBar("red", -1) } } } } } } } }
    Entity { id("valveBlue") startPosition(385, 700) color("2563eb") Type { rectangle(130, 60) }
        Events { OnTapped("valveBlue") { Actions { whenBar("brewing", "==", 1) { Actions {
            whenBar("blue", "==", 0) { Actions { broadcast("spoil") } }
            whenBar("blue", ">", 0) { Actions { sound("blip") spawnHere("dropBlue") updateBar("blue", -1) } } } } } } } }
    Entity { id("valveGreen") startPosition(690, 640) color("16a34a") Type { rectangle(130, 60) }
        Events { OnTapped("valveGreen") { Actions { whenBar("brewing", "==", 1) { Actions {
            whenBar("green", "==", 0) { Actions { broadcast("spoil") } }
            whenBar("green", ">", 0) { Actions { sound("blip") spawnHere("dropGreen") updateBar("green", -1) } } } } } } } }

    Entity { id("dropRed") template(true) color("ef4444") Type { circle(14) }
        Movement { speed(4) chase("cauldron") }
        Events { OnCollision("cauldron") { Actions { sound("bubble") burst(8, "ef4444") deleteEntity() } } } }
    Entity { id("dropBlue") template(true) color("3b82f6") Type { circle(14) }
        Movement { speed(4) chase("cauldron") }
        Events { OnCollision("cauldron") { Actions { sound("bubble") burst(8, "3b82f6") deleteEntity() } } } }
    Entity { id("dropGreen") template(true) color("22c55e") Type { circle(14) }
        Movement { speed(4) chase("cauldron") }
        Events { OnCollision("cauldron") { Actions { sound("bubble") burst(8, "22c55e") deleteEntity() } } } }
    Entity { id("bottle") template(true) color("c084fc") Type { polygon(5, 30) } }
}
