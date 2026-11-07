# Flock Count: a flock streaks across the sky. Count only the white doves,
# then tap the number. Six right answers win; three wrong ones and the lookout is fired.
Window {
    size(9, 9)
    background("0e7490")
    music("bg_3")

    Bar("answer") { initial(0) }
    Bar("asking") { initial(0) }
    Bar("right") { initial(0) OnValue(3) { Actions { setText("title", "The flocks speed up!") } } }
    winAt("right", 6, "-- SHARPEST EYES ON THE COAST --", "win")
    Bar("wrong") { initial(0) OnValue(3) { Actions { sound("lose") gameOver("-- THE LOOKOUT LOST COUNT --") } } }

    Entity { id("title") startPosition(60, 840) color("cffafe") Type { text("Count the white doves", 36) } }
    Entity { id("prompt") startPosition(60, 160) color("ffffff") Type { text("Get ready...", 40) } }

    Entity { id("dove") template(true) startPosition(0, 0) color("f8fafc") Type { triangle(36, "right") }
        Movement { speed(5) AutoMove { direction(1, 0) } }
        Events {
            OnElapsedTime { Actions {
                updateSpeed(random(4, 7))
                whenBar("right", ">=", 3) { Actions { updateSpeed(random(7, 10)) } }
            } }
            OnCollision { Actions { deleteEntity() } }
        } }
    Entity { id("crow") template(true) startPosition(0, 0) color("1f2937") Type { triangle(36, "right") }
        Movement { speed(5) AutoMove { direction(1, 0) } }
        Events {
            OnElapsedTime { Actions {
                updateSpeed(random(4, 7))
                whenBar("right", ">=", 3) { Actions { updateSpeed(random(7, 10)) } }
            } }
            OnCollision { Actions { deleteEntity() } }
        } }

    Entity { id("lookout") startPosition(820, 820) color("fbbf24") Type { circle(24) }
        Events {
            OnElapsedTime(1.5) { Actions { broadcast("flight") } }
            OnSignal("flight") { Actions {
                whenBar("answer", "==", 1) { Actions { updateBar("answer", -1) } }
                whenBar("answer", "==", 2) { Actions { updateBar("answer", -2) } }
                whenBar("answer", "==", 3) { Actions { updateBar("answer", -3) } }
                whenBar("answer", "==", 4) { Actions { updateBar("answer", -4) } }
                whenBar("answer", "==", 5) { Actions { updateBar("answer", -5) } }
                whenBar("answer", "==", 6) { Actions { updateBar("answer", -6) } }
                broadcast("hush")
                setText("prompt", "Here they come!")
                sound("blip")
                delay(0.2) { Actions { randomAction {
                    Actions { spawn("dove", 10, random(260, 760)) updateBar("answer", 1) }
                    Actions { spawn("crow", 10, random(260, 760)) }
                    Actions { } } } }
                delay(0.55) { Actions { randomAction {
                    Actions { spawn("dove", 10, random(260, 760)) updateBar("answer", 1) }
                    Actions { spawn("crow", 10, random(260, 760)) }
                    Actions { } } } }
                delay(0.9) { Actions { randomAction {
                    Actions { spawn("dove", 10, random(260, 760)) updateBar("answer", 1) }
                    Actions { spawn("crow", 10, random(260, 760)) }
                    Actions { } } } }
                delay(1.25) { Actions { randomAction {
                    Actions { spawn("dove", 10, random(260, 760)) updateBar("answer", 1) }
                    Actions { spawn("crow", 10, random(260, 760)) }
                    Actions { } } } }
                delay(1.6) { Actions { randomAction {
                    Actions { spawn("dove", 10, random(260, 760)) updateBar("answer", 1) }
                    Actions { spawn("crow", 10, random(260, 760)) }
                    Actions { } } } }
                delay(1.95) { Actions { randomAction {
                    Actions { spawn("dove", 10, random(260, 760)) updateBar("answer", 1) }
                    Actions { spawn("crow", 10, random(260, 760)) }
                    Actions { } } } }
                delay(4.4) { Actions { updateBar("asking", 1) broadcast("ask") setText("prompt", "How many doves?") } }
            } }
            OnSignal("answered") { Actions {
                broadcast("hush")
                whenBar("answer", "==", 0) { Actions { setText("prompt", "It was 0 doves") } }
                whenBar("answer", "==", 1) { Actions { setText("prompt", "It was 1 dove") } }
                whenBar("answer", "==", 2) { Actions { setText("prompt", "It was 2 doves") } }
                whenBar("answer", "==", 3) { Actions { setText("prompt", "It was 3 doves") } }
                whenBar("answer", "==", 4) { Actions { setText("prompt", "It was 4 doves") } }
                whenBar("answer", "==", 5) { Actions { setText("prompt", "It was 5 doves") } }
                whenBar("answer", "==", 6) { Actions { setText("prompt", "It was 6 doves") } }
                delay(1.6) { Actions { broadcast("flight") } }
            } }
        } }

    Entity { id("n0") startPosition(60, 50) color("67e8f9") Type { text("0", 64) }
        Events { OnSignal("ask") { Actions { setOpacity(1) } } OnSignal("hush") { Actions { setOpacity(0.3) } }
            OnTapped("n0") { Actions { whenBar("asking", "==", 1) { Actions { updateBar("asking", -1)
                whenBar("answer", "==", 0) { Actions { sound("coin") burst(12, "4ade80") updateBar("right", 1) } }
                whenBar("answer", "!=", 0) { Actions { sound("buzz") shake(5, 0.3) updateBar("wrong", 1) } }
                broadcast("answered") } } } } } }
    Entity { id("n1") startPosition(175, 50) color("67e8f9") Type { text("1", 64) }
        Events { OnSignal("ask") { Actions { setOpacity(1) } } OnSignal("hush") { Actions { setOpacity(0.3) } }
            OnTapped("n1") { Actions { whenBar("asking", "==", 1) { Actions { updateBar("asking", -1)
                whenBar("answer", "==", 1) { Actions { sound("coin") burst(12, "4ade80") updateBar("right", 1) } }
                whenBar("answer", "!=", 1) { Actions { sound("buzz") shake(5, 0.3) updateBar("wrong", 1) } }
                broadcast("answered") } } } } } }
    Entity { id("n2") startPosition(290, 50) color("67e8f9") Type { text("2", 64) }
        Events { OnSignal("ask") { Actions { setOpacity(1) } } OnSignal("hush") { Actions { setOpacity(0.3) } }
            OnTapped("n2") { Actions { whenBar("asking", "==", 1) { Actions { updateBar("asking", -1)
                whenBar("answer", "==", 2) { Actions { sound("coin") burst(12, "4ade80") updateBar("right", 1) } }
                whenBar("answer", "!=", 2) { Actions { sound("buzz") shake(5, 0.3) updateBar("wrong", 1) } }
                broadcast("answered") } } } } } }
    Entity { id("n3") startPosition(405, 50) color("67e8f9") Type { text("3", 64) }
        Events { OnSignal("ask") { Actions { setOpacity(1) } } OnSignal("hush") { Actions { setOpacity(0.3) } }
            OnTapped("n3") { Actions { whenBar("asking", "==", 1) { Actions { updateBar("asking", -1)
                whenBar("answer", "==", 3) { Actions { sound("coin") burst(12, "4ade80") updateBar("right", 1) } }
                whenBar("answer", "!=", 3) { Actions { sound("buzz") shake(5, 0.3) updateBar("wrong", 1) } }
                broadcast("answered") } } } } } }
    Entity { id("n4") startPosition(520, 50) color("67e8f9") Type { text("4", 64) }
        Events { OnSignal("ask") { Actions { setOpacity(1) } } OnSignal("hush") { Actions { setOpacity(0.3) } }
            OnTapped("n4") { Actions { whenBar("asking", "==", 1) { Actions { updateBar("asking", -1)
                whenBar("answer", "==", 4) { Actions { sound("coin") burst(12, "4ade80") updateBar("right", 1) } }
                whenBar("answer", "!=", 4) { Actions { sound("buzz") shake(5, 0.3) updateBar("wrong", 1) } }
                broadcast("answered") } } } } } }
    Entity { id("n5") startPosition(635, 50) color("67e8f9") Type { text("5", 64) }
        Events { OnSignal("ask") { Actions { setOpacity(1) } } OnSignal("hush") { Actions { setOpacity(0.3) } }
            OnTapped("n5") { Actions { whenBar("asking", "==", 1) { Actions { updateBar("asking", -1)
                whenBar("answer", "==", 5) { Actions { sound("coin") burst(12, "4ade80") updateBar("right", 1) } }
                whenBar("answer", "!=", 5) { Actions { sound("buzz") shake(5, 0.3) updateBar("wrong", 1) } }
                broadcast("answered") } } } } } }
    Entity { id("n6") startPosition(750, 50) color("67e8f9") Type { text("6", 64) }
        Events { OnSignal("ask") { Actions { setOpacity(1) } } OnSignal("hush") { Actions { setOpacity(0.3) } }
            OnTapped("n6") { Actions { whenBar("asking", "==", 1) { Actions { updateBar("asking", -1)
                whenBar("answer", "==", 6) { Actions { sound("coin") burst(12, "4ade80") updateBar("right", 1) } }
                whenBar("answer", "!=", 6) { Actions { sound("buzz") shake(5, 0.3) updateBar("wrong", 1) } }
                broadcast("answered") } } } } } }
}
