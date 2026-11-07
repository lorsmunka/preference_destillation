Window {
    size(8, 12)
    background("fce7f3")
    music("bg_4")
    gravity(700)

    Bar("sorted") { initial(0) }
    winAt("sorted", 20, "-- PERFECTLY SORTED --", "win")
    lives(3, "-- QUALITY CONTROL SHUT THE LINE --", "lose")

    Entity { id("hopper") startPosition(30, 1080) color("a855f7")
        Type { triangle(80, "down") }
        Events { Sequence {
            Phase(6, 3.2) { Actions { randomAction {
                Actions { spawn("berry", 52, 1050) } Actions { spawn("mint", 52, 1050) } } } }
            Phase(10, 2.6) { Actions { randomAction {
                Actions { spawn("berry", 52, 1050) } Actions { spawn("mint", 52, 1050) } } } }
            Phase(-1, 2.0) { Actions { flash("f0abfc", 0.2) randomAction {
                Actions { spawn("berry", 52, 1050) } Actions { spawn("mint", 52, 1050) } } } } } } }

    # three one-way belts zigzag down; the roller strip on each sets the candy rolling
    Entity { id("belt1") startPosition(0, 880) color("475569") oneWay(true) Type { rectangle(560, 16) } }
    Entity { id("push1") startPosition(0, 890) color("94a3b8") Type { rectangle(560, 14) } }
    Entity { id("belt2") startPosition(240, 640) color("475569") oneWay(true) Type { rectangle(560, 16) } }
    Entity { id("push2") startPosition(240, 650) color("94a3b8") Type { rectangle(560, 14) } }
    Entity { id("belt3") startPosition(0, 400) color("475569") oneWay(true) Type { rectangle(560, 16) } }
    Entity { id("push3") startPosition(0, 410) color("94a3b8") Type { rectangle(560, 14) } }

    # the sorting plate kicks every candy left or right, whichever kicker is switched on
    Entity { id("plate") startPosition(480, 180) color("f472b6") oneWay(true) Type { rectangle(280, 16) }
        Events {
            OnElapsedTime { Actions { setEnable("kickR", false) } }
            OnSignal("toPink") { Actions { setColor("f472b6") setEnable("kickL", true) setEnable("kickR", false) } }
            OnSignal("toBlue") { Actions { setColor("60a5fa") setEnable("kickR", true) setEnable("kickL", false) } } } }
    Entity { id("kickL") startPosition(480, 190) color("00000000") Type { rectangle(280, 14) } }
    Entity { id("kickR") startPosition(480, 190) color("00000000") Type { rectangle(280, 14) } }

    Entity { id("binPink") startPosition(0, 0) color("f9a8d4") Type { rectangle(420, 90) } }
    Entity { id("binBlue") startPosition(580, 0) color("93c5fd") Type { rectangle(220, 90) } }

    Entity { id("pinkBtn") startPosition(380, 1110) color("ec4899") Type { rectangle(190, 70) }
        Events { OnTapped("pinkBtn") { Actions { sound("click") broadcast("toPink") } } } }
    Entity { id("blueBtn") startPosition(590, 1110) color("3b82f6") Type { rectangle(190, 70) }
        Events { OnTapped("blueBtn") { Actions { sound("click") broadcast("toBlue") } } } }
    Entity { id("pinkLbl") startPosition(410, 1130) color("ffffff") Type { text("< PINK", 28) } }
    Entity { id("blueLbl") startPosition(620, 1130) color("ffffff") Type { text("BLUE >", 28) } }

    Entity {
        id("berry")
        template(true)
        startPosition(0, 0)
        color("ec4899")
        Type { circle(18) }
        Movement { speed(0) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("push1") { Actions { setVelocity(150, 0) } }
            OnCollision("push2") { Actions { setVelocity(-150, 0) } }
            OnCollision("push3") { Actions { setVelocity(160, 0) } }
            OnCollision("kickL") { Actions { sound("blip") setVelocity(-260, 0) } }
            OnCollision("kickR") { Actions { sound("blip") setVelocity(260, 0) } }
            OnCollision("binPink") { Actions { sound("coin") updateBar("sorted", 1) burst(12, "ec4899") deleteEntity() } }
            OnCollision("binBlue") { Actions {
                sound("buzz") updateBar("lives", -1) flash("binBlue", "ef4444", 0.3) shake(5, 0.3) deleteEntity() } }
        }
    }

    Entity {
        id("mint")
        template(true)
        startPosition(0, 0)
        color("0ea5e9")
        Type { circle(18) }
        Movement { speed(0) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("push1") { Actions { setVelocity(150, 0) } }
            OnCollision("push2") { Actions { setVelocity(-150, 0) } }
            OnCollision("push3") { Actions { setVelocity(160, 0) } }
            OnCollision("kickL") { Actions { sound("blip") setVelocity(-260, 0) } }
            OnCollision("kickR") { Actions { sound("blip") setVelocity(260, 0) } }
            OnCollision("binBlue") { Actions { sound("coin") updateBar("sorted", 1) burst(12, "0ea5e9") deleteEntity() } }
            OnCollision("binPink") { Actions {
                sound("buzz") updateBar("lives", -1) flash("binPink", "ef4444", 0.3) shake(5, 0.3) deleteEntity() } }
        }
    }
}
