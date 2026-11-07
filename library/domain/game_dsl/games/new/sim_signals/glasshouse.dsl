# Glasshouse: seedlings sprout in empty beds on their own. Each growth step drinks from the tank,
# and a plant with nothing to drink withers. Tap ripe tomatoes to harvest, tap withered plants to clear the bed,
# swipe up to pump water, swipe down to mist away the aphids (costs 3 water).
Window {
    size(9, 11)
    background("0d2818")
    music("bg_5")
    gravity(0)

    Bar("harvest") { initial(0) }
    winAt("harvest", 18, "-- A CRATE OF TOMATOES --", "win")
    Bar("water") { initial(8)
        OnValue(2) { Actions { broadcast("lowWater") } }
        OnValue(4) { Actions { broadcast("tankOk") } } }
    Bar("wilt") { initial(0)
        OnValue(3) { Actions { sound("beep") shake(3, 0.3) } }
        OnValue(6) { Actions { sound("lose") shake(6, 0.5) gameOver("-- THE GREENHOUSE WENT TO SEED --") } } }

    Wave(20, 8) { Actions { spawn("aphid", random(20, 860), 1060) } }
    Wave(30, 5) { Actions { spawn("aphid", random(20, 860), 1060) } }
    Wave(90, 3) { Actions { randomAction {
        Actions { spawn("aphid", random(20, 860), 1060) }
        Actions { spawn("aphid", 10, random(300, 800)) }
        Actions { spawn("aphid", 860, random(300, 800)) } } } }

    Entity { id("beds") startPosition(130, 260) color("3b2a1a") Type { rectangle(650, 490) }
        Events { OnElapsedTime(2.5) { repeatCount(-1) Actions { spawnAtEmpty("sprout", 160, 140, 270, 4, 3) } } } }

    Entity { id("tankNote") startPosition(40, 1030) color("bbf7d0") Type { text("Tank full", 34) }
        Events {
            OnSignal("lowWater") { Actions { setText("tankNote", "Tank low! Swipe up to pump") setColor("fca5a5") } }
            OnSignal("tankOk") { Actions { setText("tankNote", "Tank fine") setColor("bbf7d0") } } } }

    Entity { id("pump") startPosition(380, 60) color("0ea5e9") Type { rectangle(140, 120) } var("busy", 0)
        Events {
            OnSwipe("up") { Actions { whenVar("busy", "==", 0) { Actions { whenBar("water", "<", 12) { Actions {
                changeVar("busy", 1) updateBar("water", 1) sound("bubble") burst(8, "7dd3fc")
                delay(0.4) { Actions { changeVar("busy", -1) } } } } } } } }
            OnSwipe("down") { Actions {
                whenBar("water", "<", 3) { Actions { sound("buzz") flash("ef4444", 0.3) } }
                whenBar("water", ">=", 3) { Actions {
                    sound("glass") shake(3, 0.2) broadcast("mist")
                    loopCount(3) updateBar("water", -1) } } } } } }

    Entity {
        id("sprout")
        template(true)
        color("86efac")
        Type { circle(70) }
        var("stage", 0)
        Events {
            OnElapsedTime { Actions { setScale(0.35) } }
            Sequence {
                Phase(1, 4) { Actions {
                    whenBar("water", "==", 0) { Actions { whenVar("stage", "==", 0) { Actions {
                        changeVar("stage", 3) setColor("78350f") updateBar("wilt", 1) sound("thud") } } } }
                    whenBar("water", ">", 0) { Actions { whenVar("stage", "==", 0) { Actions {
                        changeVar("stage", 1) updateBar("water", -1) setScale(0.7) setColor("22c55e") } } } }
                } }
                Phase(1, 4) { Actions {
                    whenBar("water", "==", 0) { Actions { whenVar("stage", "==", 1) { Actions {
                        changeVar("stage", 2) setColor("78350f") updateBar("wilt", 1) sound("thud") } } } }
                    whenBar("water", ">", 0) { Actions { whenVar("stage", "==", 1) { Actions {
                        changeVar("stage", 1) updateBar("water", -1) setScale(1) setColor("ef4444") burst(6, "fca5a5") } } } }
                } }
                Phase(1, 6) { Actions {
                    whenVar("stage", "==", 2) { Actions { changeVar("stage", 1) setColor("78350f") setScale(0.8) updateBar("wilt", 1) sound("thud") } }
                } }
            }
            OnTapped { Actions {
                whenVar("stage", "==", 2) { Actions { sound("coin") burst(12, "ef4444") updateBar("harvest", 1) deleteEntity() } }
                whenVar("stage", "==", 3) { Actions { sound("pop") burst(8, "a16207") deleteEntity() } }
            } }
        }
    }

    Entity {
        id("aphid")
        template(true)
        color("a3e635")
        Type { circle(12) }
        Movement { speed(0.9) chase("sprout") }
        Events {
            OnTapped { Actions { sound("pop") burst(6, "a3e635") deleteEntity() } }
            OnSignal("mist") { Actions { burst(5, "e0f2fe") deleteEntity() } }
            OnCollision("sprout") { Actions { sound("hurt") flash("ef4444", 0.2) updateBar("wilt", 1) deleteCollider() deleteEntity() } }
        }
    }
}
