# Clover Patch: nine patches of leaves, five picks. Patches that twitch are hiding bees.
Window {
    size(7, 9)
    background("14532d")
    music("bg_2")

    Bar("luck") {
        initial(0)
        OnValue(1) { Actions { setText("sign", "Two more clovers!") } }
        OnValue(2) { Actions { setText("sign", "One more clover!") } }
    }
    winAt("luck", 3, "-- A LUCKY DAY --", "win2")
    Bar("picks") { initial(5) OnValue(0) { Actions { sound("lose") gameOver("-- OUT OF PICKS --") } } }
    Bar("stings") { initial(0) OnValue(2) { Actions { sound("lose") shake(8, 0.4) gameOver("-- STUNG TWICE --") } } }

    Entity { id("sign") startPosition(150, 820) color("fef9c3") Type { text("Find 3 four-leaf clovers", 34) } }
    Entity { id("hint") startPosition(150, 40) color("86efac") Type { text("watch for twitching bees", 28) } }

    Entity { id("patch") template(true) startPosition(0, 0) color("4ade80")
        Type { text("🌿", 72) }
        var("kind", 0)
        var("open", 0)
        Events {
            OnElapsedTime { Actions { randomAction {
                Actions { changeVar("kind", 1) }
                Actions { changeVar("kind", 1) }
                Actions { changeVar("kind", 2) }
                Actions { }
                Actions { }
            } } }
            OnElapsedTime(random(1.5, 3.5)) { repeatCount(-1) Actions {
                whenVar("kind", "==", 2) { Actions { whenVar("open", "==", 0) { Actions {
                    setScale(1.2)
                    delay(0.2) { Actions { setScale(1) } }
                } } } }
            } }
            OnTapped { Actions { whenVar("open", "==", 0) { Actions {
                changeVar("open", 1)
                updateBar("picks", -1)
                whenVar("kind", "==", 0) { Actions { sound("pop") setText("patch", "☘") setColor("86efac") } }
                whenVar("kind", "==", 1) { Actions { sound("sparkle") setText("patch", "🍀") burst(12, "22c55e") updateBar("luck", 1) } }
                whenVar("kind", "==", 2) { Actions { sound("hurt") setText("patch", "🐝") flash("facc15", 0.3) shake(6, 0.3) updateBar("stings", 1) } }
            } } } }
        } }

    Entity { id("gardener") startPosition(0, 0) color("00000000") Type { circle(1) }
        Events { OnElapsedTime { Actions {
            spawn("patch", 110, 600) spawn("patch", 310, 600) spawn("patch", 510, 600)
            spawn("patch", 110, 400) spawn("patch", 310, 400) spawn("patch", 510, 400)
            spawn("patch", 110, 200) spawn("patch", 310, 200) spawn("patch", 510, 200)
        } } } }
}
