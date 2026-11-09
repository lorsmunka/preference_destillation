# Three rounds of repeat-the-lights. Each round is longer and played faster.
Window {
    size(8, 10)
    background("0f172a")
    music("bg_2")
    lives(3, "THE ECHO FADED", "lose")
    winAt("progress", 9, "PERFECT MEMORY", "win")
    Bar("ready") { initial(0) }

    Level {
        Entity { id("label") startPosition(260, 900) color("94a3b8") Type { text("Round 1", 44) } }
        Entity { id("padA") startPosition(100, 450) color("b91c1c") Type { rectangle(280, 280) }
            Events {
                OnSignal("showA") { Actions { flash("fecaca", 0.5) sound("beep") } }
                OnTapped("padA") { Actions { whenBar("ready", "==", 1) { Actions {
                    whenBar("progress", "<", 0) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", ">", 0) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", "==", 0) { Actions { updateBar("progress", 1) sound("pickup") flash("fecaca", 0.2) } }
                } } } } } }
        Entity { id("padB") startPosition(420, 450) color("1d4ed8") Type { rectangle(280, 280) }
            Events {
                OnSignal("showB") { Actions { flash("bfdbfe", 0.5) sound("bell") } }
                OnTapped("padB") { Actions { whenBar("ready", "==", 1) { Actions {
                    whenBar("progress", "<", 1) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", ">", 1) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", "==", 1) { Actions { updateBar("progress", 1) sound("win") updateBar("ready", -1) nextLevel() } }
                } } } } } }
        Entity { id("padC") startPosition(100, 130) color("15803d") Type { rectangle(280, 280) }
            Events { OnTapped("padC") { Actions { whenBar("ready", "==", 1) { Actions { updateBar("lives", -1) sound("buzz") } } } } } }
        Entity { id("padD") startPosition(420, 130) color("a16207") Type { rectangle(280, 280) }
            Events { OnTapped("padD") { Actions { whenBar("ready", "==", 1) { Actions { updateBar("lives", -1) sound("buzz") } } } } } }
        Entity { id("director") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { Sequence {
                Phase(1, 1.2) { Actions { broadcast("showA") } }
                Phase(1, 1.0) { Actions { broadcast("showB") } }
                Phase(1, 1.0) { Actions { updateBar("ready", 1) setText("label", "Round 1: go!") } }
            } } }
    }

    Level {
        Entity { id("label") startPosition(260, 900) color("94a3b8") Type { text("Round 2", 44) } }
        Entity { id("padA") startPosition(100, 450) color("b91c1c") Type { rectangle(280, 280) }
            Events {
                OnSignal("showA") { Actions { flash("fecaca", 0.4) sound("beep") } }
                OnTapped("padA") { Actions { whenBar("ready", "==", 1) { Actions {
                    whenBar("progress", "<", 3) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", ">", 3) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", "==", 3) { Actions { updateBar("progress", 1) sound("pickup") flash("fecaca", 0.2) } }
                } } } } } }
        Entity { id("padB") startPosition(420, 450) color("1d4ed8") Type { rectangle(280, 280) }
            Events {
                OnSignal("showB") { Actions { flash("bfdbfe", 0.4) sound("bell") } }
                OnTapped("padB") { Actions { whenBar("ready", "==", 1) { Actions {
                    whenBar("progress", "<", 4) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", ">", 4) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", "==", 4) { Actions { updateBar("progress", 1) sound("win") updateBar("ready", -1) nextLevel() } }
                } } } } } }
        Entity { id("padC") startPosition(100, 130) color("15803d") Type { rectangle(280, 280) }
            Events {
                OnSignal("showC") { Actions { flash("bbf7d0", 0.4) sound("blip") } }
                OnTapped("padC") { Actions { whenBar("ready", "==", 1) { Actions {
                    whenBar("progress", "<", 2) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", ">", 2) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", "==", 2) { Actions { updateBar("progress", 1) sound("pickup") flash("bbf7d0", 0.2) } }
                } } } } } }
        Entity { id("padD") startPosition(420, 130) color("a16207") Type { rectangle(280, 280) }
            Events { OnTapped("padD") { Actions { whenBar("ready", "==", 1) { Actions { updateBar("lives", -1) sound("buzz") } } } } } }
        Entity { id("director") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { Sequence {
                Phase(1, 1.2) { Actions { broadcast("showC") } }
                Phase(1, 0.8) { Actions { broadcast("showA") } }
                Phase(1, 0.8) { Actions { broadcast("showB") } }
                Phase(1, 0.9) { Actions { updateBar("ready", 1) setText("label", "Round 2: go!") } }
            } } }
    }

    Level {
        Entity { id("label") startPosition(260, 900) color("94a3b8") Type { text("Round 3", 44) } }
        Entity { id("padA") startPosition(100, 450) color("b91c1c") Type { rectangle(280, 280) }
            Events {
                OnSignal("showA") { Actions { flash("fecaca", 0.3) sound("beep") } }
                OnTapped("padA") { Actions { whenBar("ready", "==", 1) { Actions {
                    whenBar("progress", "<", 7) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", ">", 7) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", "==", 7) { Actions { updateBar("progress", 1) sound("pickup") flash("fecaca", 0.2) } }
                } } } } } }
        Entity { id("padB") startPosition(420, 450) color("1d4ed8") Type { rectangle(280, 280) }
            Events {
                OnSignal("showB") { Actions { flash("bfdbfe", 0.3) sound("bell") } }
                OnTapped("padB") { Actions { whenBar("ready", "==", 1) { Actions {
                    whenBar("progress", "<", 5) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", ">", 5) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", "==", 5) { Actions { updateBar("progress", 1) sound("pickup") flash("bfdbfe", 0.2) } }
                } } } } } }
        Entity { id("padC") startPosition(100, 130) color("15803d") Type { rectangle(280, 280) }
            Events {
                OnSignal("showC") { Actions { flash("bbf7d0", 0.3) sound("blip") } }
                OnTapped("padC") { Actions { whenBar("ready", "==", 1) { Actions {
                    whenBar("progress", "<", 8) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", ">", 8) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", "==", 8) { Actions { updateBar("progress", 1) } }
                } } } } } }
        Entity { id("padD") startPosition(420, 130) color("a16207") Type { rectangle(280, 280) }
            Events {
                OnSignal("showD") { Actions { flash("fef08a", 0.3) sound("click") } }
                OnTapped("padD") { Actions { whenBar("ready", "==", 1) { Actions {
                    whenBar("progress", "<", 6) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", ">", 6) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", "==", 6) { Actions { updateBar("progress", 1) sound("pickup") flash("fef08a", 0.2) } }
                } } } } } }
        Entity { id("director") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { Sequence {
                Phase(1, 1.2) { Actions { broadcast("showB") } }
                Phase(1, 0.6) { Actions { broadcast("showD") } }
                Phase(1, 0.6) { Actions { broadcast("showA") } }
                Phase(1, 0.6) { Actions { broadcast("showC") } }
                Phase(1, 0.8) { Actions { updateBar("ready", 1) setText("label", "Round 3: go!") } }
            } } }
    }
}
