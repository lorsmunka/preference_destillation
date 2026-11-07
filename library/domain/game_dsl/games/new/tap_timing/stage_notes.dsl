# Stage Notes: notes fall down three lanes to the beat of the band.
# Tap a note while it is inside the glowing band. Tap it too early, or let it
# slip past, and it is a miss. Ten hits in a row start HYPE: every hit counts double.
# Finish the song with 50 points for an encore; ten misses and the crowd boos you off.
Window {
    size(6, 12)
    background("0f0a1f")
    music("bg_6")

    Bar("score") { initial(0) }
    Bar("combo") {
        initial(0)
        OnValue(10) {
            Actions {
                updateBar("combo", -10)
                whenBar("hype", "==", 0) { Actions { updateBar("hype", 1) broadcast("hype") } }
            }
        }
    }
    Bar("hype") { initial(0) }
    Bar("misses") {
        initial(0)
        OnValue(7) { Actions { setText("crowd", "The crowd is restless...") } }
        OnValue(10) { Actions { sound("lose") shake(8, 0.5) gameOver("-- BOOED OFF THE STAGE --") } }
    }

    Entity { id("band") startPosition(0, 180) color("a855f7") Type { rectangle(600, 90) }
        Events {
            OnElapsedTime { Actions { setOpacity(0.35) } }
            OnSignal("hype") { Actions { setColor("f59e0b") setOpacity(0.6) } }
            OnSignal("calm") { Actions { setColor("a855f7") setOpacity(0.35) } }
        } }
    Entity { id("edge") startPosition(0, 100) color("4c1d95") Type { rectangle(600, 10) } }

    Entity { id("noteA") template(true) startPosition(0, 0) color("f472b6") Type { circle(36) }
        var("inBand", 0)
        Movement { speed(4) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("band") { Actions { changeVar("inBand", 1) setScale(1.15) } }
            OnCollision("edge") { Actions { broadcast("miss") deleteEntity() } }
            OnTapped { Actions {
                whenVar("inBand", "==", 1) { Actions { broadcast("hit") burst(12, "f472b6") deleteEntity() } }
                whenVar("inBand", "==", 0) { Actions { broadcast("miss") flash("ef4444", 0.1) deleteEntity() } }
            } }
        } }
    Entity { id("noteB") template(true) startPosition(0, 0) color("22d3ee") Type { circle(36) }
        var("inBand", 0)
        Movement { speed(4) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("band") { Actions { changeVar("inBand", 1) setScale(1.15) } }
            OnCollision("edge") { Actions { broadcast("miss") deleteEntity() } }
            OnTapped { Actions {
                whenVar("inBand", "==", 1) { Actions { broadcast("hit") burst(12, "22d3ee") deleteEntity() } }
                whenVar("inBand", "==", 0) { Actions { broadcast("miss") flash("ef4444", 0.1) deleteEntity() } }
            } }
        } }
    Entity { id("noteC") template(true) startPosition(0, 0) color("a3e635") Type { circle(36) }
        var("inBand", 0)
        Movement { speed(4) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("band") { Actions { changeVar("inBand", 1) setScale(1.15) } }
            OnCollision("edge") { Actions { broadcast("miss") deleteEntity() } }
            OnTapped { Actions {
                whenVar("inBand", "==", 1) { Actions { broadcast("hit") burst(12, "a3e635") deleteEntity() } }
                whenVar("inBand", "==", 0) { Actions { broadcast("miss") flash("ef4444", 0.1) deleteEntity() } }
            } }
        } }

    Entity {
        id("conductor")
        startPosition(8, 1150)
        color("fde68a")
        Type { star(5, 22, 9) }
        Events {
            Sequence {
                # intro: one note every 0.8 s
                Phase(8, 0.8) { Actions { broadcast("beat") randomAction {
                    Actions { spawn("noteA", 64, 1100) }
                    Actions { spawn("noteB", 264, 1100) }
                    Actions { spawn("noteC", 464, 1100) }
                } } }
                # verse: quicker
                Phase(16, 0.6) { Actions { broadcast("beat") randomAction {
                    Actions { spawn("noteA", 64, 1100) }
                    Actions { spawn("noteB", 264, 1100) }
                    Actions { spawn("noteC", 464, 1100) }
                } } }
                # bridge: two notes at once
                Phase(8, 0.7) { Actions { broadcast("beat") randomAction {
                    Actions { spawn("noteA", 64, 1100) spawn("noteB", 264, 1100) }
                    Actions { spawn("noteB", 264, 1100) spawn("noteC", 464, 1100) }
                    Actions { spawn("noteA", 64, 1100) spawn("noteC", 464, 1100) }
                } } }
                # syncopated solo: the gaps change every beat
                Phase(12, random(0.35, 0.8)) { Actions { broadcast("beat") randomAction {
                    Actions { spawn("noteA", 64, 1100) }
                    Actions { spawn("noteB", 264, 1100) }
                    Actions { spawn("noteC", 464, 1100) }
                } } }
                # chorus: fast
                Phase(20, 0.45) { Actions { broadcast("beat") randomAction {
                    Actions { spawn("noteA", 64, 1100) }
                    Actions { spawn("noteB", 264, 1100) }
                    Actions { spawn("noteC", 464, 1100) }
                } } }
                # let the last notes land, then the verdict
                Phase(1, 3.5) { Actions {
                    whenBar("score", ">=", 50) { Actions { sound("win") shake(6, 0.6) burst(30, "fde68a") gameOver("-- ENCORE! ENCORE! --") } }
                    whenBar("score", "<", 50) { Actions { sound("lose") gameOver("-- POLITE APPLAUSE ONLY --") } }
                } }
            }
            OnSignal("hit") { Actions {
                sound("blip")
                updateBar("score", 1)
                whenBar("hype", "==", 1) { Actions { updateBar("score", 1) } }
                updateBar("combo", 1)
            } }
            OnSignal("miss") { Actions {
                sound("buzz")
                updateBar("misses", 1)
                whenBar("combo", "==", 1) { Actions { updateBar("combo", -1) } }
                whenBar("combo", "==", 2) { Actions { updateBar("combo", -2) } }
                whenBar("combo", "==", 3) { Actions { updateBar("combo", -3) } }
                whenBar("combo", "==", 4) { Actions { updateBar("combo", -4) } }
                whenBar("combo", "==", 5) { Actions { updateBar("combo", -5) } }
                whenBar("combo", "==", 6) { Actions { updateBar("combo", -6) } }
                whenBar("combo", "==", 7) { Actions { updateBar("combo", -7) } }
                whenBar("combo", "==", 8) { Actions { updateBar("combo", -8) } }
                whenBar("combo", "==", 9) { Actions { updateBar("combo", -9) } }
                whenBar("hype", "==", 1) { Actions { updateBar("hype", -1) broadcast("calm") } }
            } }
        } }

    Entity { id("speakerL") startPosition(10, 600) color("334155") Type { rectangle(40, 120) }
        Events {
            OnSignal("beat") { Actions { setScale(1.12) delay(0.1) { Actions { setScale(1) } } } }
            OnSignal("hype") { Actions { setColor("f59e0b") } }
            OnSignal("calm") { Actions { setColor("334155") } }
        } }
    Entity { id("speakerR") startPosition(550, 600) color("334155") Type { rectangle(40, 120) }
        Events {
            OnSignal("beat") { Actions { setScale(1.12) delay(0.1) { Actions { setScale(1) } } } }
            OnSignal("hype") { Actions { setColor("f59e0b") } }
            OnSignal("calm") { Actions { setColor("334155") } }
        } }

    Entity { id("crowd") startPosition(30, 30) color("c4b5fd") Type { text("Tap the notes in the band", 30) }
        Events {
            OnSignal("hype") { Actions { setText("crowd", "HYPE! Double points!") sound("sparkle") shake(3, 0.2) } }
            OnSignal("calm") { Actions { setText("crowd", "Combo lost") } }
        } }
}
