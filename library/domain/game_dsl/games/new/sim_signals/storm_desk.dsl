# Storm Desk: weather fronts roll in from four sides toward the weather station.
# The radar calls each front a moment early. Swipe toward a storm to raise that side's shield (costs battery),
# but let the sunshine through: it recharges the battery and gives a reading too.
Window {
    size(9, 12)
    background("101826")
    music("bg_4")
    gravity(0)

    lives(3, "-- THE STATION WAS WRECKED --", "lose")
    Bar("readings") { initial(0) }
    winAt("readings", 20, "-- THE SEASON'S DATA IS IN --", "win")
    Bar("battery") {
        initial(6)
        OnValue(2) { Actions { broadcast("lowBattery") } }
        OnValue(3) { Actions { broadcast("batteryOk") } }
    }

    Entity {
        id("station")
        startPosition(400, 550)
        color("e2e8f0")
        Type { rectangle(100, 100) }
    }

    Entity {
        id("cue")
        startPosition(40, 1130)
        color("cbd5e1")
        Type { text("RADAR WARMING UP", 36) }
    }

    Entity {
        id("batteryLamp")
        startPosition(820, 1130)
        color("22c55e")
        Type { circle(22) }
        Events {
            OnSignal("lowBattery") { Actions { setColor("ef4444") sound("beep") } }
            OnSignal("batteryOk") { Actions { setColor("22c55e") } }
        }
    }

    Entity {
        id("desk")
        startPosition(250, 20)
        color("334155")
        Type { rectangle(400, 70) }
        Events {
            OnSwipe("left") { Actions {
                whenBar("battery", "==", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                whenBar("battery", ">", 0) { Actions { updateBar("battery", -1) broadcast("raiseW") sound("blip") } } } }
            OnSwipe("right") { Actions {
                whenBar("battery", "==", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                whenBar("battery", ">", 0) { Actions { updateBar("battery", -1) broadcast("raiseE") sound("blip") } } } }
            OnSwipe("up") { Actions {
                whenBar("battery", "==", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                whenBar("battery", ">", 0) { Actions { updateBar("battery", -1) broadcast("raiseN") sound("blip") } } } }
            OnSwipe("down") { Actions {
                whenBar("battery", "==", 0) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                whenBar("battery", ">", 0) { Actions { updateBar("battery", -1) broadcast("raiseS") sound("blip") } } } }
        }
    }

    Entity { id("shW") startPosition(290, 520) color("60a5fa") Type { rectangle(30, 160) }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("raiseW") { Actions { setEnable(true) flash("ffffff", 0.15) delay(1.6) { Actions { setEnable(false) } } } } } }
    Entity { id("shE") startPosition(580, 520) color("60a5fa") Type { rectangle(30, 160) }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("raiseE") { Actions { setEnable(true) flash("ffffff", 0.15) delay(1.6) { Actions { setEnable(false) } } } } } }
    Entity { id("shN") startPosition(370, 730) color("60a5fa") Type { rectangle(160, 30) }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("raiseN") { Actions { setEnable(true) flash("ffffff", 0.15) delay(1.6) { Actions { setEnable(false) } } } } } }
    Entity { id("shS") startPosition(370, 440) color("60a5fa") Type { rectangle(160, 30) }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("raiseS") { Actions { setEnable(true) flash("ffffff", 0.15) delay(1.6) { Actions { setEnable(false) } } } } } }

    Entity {
        id("radar")
        startPosition(60, 1000)
        color("475569")
        Type { ring(40, 28) }
        Events {
            Sequence {
                Phase(6, 3.2) { Actions { broadcast("front") } }
                Phase(10, 2.4) { Actions { broadcast("front") } }
                Phase(-1, 1.7) { Actions { broadcast("front") } }
            }
            OnSignal("front") { Actions { sound("beep") randomAction {
                Actions { setColor("94a3b8") setText("cue", "STORM FROM THE WEST") delay(0.8) { Actions { spawn("storm", 20, 580) } } }
                Actions { setColor("94a3b8") setText("cue", "STORM FROM THE EAST") delay(0.8) { Actions { spawn("storm", 820, 580) } } }
                Actions { setColor("94a3b8") setText("cue", "STORM FROM THE NORTH") delay(0.8) { Actions { spawn("storm", 430, 960) } } }
                Actions { setColor("94a3b8") setText("cue", "STORM FROM THE SOUTH") delay(0.8) { Actions { spawn("storm", 430, 160) } } }
                Actions { setColor("94a3b8") setText("cue", "STORM FROM THE WEST") delay(0.8) { Actions { spawn("storm", 20, 580) } } }
                Actions { setColor("facc15") setText("cue", "SUNSHINE FROM THE EAST") delay(0.8) { Actions { spawn("sun", 820, 580) } } }
                Actions { setColor("facc15") setText("cue", "SUNSHINE FROM THE NORTH") delay(0.8) { Actions { spawn("sun", 430, 960) } } }
                Actions { setColor("facc15") setText("cue", "SUNSHINE FROM THE SOUTH") delay(0.8) { Actions { spawn("sun", 430, 160) } } }
            } } }
        }
    }

    Entity {
        id("storm")
        template(true)
        color("64748b")
        Type { circle(36) }
        Movement { speed(1.6) chase("station") }
        Events {
            OnElapsedTime { Actions { setOpacity(0.3) delay(0.3) { Actions { setOpacity(0.7) delay(0.3) { Actions { setOpacity(1) } } } } } }
            OnElapsedTime(0.7) { repeatCount(-1) Actions { flash("fde047", 0.1) } }
            OnCollision("shW") { Actions { sound("thud") burst(12, "94a3b8") updateBar("readings", 1) deleteEntity() } }
            OnCollision("shE") { Actions { sound("thud") burst(12, "94a3b8") updateBar("readings", 1) deleteEntity() } }
            OnCollision("shN") { Actions { sound("thud") burst(12, "94a3b8") updateBar("readings", 1) deleteEntity() } }
            OnCollision("shS") { Actions { sound("thud") burst(12, "94a3b8") updateBar("readings", 1) deleteEntity() } }
            OnCollision("station") { Actions { sound("hurt") shake(8, 0.4) flash("station", "ef4444", 0.3) updateBar("lives", -1) deleteEntity() } }
        }
    }

    Entity {
        id("sun")
        template(true)
        color("fde047")
        Type { circle(30) }
        Movement { speed(1.6) chase("station") }
        Events {
            OnElapsedTime { Actions { setOpacity(0.3) delay(0.3) { Actions { setOpacity(0.7) delay(0.3) { Actions { setOpacity(1) } } } } } }
            OnCollision("shW") { Actions { sound("pop") burst(6, "fde047") deleteEntity() } }
            OnCollision("shE") { Actions { sound("pop") burst(6, "fde047") deleteEntity() } }
            OnCollision("shN") { Actions { sound("pop") burst(6, "fde047") deleteEntity() } }
            OnCollision("shS") { Actions { sound("pop") burst(6, "fde047") deleteEntity() } }
            OnCollision("station") { Actions {
                sound("sparkle") burst(14, "fde047") updateBar("readings", 1)
                whenBar("battery", "<", 9) { Actions { updateBar("battery", 1) } }
                whenBar("battery", "<", 9) { Actions { updateBar("battery", 1) } }
                deleteEntity() } }
        }
    }
}
