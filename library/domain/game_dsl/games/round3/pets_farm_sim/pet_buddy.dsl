Window {
    size(8, 12)
    background("fde7c8")
    music("bg_ambient")
    Bar("hunger") {
        initial(3)
        OnValue(10) { Actions { sound("lose") gameOver("BUDDY RAN AWAY HUNGRY") } }
        OnValue(6) { Actions { broadcast("hungry") } }
        OnValue(5) { Actions { broadcast("fine") } }
    }
    Bar("happy") {
        initial(0)
        OnValue(8) { Actions { sound("win") gameOver("BUDDY LOVES YOU") } }
    }

    Entity {
        id("pet")
        startPosition(300, 600)
        color("fbbf24")
        Type { circle(100) }
        Events {
            OnTapped { Actions {
                updateBar("happy", 1)
                burst(6, "f472b6")
                setScale("pet", 1.2)
                delay(0.2) { Actions { setScale("pet", 1) } }
            } }
            OnSignal("hungry") { Actions { setColor("94a3b8") setText("status", "Buddy is starving!") } }
            OnSignal("fine") { Actions { setColor("fbbf24") setText("status", "Buddy feels fine") } }
        }
    }
    Entity {
        id("status")
        startPosition(180, 1050)
        color("7c2d12")
        Type { text("Buddy feels fine", 40) }
    }
    Entity {
        id("hungerTimer")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(2.5) {
                repeatCount(-1)
                Actions { updateBar("hunger", 1) }
            }
        }
    }
    Entity {
        id("feedButton")
        startPosition(100, 200)
        color("22c55e")
        Type { rectangle(260, 140) }
        Events {
            OnTapped("feedButton") { Actions {
                updateBar("hunger", -1)
                sound("pop")
                flash("feedButton", "ffffff", 0.15)
            } }
        }
    }
    Entity {
        id("feedLabel")
        startPosition(150, 250)
        color("ffffff")
        Type { text("FEED", 48) }
    }
    Entity {
        id("playButton")
        startPosition(440, 200)
        color("3b82f6")
        Type { rectangle(260, 140) }
        Events {
            OnTapped("playButton") { Actions {
                updateBar("happy", 1)
                updateBar("hunger", 1)
                sound("blip")
                burst(8, "93c5fd")
            } }
        }
    }
    Entity {
        id("playLabel")
        startPosition(500, 250)
        color("ffffff")
        Type { text("PLAY", 48) }
    }
}
