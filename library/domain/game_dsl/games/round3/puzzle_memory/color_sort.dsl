Window {
    size(6, 8)
    background("0f172a")
    music("bg_2")
    lives(3, "TOO MANY MISTAKES", "lose")
    winAt("score", 10, "SORTED 10 ORBS", "win")

    Entity {
        id("label")
        startPosition(60, 700)
        color("f8fafc")
        Type { text("Sort the orb by colour", 38) }
    }
    Entity {
        id("scoreText")
        startPosition(60, 640)
        color("94a3b8")
        Type { text("Goal: 10 orbs, 3 mistakes", 30) }
        Events {
            OnSignal("good") { Actions { setText("scoreText", "Good one!") } }
            OnSignal("sortRed") { Actions { delay(0.1) { Actions { setText("scoreText", "Next orb...") } } } }
        }
    }

    Entity {
        id("spawner")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.5) { Actions { broadcast("next") } }
            OnSignal("next") { Actions {
                randomAction {
                    Actions { spawn("redOrb", 240, 420) }
                    Actions { spawn("greenOrb", 240, 420) }
                    Actions { spawn("blueOrb", 240, 420) }
                }
            } }
        }
    }

    Entity {
        id("redOrb")
        template(true)
        startPosition(0, 0)
        color("ef4444")
        Type { circle(60) }
        Events {
            OnSignal("sortRed") { Actions { updateBar("score", 1) broadcast("good") sound("coin") burst(8, "ef4444") broadcast("next") deleteEntity() } }
            OnSignal("sortGreen") { Actions { updateBar("lives", -1) sound("buzz") broadcast("next") deleteEntity() } }
            OnSignal("sortBlue") { Actions { updateBar("lives", -1) sound("buzz") broadcast("next") deleteEntity() } }
            OnElapsedTime(4) { Actions { updateBar("lives", -1) sound("buzz") broadcast("next") deleteEntity() } }
        }
    }
    Entity {
        id("greenOrb")
        template(true)
        startPosition(0, 0)
        color("22c55e")
        Type { circle(60) }
        Events {
            OnSignal("sortGreen") { Actions { updateBar("score", 1) broadcast("good") sound("coin") burst(8, "22c55e") broadcast("next") deleteEntity() } }
            OnSignal("sortRed") { Actions { updateBar("lives", -1) sound("buzz") broadcast("next") deleteEntity() } }
            OnSignal("sortBlue") { Actions { updateBar("lives", -1) sound("buzz") broadcast("next") deleteEntity() } }
            OnElapsedTime(4) { Actions { updateBar("lives", -1) sound("buzz") broadcast("next") deleteEntity() } }
        }
    }
    Entity {
        id("blueOrb")
        template(true)
        startPosition(0, 0)
        color("3b82f6")
        Type { circle(60) }
        Events {
            OnSignal("sortBlue") { Actions { updateBar("score", 1) broadcast("good") sound("coin") burst(8, "3b82f6") broadcast("next") deleteEntity() } }
            OnSignal("sortRed") { Actions { updateBar("lives", -1) sound("buzz") broadcast("next") deleteEntity() } }
            OnSignal("sortGreen") { Actions { updateBar("lives", -1) sound("buzz") broadcast("next") deleteEntity() } }
            OnElapsedTime(4) { Actions { updateBar("lives", -1) sound("buzz") broadcast("next") deleteEntity() } }
        }
    }

    Entity {
        id("redBasket")
        startPosition(40, 80)
        color("991b1b")
        Type { rectangle(150, 120) }
        Events { OnTapped("redBasket") { Actions { broadcast("sortRed") flash("fecaca", 0.15) } } }
    }
    Entity {
        id("greenBasket")
        startPosition(225, 80)
        color("166534")
        Type { rectangle(150, 120) }
        Events { OnTapped("greenBasket") { Actions { broadcast("sortGreen") flash("bbf7d0", 0.15) } } }
    }
    Entity {
        id("blueBasket")
        startPosition(410, 80)
        color("1e40af")
        Type { rectangle(150, 120) }
        Events { OnTapped("blueBasket") { Actions { broadcast("sortBlue") flash("bfdbfe", 0.15) } } }
    }
}
