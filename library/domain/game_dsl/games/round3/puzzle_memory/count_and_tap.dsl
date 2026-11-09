Window {
    size(6, 8)
    background("1a2e1a")
    music("bg_nature")
    Bar("tally") { initial(0) }

    Level {
        Entity {
            id("goal")
            startPosition(120, 690)
            color("fef08a")
            Type { text("Make exactly 3", 48) }
        }
        Entity {
            id("dot")
            template(true)
            startPosition(0, 0)
            color("86efac")
            Type { circle(30) }
        }
        Entity {
            id("plus")
            startPosition(60, 80)
            color("22c55e")
            Type { rectangle(220, 140) }
            Events { OnTapped("plus") { Actions { updateBar("tally", 1) spawn("dot", random(100, 480), random(300, 580)) sound("pop") } } }
        }
        Entity {
            id("check")
            startPosition(320, 80)
            color("f59e0b")
            Type { rectangle(220, 140) }
            Events { OnTapped("check") { Actions {
                whenBar("tally", "!=", 3) { Actions { sound("buzz") restart() } }
                whenBar("tally", "==", 3) { Actions { updateBar("tally", -3) sound("win2") nextLevel() } }
            } } }
        }
        Entity {
            id("timer")
            startPosition(0, 0)
            color("00000000")
            Type { circle(1) }
            Events { OnElapsedTime(15) { Actions { gameOver("TOO SLOW") } } }
        }
    }

    Level {
        Entity {
            id("goal")
            startPosition(120, 690)
            color("fef08a")
            Type { text("Make exactly 5", 48) }
        }
        Entity {
            id("dot")
            template(true)
            startPosition(0, 0)
            color("93c5fd")
            Type { circle(26) }
        }
        Entity {
            id("plus")
            startPosition(60, 80)
            color("3b82f6")
            Type { rectangle(220, 140) }
            Events { OnTapped("plus") { Actions { updateBar("tally", 1) spawn("dot", random(100, 480), random(300, 580)) sound("pop") } } }
        }
        Entity {
            id("check")
            startPosition(320, 80)
            color("f59e0b")
            Type { rectangle(220, 140) }
            Events { OnTapped("check") { Actions {
                whenBar("tally", "!=", 5) { Actions { sound("buzz") restart() } }
                whenBar("tally", "==", 5) { Actions { updateBar("tally", -5) sound("win2") nextLevel() } }
            } } }
        }
        Entity {
            id("timer")
            startPosition(0, 0)
            color("00000000")
            Type { circle(1) }
            Events { OnElapsedTime(15) { Actions { gameOver("TOO SLOW") } } }
        }
    }

    Level {
        Entity {
            id("goal")
            startPosition(120, 690)
            color("fef08a")
            Type { text("Make exactly 8", 48) }
        }
        Entity {
            id("dot")
            template(true)
            startPosition(0, 0)
            color("f9a8d4")
            Type { circle(22) }
        }
        Entity {
            id("plus")
            startPosition(60, 80)
            color("ec4899")
            Type { rectangle(220, 140) }
            Events { OnTapped("plus") { Actions { updateBar("tally", 1) spawn("dot", random(100, 480), random(300, 580)) sound("pop") } } }
        }
        Entity {
            id("check")
            startPosition(320, 80)
            color("f59e0b")
            Type { rectangle(220, 140) }
            Events { OnTapped("check") { Actions {
                whenBar("tally", "!=", 8) { Actions { sound("buzz") restart() } }
                whenBar("tally", "==", 8) { Actions { sound("win") nextLevel() } }
            } } }
        }
        Entity {
            id("timer")
            startPosition(0, 0)
            color("00000000")
            Type { circle(1) }
            Events { OnElapsedTime(20) { Actions { gameOver("TOO SLOW") } } }
        }
    }
}
