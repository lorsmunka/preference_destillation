Window {
    size(6, 7)
    background("14213d")
    music("bg_4")
    Bar("moves") {
        initial(0)
        OnValue(18) { Actions { sound("lose") gameOver("OUT OF MOVES") } }
    }

    Entity {
        id("label")
        startPosition(40, 640)
        color("fca311")
        Type { text("Free the red car", 40) }
    }
    Entity {
        id("hint")
        startPosition(40, 20)
        color("e5e5e5")
        Type { text("Tap a block to push it. 18 moves.", 28) }
        Events {
            OnSignal("moved") { Actions { setText("hint", "Keep going...") } }
        }
    }

    Entity {
        id("exit")
        startPosition(500, 300)
        color("22c55e")
        Type { rectangle(100, 100) }
    }

    Entity {
        id("redCar")
        startPosition(0, 300)
        color("ef4444")
        solid(true)
        Type { rectangle(200, 100) }
        Movement { GridMove(100, 0.15) { direction(0, 0) } }
        Events {
            OnTapped("redCar") { Actions { gridStep(1, 0) updateBar("moves", 1) sound("click") broadcast("moved") } }
            OnCollision("exit") { Actions { sound("win") burst(20, "22c55e") gameOver("YOU ESCAPED") } }
        }
    }
    Entity {
        id("blockA")
        startPosition(300, 200)
        color("60a5fa")
        solid(true)
        Type { rectangle(100, 200) }
        Movement { GridMove(100, 0.15) { direction(0, 0) } }
        Events {
            OnTapped("blockA") { Actions { gridStep(0, 1) updateBar("moves", 1) sound("click") broadcast("moved") } }
        }
    }
    Entity {
        id("blockB")
        startPosition(400, 300)
        color("a78bfa")
        solid(true)
        Type { rectangle(100, 200) }
        Movement { GridMove(100, 0.15) { direction(0, 0) } }
        Events {
            OnTapped("blockB") { Actions { gridStep(0, 1) updateBar("moves", 1) sound("click") broadcast("moved") } }
        }
    }
    Entity {
        id("slider")
        startPosition(300, 500)
        color("fbbf24")
        solid(true)
        Type { rectangle(100, 100) }
        Movement { GridMove(100, 0.15) { direction(0, 0) } }
        Events {
            OnTapped("slider") { Actions { gridStep(1, 0) updateBar("moves", 1) sound("click") broadcast("moved") } }
        }
    }
}
