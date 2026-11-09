Window {
    size(8, 8)
    background("082f49")
    winAt(1, "-- FOUND THE TREASURE --", "win")

    Entity {
        id("sailor")
        startPosition(100, 100)
        color("fde68a")
        Type { circle(40) }
        Movement { GridMove(100, 0.3) { direction(0, 0) } }
        Events {
            OnSwipe("up") { Actions { setDirection("sailor", 0, 1) } }
            OnSwipe("down") { Actions { setDirection("sailor", 0, -1) } }
            OnSwipe("left") { Actions { setDirection("sailor", -1, 0) } }
            OnSwipe("right") { Actions { setDirection("sailor", 1, 0) } }
            OnGridBlocked { Actions { sound("lose") gameOver("-- HIT THE EDGE --") } }
        }
    }

    Entity {
        id("rock")
        startPosition(300, 400)
        color("78716c")
        Type { polygon(6, 40) }
        hostile("sailor", "-- WRECKED ON A ROCK --", "hurt")
    }

    Entity {
        id("chest")
        startPosition(600, 600)
        color("f59e0b")
        Type { rectangle(60, 60) }
        Events { OnCollision("sailor") { Actions { sound("coin") updateBar(1) } } }
    }
}
