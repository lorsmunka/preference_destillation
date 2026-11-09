Window {
    size(8, 12)
    background("7dd3fc")
    gravity(700)
    music("bg_6")
    winAt(5, "YOU FLEW HOME", "win")
    Entity {
        id("bird")
        startPosition(200, 700)
        color("facc15")
        Type { circle(26) }
        Movement { speed(0) AutoMove { direction(0, 0) } }
        Events {
            OnCollision("pipeLow") { Actions { sound("hurt") gameOver("HIT THE PIPE") } }
            OnCollision("pipeHigh") { Actions { sound("hurt") gameOver("HIT THE PIPE") } }
            OnCollision { Actions { sound("lose") gameOver("CRASHED") } }
        }
    }
    Entity {
        id("swiper")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSwipe("up") { Actions { setVelocity("bird", 0, 380) sound("jump") } }
            OnSwipe("down") { Actions { setVelocity("bird", 0, -300) } }
        }
    }
    Entity {
        id("pipeLow")
        startPosition(700, 0)
        color("16a34a")
        gravity(0)
        Type { rectangle(90, 520) }
        Movement { speed(3) AutoMove { direction(-1, 0) } }
        Events {
            OnCollision { Actions { setPosition("pipeLow", 700, 0, 0) updateBar(1) sound("coin") } }
        }
    }
    Entity {
        id("pipeHigh")
        startPosition(1100, 800)
        color("15803d")
        gravity(0)
        Type { rectangle(90, 400) }
        Movement { speed(3) AutoMove { direction(-1, 0) } }
        Events {
            OnCollision { Actions { setPosition("pipeHigh", 1100, 800, 0) } }
        }
    }
    Entity {
        id("tip")
        startPosition(20, 1150)
        color("0c4a6e")
        Type { text("Swipe up to flap", 30) }
    }
}
