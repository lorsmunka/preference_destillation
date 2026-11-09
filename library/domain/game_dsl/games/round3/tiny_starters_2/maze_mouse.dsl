Window {
    size(8, 10)
    background("1c1917")
    joystick("mouse", 70, 5)
    wallBox(0, 0, 800, 1000, "57534e", 20)
    Entity {
        id("mouse")
        startPosition(60, 160)
        color("d6d3d1")
        Type { circle(22) }
        Movement { speed(0) Control { movementControl(0) } }
        Events {
            OnCollision("cheese") { Actions { sound("win") burst(12, "fde047") gameOver("CHEESE!") } }
        }
    }
    Entity {
        id("inner")
        startPosition(20, 500)
        color("78716c")
        solid(true)
        Type { rectangle(560, 24) }
    }
    Entity {
        id("inner2")
        startPosition(220, 760)
        color("78716c")
        solid(true)
        Type { rectangle(560, 24) }
    }
    Entity {
        id("cheese")
        startPosition(680, 880)
        color("fde047")
        Type { triangle(50, "right") }
    }
    Entity {
        id("cat")
        startPosition(600, 300)
        color("fb923c")
        Type { circle(30) }
        Movement { speed(1.5) chase("mouse") }
        hostile("mouse", "THE CAT POUNCED", "lose")
    }
}
