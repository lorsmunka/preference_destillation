Window {
    size(8, 10)
    background("1e293b")
    music("bg_2")
    winAt(3, "THE CAT WINS", "win")
    Entity {
        id("cat")
        startPosition(380, 480)
        color("fb923c")
        Type { circle(32) }
        Movement { speed(6) Control { movementControl(0) } }
    }
    Entity {
        id("mouseA")
        startPosition(120, 800)
        color("d4d4d8")
        Type { circle(18) }
        Movement { speed(3) AutoMove { direction(1, -1) } }
        Events {
            OnCollision { Actions { bounceBack() } }
            OnCollision("cat") { Actions { sound("pop") burst(8, "d4d4d8") updateBar(1) deleteEntity() } }
        }
    }
    Entity {
        id("mouseB")
        startPosition(640, 700)
        color("a8a29e")
        Type { circle(18) }
        Movement { speed(4) AutoMove { direction(-1, 1) } }
        Events {
            OnCollision { Actions { bounceBack() } }
            OnCollision("cat") { Actions { sound("pop") burst(8, "a8a29e") updateBar(1) deleteEntity() } }
        }
    }
    Entity {
        id("mouseC")
        startPosition(400, 200)
        color("e7e5e4")
        Type { circle(18) }
        Movement { speed(5) AutoMove { direction(1, 1) } }
        Events {
            OnCollision { Actions { bounceBack() } }
            OnCollision("cat") { Actions { sound("pop") burst(8, "e7e5e4") updateBar(1) deleteEntity() } }
        }
    }
    Entity {
        id("dog")
        startPosition(40, 40)
        color("78350f")
        Type { rectangle(60, 44) }
        Movement { speed(2) chase("cat") }
        hostile("cat", "THE DOG GOT YOU", "lose")
    }
}
