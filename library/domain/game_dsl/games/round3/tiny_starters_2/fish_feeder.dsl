Window {
    size(8, 12)
    background("0c4a6e")
    music("bg_ambient")
    lives(3, "THE FISH GOT STUNG", "lose")
    winAt("score", 8, "WELL FED", "win")
    Entity {
        id("fish")
        startPosition(360, 40)
        color("fb923c")
        Type { polygon(5, 36) }
        Movement { speed(6) Control { movementControl(2) } }
    }
    Entity {
        id("food")
        template(true)
        startPosition(0, 0)
        color("a3e635")
        Type { circle(18) }
        Movement { speed(3) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("fish") { Actions { sound("pickup") updateBar("score", 1) deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }
    Entity {
        id("jelly")
        template(true)
        startPosition(0, 0)
        color("e879f9")
        Type { star(6, 30, 14) }
        Movement { speed(4) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("fish") { Actions { sound("hurt") shake(5, 0.3) updateBar("lives", -1) deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }
    Entity {
        id("feeder")
        startPosition(0, 1150)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.8) { repeatCount(-1) Actions { spawn("food", random(40, 740), 1120) } }
            OnElapsedTime(2.1) { repeatCount(-1) Actions { spawn("jelly", random(40, 740), 1120) } }
        }
    }
}
