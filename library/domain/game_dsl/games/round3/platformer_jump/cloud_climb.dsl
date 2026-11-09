Window {
    size(7, 12)
    background("7dd3fc")
    music("bg_nature")
    gravity(650)
    winAt("coins", 5, "ALL COINS", "win")
    lives(2, "YOU FELL TOO OFTEN")

    Entity {
        id("player")
        startPosition(300, 100)
        color("f43f5e")
        Type { rectangle(36, 50) }
        Movement { speed(6.5) Control { movementControl(2) } }
        Events {
            OnCollision("pit") {
                Actions {
                    sound("lose")
                    updateBar("lives", -1)
                    setPosition("player", 300, 100, 0)
                }
            }
        }
    }

    Entity { id("controller") startPosition(0, 0) color("00000000")
        Type { circle(1) }
        Events {
            OnSwipe("up") { Actions {
                sound("jump")
                applyImpulse("player", 0, 540)
            } }
        }
    }

    Entity { id("ground") startPosition(0, 20) color("16a34a")
        Type { rectangle(700, 40) } oneWay(true) }
    Entity { id("pit") startPosition(-300, -60) color("00000000")
        Type { rectangle(1300, 30) } }

    Entity { id("cloud1") startPosition(80, 230) color("ffffff")
        Type { rectangle(180, 18) } oneWay(true) }
    Entity { id("cloud2") startPosition(430, 360) color("ffffff")
        Type { rectangle(180, 18) } oneWay(true) }
    Entity { id("cloud3") startPosition(100, 500) color("ffffff")
        Type { rectangle(180, 18) } oneWay(true) }

    Entity { id("mover") startPosition(100, 650) color("e0f2fe")
        Type { rectangle(150, 18) } oneWay(true)
        Movement { speed(1.2) Path {
            waypoint(60, 650)
            waypoint(480, 650)
        } } }
    Entity { id("cloud4") startPosition(440, 800) color("ffffff")
        Type { rectangle(180, 18) } oneWay(true) }
    Entity { id("cloud5") startPosition(80, 940) color("ffffff")
        Type { rectangle(180, 18) } oneWay(true) }

    Entity { id("coinA") startPosition(150, 260) color("facc15")
        Type { circle(14) }
        Events { OnCollision("player") { Actions {
            sound("coin")
            updateBar("coins", 1)
            deleteEntity()
        } } } }
    Entity { id("coinB") startPosition(500, 390) color("facc15")
        Type { circle(14) }
        Events { OnCollision("player") { Actions {
            sound("coin")
            updateBar("coins", 1)
            deleteEntity()
        } } } }
    Entity { id("coinC") startPosition(170, 530) color("facc15")
        Type { circle(14) }
        Events { OnCollision("player") { Actions {
            sound("coin")
            updateBar("coins", 1)
            deleteEntity()
        } } } }
    Entity { id("coinD") startPosition(510, 830) color("facc15")
        Type { circle(14) }
        Events { OnCollision("player") { Actions {
            sound("coin")
            updateBar("coins", 1)
            deleteEntity()
        } } } }
    Entity { id("coinE") startPosition(150, 970) color("facc15")
        Type { circle(14) }
        Events { OnCollision("player") { Actions {
            sound("coin")
            updateBar("coins", 1)
            deleteEntity()
        } } } }

    Entity { id("hint") startPosition(150, 1120) color("0c4a6e")
        Type { text("Swipe up to jump, drag to move", 24) } }
}
