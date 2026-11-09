# Notes fall down three lanes in a rhythm. Tap each one before it hits the floor.
Window {
    size(6, 12)
    background("0b1020")
    music("bg_4")
    lives(5, "THE SONG FELL APART", "lose")
    winAt("score", 20, "SONG COMPLETE", "win")

    Entity { id("floor") startPosition(0, 40) color("1e293b") Type { rectangle(600, 20) } }
    Entity { id("laneLeft") startPosition(100, 60) color("1e3a8a") Type { rectangle(4, 1100) } }
    Entity { id("laneMid") startPosition(300, 60) color("1e3a8a") Type { rectangle(4, 1100) } }
    Entity { id("laneRight") startPosition(500, 60) color("1e3a8a") Type { rectangle(4, 1100) } }

    Entity {
        id("noteLeft")
        template(true)
        startPosition(0, 0)
        color("f472b6")
        Type { circle(45) }
        Movement { speed(3) AutoMove { direction(0, -1) } }
        Events {
            OnTapped { Actions { updateBar("score", 1) sound("pickup") burst(8, "f472b6") deleteEntity() } }
            OnCollision("floor") { Actions { updateBar("lives", -1) sound("thud") deleteEntity() } }
        }
    }
    Entity {
        id("noteMid")
        template(true)
        startPosition(0, 0)
        color("facc15")
        Type { rectangle(80, 80) }
        Movement { speed(4) AutoMove { direction(0, -1) } }
        Events {
            OnTapped { Actions { updateBar("score", 1) sound("coin") burst(8, "facc15") deleteEntity() } }
            OnCollision("floor") { Actions { updateBar("lives", -1) sound("thud") deleteEntity() } }
        }
    }
    Entity {
        id("noteRight")
        template(true)
        startPosition(0, 0)
        color("38bdf8")
        Type { diamond(80, 90) }
        Movement { speed(3.5) AutoMove { direction(0, -1) } }
        Events {
            OnTapped { Actions { updateBar("score", 1) sound("bell") burst(8, "38bdf8") deleteEntity() } }
            OnCollision("floor") { Actions { updateBar("lives", -1) sound("thud") deleteEntity() } }
        }
    }

    Entity {
        id("conductor")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(1, 1.0) { Actions { spawn("noteLeft", 55, 1100) } }
                Phase(1, 0.8) { Actions { spawn("noteMid", 260, 1100) } }
                Phase(1, 0.8) { Actions { spawn("noteRight", 455, 1100) } }
                Phase(1, 0.8) { Actions { spawn("noteMid", 260, 1100) } }
                Phase(2, 0.7) { Actions { spawn("noteLeft", 55, 1100) } }
                Phase(2, 0.7) { Actions { spawn("noteRight", 455, 1100) } }
                Phase(3, 0.5) { Actions { spawn("noteMid", 260, 1100) } }
                Phase(-1, 0.6) { Actions {
                    randomAction {
                        Actions { spawn("noteLeft", 55, 1100) }
                        Actions { spawn("noteMid", 260, 1100) }
                        Actions { spawn("noteRight", 455, 1100) }
                    }
                } }
            }
        }
    }
}
