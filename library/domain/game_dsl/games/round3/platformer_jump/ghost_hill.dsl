Window {
    size(7, 10)
    worldSize(7, 20)
    cameraFollow("kid")
    background("0f172a")
    music("bg_ambient")
    gravity(750)
    lives(3, "HAUNTED")
    winAt("candles", 4, "THE HILL IS LIT", "win")

    Entity {
        id("kid")
        startPosition(300, 100)
        color("fb923c")
        Type { rectangle(32, 44) }
        Movement { speed(6) Control { movementControl(2) } }
        Events {
            OnCollision("ghost1") { Actions {
                sound("hurt")
                flash("ffffff", 0.3)
                updateBar("lives", -1)
                setPosition("kid", 300, 100, 0)
            } }
            OnCollision("ghost2") { Actions {
                sound("hurt")
                flash("ffffff", 0.3)
                updateBar("lives", -1)
                setPosition("kid", 300, 100, 0)
            } }
        }
    }

    Entity { id("swipeCatcher") startPosition(0, 0) color("00000000")
        Type { circle(1) }
        Events { OnSwipe("up") { Actions {
            sound("jump")
            applyImpulse("kid", 0, 560)
        } } } }

    Entity { id("grass") startPosition(0, 0) color("1e293b")
        Type { rectangle(700, 60) } solid(true) }

    Entity { id("step1") startPosition(380, 200) color("64748b")
        Type { rectangle(220, 16) } oneWay(true) }
    Entity { id("step2") startPosition(60, 340) color("64748b")
        Type { rectangle(220, 16) } oneWay(true) }
    Entity { id("step3") startPosition(380, 490) color("64748b")
        Type { rectangle(220, 16) } oneWay(true) }
    Entity { id("step4") startPosition(60, 640) color("64748b")
        Type { rectangle(220, 16) } oneWay(true) }
    Entity { id("step5") startPosition(380, 790) color("64748b")
        Type { rectangle(220, 16) } oneWay(true) }
    Entity { id("step6") startPosition(60, 940) color("64748b")
        Type { rectangle(220, 16) } oneWay(true) }
    Entity { id("step7") startPosition(380, 1090) color("64748b")
        Type { rectangle(220, 16) } oneWay(true) }
    Entity { id("step8") startPosition(60, 1240) color("64748b")
        Type { rectangle(220, 16) } oneWay(true) }
    Entity { id("step9") startPosition(380, 1390) color("64748b")
        Type { rectangle(220, 16) } oneWay(true) }
    Entity { id("summit") startPosition(100, 1560) color("94a3b8")
        Type { rectangle(300, 16) } oneWay(true) }

    Entity { id("ghost1") startPosition(100, 400) color("e2e8f0")
        Type { circle(22) }
        Movement { speed(1.3) Path {
            waypoint(80, 420)
            waypoint(560, 560)
            waypoint(80, 700)
        } } }
    Entity { id("ghost2") startPosition(500, 1000) color("cbd5e1")
        Type { circle(22) }
        Movement { speed(1.8) Path {
            waypoint(500, 1000)
            waypoint(100, 1150)
            waypoint(500, 1300)
            dwell(0.5)
        } } }

    Entity { id("candle1") startPosition(470, 216) color("fde047")
        Type { rectangle(10, 26) }
        Events { OnCollision("kid") { Actions {
            sound("pickup")
            updateBar("candles", 1)
            deleteEntity()
        } } } }
    Entity { id("candle2") startPosition(150, 356) color("fde047")
        Type { rectangle(10, 26) }
        Events { OnCollision("kid") { Actions {
            sound("pickup")
            updateBar("candles", 1)
            deleteEntity()
        } } } }
    Entity { id("candle3") startPosition(470, 806) color("fde047")
        Type { rectangle(10, 26) }
        Events { OnCollision("kid") { Actions {
            sound("pickup")
            updateBar("candles", 1)
            deleteEntity()
        } } } }
    Entity { id("candle4") startPosition(200, 1576) color("fde047")
        Type { rectangle(10, 26) }
        Events { OnCollision("kid") { Actions {
            sound("pickup")
            updateBar("candles", 1)
            deleteEntity()
        } } } }
}
