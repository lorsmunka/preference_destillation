# Orchard archery: apples swing along their branches; shoot ahead of them.
Window {
    size(8, 12)
    background("ecfccb")
    music("bg_4")

    Bar("apples") { initial(0) }
    winAt("apples", 10, "-- BASKET FULL --", "win")
    Bar("quiver") {
        initial(15)
        OnValue(3) { Actions { setText("shoot", "3 ARROWS LEFT") } }
        OnValue(1) { Actions { setText("shoot", "LAST ARROW") flash("bow", "ef4444", 0.4) } }
        OnValue(0) { Actions { delay(1.2) { Actions {
            whenBar("apples", "<", 10) { Actions { sound("lose") gameOver("-- OUT OF ARROWS --") } }
        } } } }
    }

    Entity { id("bow") startPosition(370, 140) color("78350f") Type { triangle(60, "up") } }

    Entity { id("shoot") startPosition(250, 40) color("365314") Type { text("TAP TO SHOOT", 52) }
        Events {
            OnTapped("shoot") { Actions {
                whenBar("quiver", ">", 0) { Actions {
                    sound("blip")
                    updateBar("quiver", -1)
                    spawn("arrow", 396, 210)
                    flash("bow", "fbbf24", 0.1)
                } }
            } }
        } }

    Entity { id("arrow") template(true) startPosition(0, 0) color("451a03")
        Type { rectangle(8, 50) }
        Movement { speed(9) AutoMove { direction(0, 1) } }
        Events {
            OnCollision("wasp") { Actions { sound("buzz") burst(8, "facc15") flash("wasp", "ef4444", 0.3) deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        } }

    Entity { id("apple1") startPosition(40, 520) color("dc2626") Type { circle(30) }
        Movement { speed(2) Path { waypoint(40, 520) waypoint(700, 520) } }
        Events { OnCollision("arrow") { Actions {
            sound("coin") burst(10, "dc2626") deleteCollider()
            updateBar("apples", 1)
            setEnable(false)
            delay(1.5) { Actions { setEnable(true) } }
        } } } }

    Entity { id("apple2") startPosition(700, 760) color("16a34a") Type { circle(30) }
        Movement { speed(3) Path { waypoint(700, 760) waypoint(40, 760) dwell(0.4) } }
        Events { OnCollision("arrow") { Actions {
            sound("coin") burst(10, "16a34a") deleteCollider()
            updateBar("apples", 1)
            setEnable(false)
            delay(1.5) { Actions { setEnable(true) } }
        } } } }

    Entity { id("golden") startPosition(40, 1010) color("facc15") Type { circle(26) }
        Movement { speed(4.5) Path { waypoint(40, 1010) waypoint(710, 1010) } }
        Events { OnCollision("arrow") { Actions {
            sound("sparkle") burst(18, "facc15") shake(4, 0.2) deleteCollider()
            setEnable(false)
            delay(3) { Actions { setEnable(true) } }
            loopCount(2)
            updateBar("apples", 1)
        } } } }

    Entity { id("wasp") startPosition(100, 640) color("a16207") Type { star(6, 22, 10) }
        Movement { speed(2.5) Path { waypoint(100, 640) waypoint(640, 880) waypoint(320, 900) waypoint(620, 620) } } }
}
