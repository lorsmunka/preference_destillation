# Circus Juggle: tap a ball to toss it back up before it hits the stage.
# A second ball joins at 10 s, a third at 25 s, and later they shrink.
Window {
    size(8, 12)
    background("3b0764")
    gravity(450)
    music("bg_2")

    Bar("tosses") {
        initial(0)
        OnValue(20) { Actions { setText("banner", "The crowd is cheering!") } }
    }
    winAt("tosses", 45, "-- STANDING OVATION --", "win")
    Bar("drops") {
        initial(0)
        OnValue(1) { Actions { setText("banner", "Careful now...") } }
        OnValue(3) { Actions { sound("lose") shake(8, 0.5) gameOver("-- THE CROWD WENT HOME --") } }
    }

    Entity { id("stage") startPosition(0, 0) color("7c2d12") Type { rectangle(800, 40) } }

    Entity { id("ball1") startPosition(380, 700) color("f43f5e") Type { circle(36) }
        Movement { speed(1) AutoMove { direction(0, 0) } }
        Events {
            OnTapped("ball1") { Actions { sound("jump") setVelocity(random(-150, 150), 640) updateBar("tosses", 1) flash("ffffff", 0.1) } }
            OnCollision { Actions { bounceBack(0) } }
            OnCollision("stage") { Actions {
                sound("thud") shake(5, 0.3) burst(14, "f43f5e") updateBar("drops", 1)
                setVelocity(0, 0) setPosition("ball1", 380, 950, 0)
                setOpacity(0.4) delay(0.8) { Actions { setOpacity(1) } }
            } }
        } }

    Entity { id("ball2") startPosition(150, 950) color("22d3ee") Type { circle(36) }
        Movement { speed(1) AutoMove { direction(0, 0) } }
        Events {
            OnTapped("ball2") { Actions { sound("jump") setVelocity(random(-150, 150), 640) updateBar("tosses", 1) flash("ffffff", 0.1) } }
            OnCollision { Actions { bounceBack(0) } }
            OnCollision("stage") { Actions {
                sound("thud") shake(5, 0.3) burst(14, "22d3ee") updateBar("drops", 1)
                setVelocity(0, 0) setPosition("ball2", 150, 950, 0)
                setOpacity(0.4) delay(0.8) { Actions { setOpacity(1) } }
            } }
        } }

    Entity { id("ball3") startPosition(600, 950) color("facc15") Type { circle(36) }
        Movement { speed(1) AutoMove { direction(0, 0) } }
        Events {
            OnTapped("ball3") { Actions { sound("jump") setVelocity(random(-150, 150), 640) updateBar("tosses", 1) flash("ffffff", 0.1) } }
            OnCollision { Actions { bounceBack(0) } }
            OnCollision("stage") { Actions {
                sound("thud") shake(5, 0.3) burst(14, "facc15") updateBar("drops", 1)
                setVelocity(0, 0) setPosition("ball3", 600, 950, 0)
                setOpacity(0.4) delay(0.8) { Actions { setOpacity(1) } }
            } }
        } }

    Entity { id("balloon") template(true) startPosition(0, 0) color("a3e635") gravity(0)
        Type { star(5, 30, 14) }
        Movement { speed(2) AutoMove { direction(1, 0) } }
        Events {
            OnTapped { Actions { sound("sparkle") burst(16, "a3e635") loopCount(3) updateBar("tosses", 1) deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        } }

    Entity { id("ringmaster") startPosition(20, 1120) color("fde68a") Type { text("Ladies and gentlemen!", 34) }
        Events {
            OnElapsedTime { Actions { setEnable("ball2", false) setEnable("ball3", false) } }
            Sequence {
                Phase(1, 10) { Actions { setEnable("ball2", true) sound("bell") setText("ringmaster", "A second ball!") } }
                Phase(1, 15) { Actions { setEnable("ball3", true) sound("bell") setText("ringmaster", "And a third!") } }
                Phase(1, 15) { Actions {
                    sound("bell") setText("ringmaster", "Smaller balls!")
                    setScale("ball1", 0.75) setScale("ball2", 0.75) setScale("ball3", 0.75)
                } }
            }
            OnElapsedTime(random(6, 10)) { repeatCount(-1) Actions { spawn("balloon", 10, random(800, 1000)) } }
        } }

    Entity { id("banner") startPosition(20, 60) color("f5d0fe") Type { text("Tap the balls to toss them", 34) } }
}
