Window {
    size(8, 12)
    background("e0f2fe")
    music("bg_5")
    gravity(0)

    Bar("goals") { initial(0) }
    winAt("goals", 6, "-- SIX IN THE NET --", "win")
    Bar("shots") { initial(10) }

    wallBox(0, 0, 800, 1200, "1e3a8a", 18)

    Entity {
        id("puck")
        startPosition(384, 150)
        color("111827")
        Type { circle(16) }
        Movement { speed(0) AutoMove { direction(0, 1) } }
        var("ready", 1)
        Events {
            OnSignal("bankL") { Actions { whenVar("ready", "==", 1) { Actions {
                setVelocity(-386, 522)
                broadcast("shot") } } } }
            OnSignal("straight") { Actions { whenVar("ready", "==", 1) { Actions {
                setVelocity(0, 650)
                broadcast("shot") } } } }
            OnSignal("bankR") { Actions { whenVar("ready", "==", 1) { Actions {
                setVelocity(386, 522)
                broadcast("shot") } } } }
            # every shot lasts four seconds, then the whistle brings the puck back
            OnSignal("shot") {
                Actions {
                    sound("click")
                    changeVar("ready", -1)
                    updateBar("shots", -1)
                    delay(4) { Actions { broadcast("whistle") } }
                }
            }
            OnSignal("whistle") {
                Actions {
                    sound("beep")
                    setPosition("puck", 384, 150, 0)
                    setVelocity(0, 0)
                    changeVar("ready", 1)
                    whenBar("shots", "==", 0) { Actions { sound("lose") gameOver("-- OUT OF PUCKS --") } }
                }
            }
            OnCollision("wallL") { Actions { sound("click") bounceBack(0) } }
            OnCollision("wallR") { Actions { sound("click") bounceBack(0) } }
            OnCollision("wallT") { Actions { sound("click") bounceBack(0) } }
            OnCollision("wallB") { Actions { sound("click") bounceBack(0) } }
            OnCollision("goalie") { Actions { sound("thud") bounceBack() flash("goalie", "ffffff", 0.2) } }
            OnCollision("defender") { Actions { sound("bump") bounceBack() burst(6, "94a3b8") } }
            OnCollision("net") {
                Actions {
                    sound("win2")
                    updateBar("goals", 1)
                    burst(26, "ef4444")
                    shake(6, 0.4)
                    setPosition("puck", 384, 150, 0)
                    setVelocity(0, 0)
                }
            }
        }
    }

    Entity { id("net") startPosition(320, 1140) color("ef4444") Type { rectangle(160, 34) } }
    Entity { id("goalie") startPosition(300, 1070) color("1d4ed8")
        Type { rectangle(60, 26) }
        Movement { speed(1.6) Path { waypoint(300, 1070) waypoint(440, 1070) } } }
    Entity { id("defender") startPosition(40, 760) color("64748b")
        Type { rectangle(90, 26) }
        Movement { speed(2.4) Path { waypoint(40, 760) waypoint(670, 760) dwell(0.3) } } }

    Entity { id("padBankL") startPosition(30, 24) color("93c5fd") Type { rectangle(230, 60) }
        Events { OnTapped("padBankL") { Actions { broadcast("bankL") } } } }
    Entity { id("padStraight") startPosition(285, 24) color("bfdbfe") Type { rectangle(230, 60) }
        Events { OnTapped("padStraight") { Actions { broadcast("straight") } } } }
    Entity { id("padBankR") startPosition(540, 24) color("93c5fd") Type { rectangle(230, 60) }
        Events { OnTapped("padBankR") { Actions { broadcast("bankR") } } } }
    Entity { id("lblBankL") startPosition(85, 40) color("1e3a8a") Type { text("< BANK", 28) } }
    Entity { id("lblStraight") startPosition(345, 40) color("1e3a8a") Type { text("STRAIGHT", 28) } }
    Entity { id("lblBankR") startPosition(600, 40) color("1e3a8a") Type { text("BANK >", 28) } }
}
