Window {
    size(8, 8)
    background("0c0a1f")
    music("bg_1")
    gravity(0)
    lives(2, "-- POPPED --", "lose")
    Bar("beats") { initial(0) }
    winAt("beats", 10, "-- PARTY SURVIVED --", "win2")
    joystick("spark", 80, 5.5)

    Entity { id("spark") startPosition(385, 120) color("fef08a") Type { star(4, 16, 7) }
        Movement { speed(0) Control { movementControl(0) } } }

    # every 5 s the cannon flashes, then fires one more bouncer into the room
    Entity { id("cannon") startPosition(370, 760) color("c026d3") Type { rectangle(60, 40) }
        Events { OnElapsedTime(5) { repeatCount(10) Actions {
            updateBar("beats", 1) flash("ffffff", 0.4) sound("beep")
            delay(0.5) { Actions { sound("pop") spawn("bouncer", 380, 700) } } } } } }

    Entity { id("bouncer") template(true) startPosition(0, 0) color("22d3ee") Type { circle(22) }
        Movement { speed(3) AutoMove { direction(1, -1) } }
        Events {
            OnElapsedTime { Actions {
                randomAction {
                    Actions { setVelocity(220, -260) setColor("22d3ee") }
                    Actions { setVelocity(-220, -260) setColor("f472b6") }
                    Actions { setVelocity(260, -180) setColor("a3e635") }
                    Actions { setVelocity(-260, -180) setColor("fb923c") }
                }
                updateSpeed(random(2.6, 4.2)) } }
            OnCollision { Actions { bounceBack(0) burst(3, "e9d5ff") } }
            OnCollision("spark") { Actions {
                sound("hurt") updateBar("lives", -1) flash("spark", "ff0000", 0.4)
                shake(7, 0.35) burst(18, "f472b6") deleteEntity() } }
            OnSignal("chill") { Actions { updateSpeed(1.6) setColor("94a3b8") } }
        } }

    # a slow-motion record drops in now and then; grabbing it calms every bouncer in the room
    Entity {
        id("record")
        template(true)
        startPosition(0, 0)
        color("e5e7eb")
        Type { ring(20, 6) }
        Events {
            OnCollision("spark") {
                Actions {
                    sound("sparkle")
                    burst(20, "e5e7eb")
                    broadcast("chill")
                    deleteEntity()
                }
            }
            OnElapsedTime(4) { Actions { deleteEntity() } }
        }
    }

    Entity { id("dj") startPosition(0, 0) color("00000000") Type { circle(1) }
        Events { OnElapsedTime(13) { repeatCount(-1) Actions {
            spawn("record", random(60, 720), random(60, 640)) } } } }
}
