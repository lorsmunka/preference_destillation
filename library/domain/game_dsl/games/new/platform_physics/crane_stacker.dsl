Window {
    size(7, 12)
    background("bae6fd")
    music("bg_2")
    gravity(650)

    Bar("floors") { initial(0) }
    winAt("floors", 8, "-- TOPPED OUT --", "win2")

    # the crane trolley runs along the jib; every drop hangs a crate under it
    Entity { id("trolley") startPosition(40, 1080) color("f59e0b")
        Type { rectangle(100, 20) }
        Movement { speed(2.2) Path { waypoint(40, 1080) waypoint(560, 1080) } }
        Events {
            OnSignal("drop") { Actions { sound("click") spawnHere("crate") } }
            OnSignal("placed") { Actions {
                whenBar("floors", ">=", 3) { Actions { updateSpeed(2.9) } }
                whenBar("floors", ">=", 6) { Actions { updateSpeed(3.6) } } } } } }

    Entity { id("dropBtn") startPosition(500, 1130) color("ef4444")
        Type { rectangle(180, 60) }
        var("ready", 1)
        Events { OnTapped("dropBtn") { Actions {
            whenVar("ready", "==", 1) { Actions {
                broadcast("drop")
                changeVar("ready", -1)
                setOpacity(0.4)
                delay(1.3) { Actions { changeVar("ready", 1) setOpacity(1) } } } } } } } }
    Entity { id("dropLbl") startPosition(545, 1145) color("ffffff") Type { text("DROP", 30) } }

    # crates are one-way platforms, so each one becomes the floor for the next
    Entity { id("crate") template(true) startPosition(0, 0) color("b45309") oneWay(true)
        Type { rectangle(100, 60) }
        Movement { speed(0) AutoMove { direction(0, -1) } }
        Events {
            OnElapsedTime(2.4) { Actions { sound("thud") updateBar("floors", 1) burst(10, "fde68a") broadcast("placed") } }
            OnCollision("mud") { Actions { sound("death") shake(10, 0.6) burst(30, "78350f") restart() } }
            OnCollision("wrecker") { Actions {
                sound("clang") shake(8, 0.5) burst(24, "f97316")
                gameOver("-- THE WRECKING BALL SMASHED A CRATE --") } } } }

    Entity { id("foundation") startPosition(300, 100) color("475569") oneWay(true)
        Type { rectangle(100, 40) } }
    Entity { id("mud") startPosition(0, 0) color("78350f")
        Type { rectangle(700, 100) }
        Events { OnElapsedTime(2) { repeatCount(-1) Actions { burst(3, "a16207") } } } }

    # the neighbour's demolition ball swings across the drop zone
    Entity { id("wrecker") startPosition(40, 860) color("1f2937")
        Type { circle(36) }
        Movement { speed(3) Path {
            waypoint(40, 860) waypoint(170, 790) waypoint(310, 760) waypoint(450, 790) waypoint(590, 860)
            waypoint(450, 790) waypoint(310, 760) waypoint(170, 790) } } }
}
