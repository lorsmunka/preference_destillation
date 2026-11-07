Window {
    size(8, 12)
    background("38bdf8")
    music("bg_1")
    gravity(420)

    lives(3, "-- GROUNDED FOR GOOD --", "lose")
    Bar("points") { initial(0) }
    winAt("points", 15, "-- GOLD WINGS --", "win")

    # any tap that is not on the diver pulls the ripcord
    Entity { id("air") startPosition(0, 0) color("00000000")
        Type { rectangle(800, 1200) }
        Events { OnTapped("air") { Actions { broadcast("pull") } } } }

    Entity { id("lowline") startPosition(0, 360) color("80ffffff")
        Type { rectangle(800, 8) } }
    Entity { id("lowLbl") startPosition(20, 372) color("e0f2fe")
        Type { text("open below this line for +2", 22) } }

    Entity {
        id("diver")
        startPosition(380, 1120)
        color("f97316")
        Type { diamond(40, 46) }
        Movement { speed(4) Control { movementControl(2) } }
        var("open", 0)
        var("brave", 0)
        Events {
            OnSignal("pull") { Actions { whenVar("open", "==", 0) { Actions {
                changeVar("open", 1)
                setColor("fde047")
                setVelocity(0, -90)
                sound("pop")
                burst(12, "fde047") } } } }
            # an open canopy pins the fall to a slow drift; in free fall the wind shoves you about
            OnElapsedTime(0.1) { repeatCount(-1) Actions {
                whenVar("open", "==", 1) { Actions { setVelocity(0, -90) } } } }
            OnElapsedTime(0.8) { repeatCount(-1) Actions {
                whenVar("open", "==", 0) { Actions { applyImpulse(random(-70, 70), 0) } } } }
            OnCollision("lowline") { Actions { whenVar("open", "==", 0) { Actions {
                changeVar("brave", 1)
                flash("ffffff", 0.2)
                sound("blip") } } } }
            OnCollision("gull") { Actions { whenVar("open", "==", 1) { Actions {
                changeVar("open", -1)
                setColor("f97316")
                sound("buzz")
                burst(10, "ffffff") } } } }
            OnCollision("bull") { Actions {
                whenVar("open", "==", 0) { Actions { broadcast("crash") } }
                whenVar("open", "==", 1) { Actions {
                    sound("coin")
                    updateBar("points", 3)
                    burst(20, "fde047")
                    whenVar("brave", "==", 1) { Actions { sound("sparkle") updateBar("points", 2) } }
                    broadcast("again") } } } }
            OnCollision("innerL") { Actions {
                whenVar("open", "==", 0) { Actions { broadcast("crash") } }
                whenVar("open", "==", 1) { Actions {
                    sound("coin")
                    updateBar("points", 1)
                    burst(12, "fde047")
                    whenVar("brave", "==", 1) { Actions { sound("sparkle") updateBar("points", 2) } }
                    broadcast("again") } } } }
            OnCollision("innerR") { Actions {
                whenVar("open", "==", 0) { Actions { broadcast("crash") } }
                whenVar("open", "==", 1) { Actions {
                    sound("coin")
                    updateBar("points", 1)
                    burst(12, "fde047")
                    whenVar("brave", "==", 1) { Actions { sound("sparkle") updateBar("points", 2) } }
                    broadcast("again") } } } }
            OnCollision("fieldL") { Actions {
                whenVar("open", "==", 0) { Actions { broadcast("crash") } }
                whenVar("open", "==", 1) { Actions { sound("bump") broadcast("again") } } } }
            OnCollision("fieldR") { Actions {
                whenVar("open", "==", 0) { Actions { broadcast("crash") } }
                whenVar("open", "==", 1) { Actions { sound("bump") broadcast("again") } } } }
            OnSignal("crash") { Actions {
                sound("hurt")
                shake(8, 0.5)
                burst(20, "ef4444")
                updateBar("lives", -1)
                broadcast("again") } }
            OnSignal("again") { Actions {
                setPosition("diver", random(80, 680), 1120, 0)
                setVelocity(0, 0)
                setColor("f97316")
                whenVar("open", "==", 1) { Actions { changeVar("open", -1) } }
                whenVar("brave", "==", 1) { Actions { changeVar("brave", -1) } } } }
        }
    }

    Entity { id("gull") startPosition(0, 680) color("ffffff") facing("left")
        Type { text("🐦", 40) }
        Movement { speed(2.2) Path { waypoint(0, 680) waypoint(740, 680) } } }

    Entity { id("fieldL") startPosition(0, 0) color("16a34a") Type { rectangle(230, 24) } }
    Entity { id("innerL") startPosition(230, 0) color("fde047") Type { rectangle(120, 24) } }
    Entity { id("bull") startPosition(350, 0) color("ef4444") Type { rectangle(100, 24) } }
    Entity { id("innerR") startPosition(450, 0) color("fde047") Type { rectangle(120, 24) } }
    Entity { id("fieldR") startPosition(570, 0) color("16a34a") Type { rectangle(230, 24) } }
}
