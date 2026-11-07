Window {
    size(8, 10)
    background("18181b")
    music("bg_ambient")
    gravity(0)

    lives(3, "-- THE LANTERN WENT DARK --", "lose")
    Bar("banished") {
        initial(0)
        OnValue(10) { Actions { broadcast("midnight") } }
        OnValue(20) { Actions { sound("win") shake(6, 0.5) burst("lantern", 40, "fde68a") gameOver("-- THE CRYPT IS QUIET --") } }
    }

    Entity { id("cryptL") startPosition(60, 860) color("52525b") Type { rectangle(80, 110) }
        Events {
            OnElapsedTime(random(2.5, 4)) { repeatCount(-1) Actions { spawnHere("ghost") } }
            OnSignal("midnight") { Actions { sound("bell") flash("a855f7", 0.6) spawnHere("wraith") } }
        } }
    Entity { id("cryptR") startPosition(660, 860) color("52525b") Type { rectangle(80, 110) }
        Events {
            OnElapsedTime(random(2.5, 4)) { repeatCount(-1) Actions { spawnHere("ghost") } }
            OnElapsedTime(14) { repeatCount(-1) Actions { spawnHere("wraith") } }
            OnSignal("midnight") { Actions { shake(6, 0.5) flash("a855f7", 0.6) spawnHere("wraith") } }
        } }

    Entity { id("lantern") startPosition(376, 100) color("fde68a") Type { circle(24) }
        Movement { speed(6) Control { movementControl(0) } }
        Events {
            OnElapsedTime(1.4) { repeatCount(-1) Actions { spawnHere("glow") } }
        } }

    Entity { id("glow") template(true) color("90fde68a") Type { ring(100, 88) }
        Events {
            OnElapsedTime(0.12) { Actions { setOpacity(0.5) } }
            OnElapsedTime(0.25) { Actions { deleteEntity() } }
        } }

    # ghosts drift and flicker; while faded they can neither be hurt nor hurt you
    Entity { id("ghost") template(true) color("e0e7ff") Type { circle(28) }
        var("hp", 2)
        var("seen", 1)
        Movement { speed(1.2) AutoMove { direction(1, -1) } }
        Events {
            OnElapsedTime { Actions { setVelocity(random(-130, 130), random(-120, -40)) } }
            OnCollision { Actions { bounceBack(0) } }
            OnElapsedTime(random(0.8, 2.2)) { repeatCount(-1) Actions { randomAction {
                Actions { whenVar("seen", "==", 0) { Actions { changeVar("seen", 1) setOpacity(1) } } }
                Actions { whenVar("seen", "==", 1) { Actions { changeVar("seen", -1) setOpacity(0.1) } } } } } }
            OnCollision("glow") { Actions { whenVar("seen", "==", 1) { Actions {
                changeVar("hp", -1) flash("fde68a", 0.2) sound("sparkle")
                whenVar("hp", "<=", 0) { Actions { sound("pop") burst(16, "e0e7ff") updateBar("banished", 1) deleteEntity() } } } } } }
            OnCollision("lantern") { Actions { whenVar("seen", "==", 1) { Actions {
                sound("hurt") updateBar("lives", -1) flash("lantern", "ff0000", 0.3) shake(6, 0.3) burst(12, "e0e7ff") deleteEntity() } } } }
        } }

    # wraiths hunt the lantern and hurt it even while faded; a hit sends them reeling back to the crypts
    Entity { id("wraith") template(true) color("a855f7") Type { star(6, 30, 18) }
        var("hp", 4)
        var("seen", 1)
        Movement { speed(0.7) chase("lantern") }
        Events {
            OnElapsedTime(random(1, 2.5)) { repeatCount(-1) Actions { randomAction {
                Actions { whenVar("seen", "==", 0) { Actions { changeVar("seen", 1) setOpacity(1) } } }
                Actions { whenVar("seen", "==", 1) { Actions { changeVar("seen", -1) setOpacity(0.15) } } } } } }
            OnCollision("glow") { Actions { whenVar("seen", "==", 1) { Actions {
                changeVar("hp", -1) flash("fde68a", 0.2) sound("sparkle")
                whenVar("hp", "<=", 0) { Actions { sound("pop") burst(24, "a855f7") updateBar("banished", 1) deleteEntity() } } } } } }
            OnCollision("lantern") { Actions {
                sound("hurt") updateBar("lives", -1) flash("lantern", "ff0000", 0.3) shake(7, 0.4)
                burst(14, "a855f7") setPosition("wraith", random(60, 680), 860, 0.6) } }
        } }
}
