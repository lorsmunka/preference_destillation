# Siege Catapult: every rock flies the same arc, so you aim by sliding the catapult inside its yard.
# Tap the sky to throw. Knock down every tower of the fort, then the stone keep, while arrows rain back.
Window {
    size(12, 8)
    background("bae6fd")
    music("bg_2")
    gravity(500)

    lives(4, "-- THE CATAPULT IS SPLINTERED --", "lose")
    Bar("towers") { initial(0) }

    Level {
        Entity { id("sky") startPosition(0, 240) color("00000000") Type { rectangle(1200, 560) }
            Events { OnTapped("sky") { Actions { broadcast("launch") } } } }
        Entity { id("fence") startPosition(440, 0) color("78716c") Type { rectangle(20, 70) } solid(true) }

        Entity { id("catapult") startPosition(150, 0) color("92400e") Type { rectangle(100, 50) } gravity(0)
            var("ready", 1)
            Movement { speed(5) Control { movementControl(2) } }
            Events { OnSignal("launch") { Actions { whenVar("ready", "==", 1) { Actions {
                spawnHere("rock") changeVar("ready", -1) sound("thud") setColor("57534e")
                delay(1.1) { Actions { changeVar("ready", 1) setColor("92400e") } } } } } } } }

        Entity { id("rock") template(true) color("57534e") Type { circle(14) }
            Movement { speed(1) AutoMove { direction(3.3, 5.6) } }
            Events { OnCollision { Actions { burst(6, "a8a29e") deleteEntity() } } } }

        Entity { id("arrow") template(true) color("1c1917") Type { rectangle(26, 6) }
            Movement { speed(4) AutoMove { direction(-1, 0.8) } }
            Events {
                OnElapsedTime { Actions { setVelocity(random(-430, -300), random(250, 380)) } }
                OnCollision("catapult") { Actions {
                    sound("hurt") updateBar("lives", -1) flash("catapult", "ff0000", 0.3) shake(6, 0.3) deleteEntity() } }
                OnCollision { Actions { deleteEntity() } } } }

        Entity { id("towerA") startPosition(780, 0) color("a16207") Type { rectangle(90, 240) }
            var("hp", 2)
            Events {
                OnElapsedTime(random(2.5, 4.5)) { repeatCount(-1) Actions { spawnHere("arrow") sound("blip") } }
                OnCollision("rock") { Actions {
                    deleteCollider() changeVar("hp", -1) flash("ffffff", 0.2) shake(5, 0.3) sound("thud") burst(12, "a16207") setOpacity(0.7)
                    whenVar("hp", "<=", 0) { Actions { broadcast("towerDown") sound("glass") burst(30, "a16207") deleteEntity() } } } } } }
        Entity { id("towerB") startPosition(980, 0) color("a16207") Type { rectangle(90, 180) }
            var("hp", 2)
            Events {
                OnElapsedTime(random(3, 5)) { repeatCount(-1) Actions { spawnHere("arrow") sound("blip") } }
                OnCollision("rock") { Actions {
                    deleteCollider() changeVar("hp", -1) flash("ffffff", 0.2) shake(5, 0.3) sound("thud") burst(12, "a16207") setOpacity(0.7)
                    whenVar("hp", "<=", 0) { Actions { broadcast("towerDown") sound("glass") burst(30, "a16207") deleteEntity() } } } } } }

        Entity { id("judge") startPosition(0, 790) color("00000000") Type { circle(1) }
            var("left", 2)
            Events { OnSignal("towerDown") { Actions {
                changeVar("left", -1) updateBar("towers", 1)
                whenVar("left", "==", 0) { Actions { sound("win2") delay(1.5) { Actions { nextLevel() } } } } } } } }
    }

    Level {
        Entity { id("sky") startPosition(0, 240) color("00000000") Type { rectangle(1200, 560) }
            Events { OnTapped("sky") { Actions { broadcast("launch") } } } }
        Entity { id("fence") startPosition(440, 0) color("78716c") Type { rectangle(20, 70) } solid(true) }

        Entity { id("catapult") startPosition(150, 0) color("92400e") Type { rectangle(100, 50) } gravity(0)
            var("ready", 1)
            Movement { speed(5) Control { movementControl(2) } }
            Events { OnSignal("launch") { Actions { whenVar("ready", "==", 1) { Actions {
                spawnHere("rock") changeVar("ready", -1) sound("thud") setColor("57534e")
                delay(1.1) { Actions { changeVar("ready", 1) setColor("92400e") } } } } } } } }

        # a wooden palisade rides up and down mid-field and swallows rocks that hit it
        Entity { id("palisade") startPosition(600, 0) color("a16207") Type { rectangle(24, 140) }
            Movement { speed(1.2) Path { waypoint(600, 0) waypoint(600, 330) } } }

        Entity { id("rock") template(true) color("57534e") Type { circle(14) }
            Movement { speed(1) AutoMove { direction(3.3, 5.6) } }
            Events {
                OnCollision("palisade") { Actions { sound("bump") burst(8, "a16207") deleteEntity() } }
                OnCollision { Actions { burst(6, "a8a29e") deleteEntity() } } } }

        Entity { id("arrow") template(true) color("1c1917") Type { rectangle(26, 6) }
            Movement { speed(4) AutoMove { direction(-1, 0.8) } }
            Events {
                OnElapsedTime { Actions { setVelocity(random(-520, -380), random(300, 420)) } }
                OnCollision("catapult") { Actions {
                    sound("hurt") updateBar("lives", -1) flash("catapult", "ff0000", 0.3) shake(6, 0.3) deleteEntity() } }
                OnCollision { Actions { deleteEntity() } } } }

        Entity { id("towerC") startPosition(760, 0) color("9ca3af") Type { rectangle(80, 200) }
            var("hp", 3)
            Events {
                OnElapsedTime(random(2.5, 4)) { repeatCount(-1) Actions { spawnHere("arrow") sound("blip") } }
                OnCollision("rock") { Actions {
                    deleteCollider() changeVar("hp", -1) flash("ffffff", 0.2) shake(5, 0.3) sound("clang") burst(12, "9ca3af")
                    whenVar("hp", "==", 1) { Actions { setOpacity(0.6) } }
                    whenVar("hp", "<=", 0) { Actions { broadcast("towerDown") sound("glass") burst(30, "9ca3af") deleteEntity() } } } } } }
        Entity { id("towerD") startPosition(900, 0) color("6b7280") Type { rectangle(80, 280) }
            var("hp", 3)
            Events {
                OnElapsedTime(random(2, 3.5)) { repeatCount(-1) Actions { spawnHere("arrow") sound("blip") } }
                OnCollision("rock") { Actions {
                    deleteCollider() changeVar("hp", -1) flash("ffffff", 0.2) shake(5, 0.3) sound("clang") burst(12, "6b7280")
                    whenVar("hp", "==", 1) { Actions { setOpacity(0.6) } }
                    whenVar("hp", "<=", 0) { Actions { broadcast("towerDown") sound("glass") burst(30, "6b7280") deleteEntity() } } } } } }
        Entity { id("towerE") startPosition(1040, 0) color("9ca3af") Type { rectangle(80, 160) }
            var("hp", 3)
            Events {
                OnElapsedTime(random(3, 5)) { repeatCount(-1) Actions { spawnHere("arrow") sound("blip") } }
                OnCollision("rock") { Actions {
                    deleteCollider() changeVar("hp", -1) flash("ffffff", 0.2) shake(5, 0.3) sound("clang") burst(12, "9ca3af")
                    whenVar("hp", "==", 1) { Actions { setOpacity(0.6) } }
                    whenVar("hp", "<=", 0) { Actions { broadcast("towerDown") sound("glass") burst(30, "9ca3af") deleteEntity() } } } } } }

        Entity { id("judge") startPosition(0, 790) color("00000000") Type { circle(1) }
            var("left", 3)
            Events { OnSignal("towerDown") { Actions {
                changeVar("left", -1) updateBar("towers", 1)
                whenVar("left", "==", 0) { Actions { sound("win") shake(8, 0.6) gameOver("-- THE KEEP HAS FALLEN --") } } } } } }
    }
}
