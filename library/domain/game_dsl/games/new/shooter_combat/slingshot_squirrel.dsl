# Slingshot Squirrel: the pouch fills while you wait (acorn, chestnut, coconut). Tap the sky to fling
# whatever is loaded at the crows that pelt you with pinecones.
Window {
    size(8, 12)
    background("bae6fd")
    music("bg_1")
    gravity(0)

    winAt("crows", 10, "-- THE OAK IS SAFE --", "win")
    lives(3, "-- BONKED --", "lose")

    Entity { id("sky") startPosition(0, 220) color("00000000") Type { rectangle(800, 980) }
        Events { OnTapped("sky") { Actions { broadcast("fling") } } } }

    Entity { id("pouch") startPosition(720, 1110) color("b45309") Type { circle(16) }
        Events { OnElapsedTime { Actions { setScale(0.5) } } } }

    Entity { id("squirrel") startPosition(360, 40) color("a16207") Type { rectangle(80, 56) }
        Movement { speed(7) Control { movementControl(2) } }
        var("charge", 0)
        Events {
            OnElapsedTime(0.6) { repeatCount(-1) Actions {
                whenVar("charge", "<", 3) { Actions { changeVar("charge", 1) } }
                whenVar("charge", "==", 1) { Actions { setScale("pouch", 1) } }
                whenVar("charge", "==", 2) { Actions { setScale("pouch", 1.4) } }
                whenVar("charge", "==", 3) { Actions { setScale("pouch", 1.8) flash("pouch", "fde047", 0.2) } } } }
            OnSignal("fling") { Actions {
                whenVar("charge", "==", 3) { Actions { spawnHere("coconut") changeVar("charge", -3) sound("jump") } }
                whenVar("charge", "==", 2) { Actions { spawnHere("chestnut") changeVar("charge", -2) sound("jump") } }
                whenVar("charge", "==", 1) { Actions { spawnHere("acorn") changeVar("charge", -1) sound("jump") } }
                setScale("pouch", 0.5) } }
        } }

    Entity { id("acorn") template(true) color("d97706") Type { circle(9) }
        Movement { speed(7) AutoMove { direction(0, 1) } }
        Events { OnCollision { Actions { deleteEntity() } } } }
    Entity { id("chestnut") template(true) color("7c2d12") Type { circle(15) }
        Movement { speed(6) AutoMove { direction(0, 1) } }
        Events { OnCollision { Actions { deleteEntity() } } } }
    Entity { id("coconut") template(true) color("57534e") Type { circle(24) }
        Movement { speed(5) AutoMove { direction(0, 1) } }
        Events { OnCollision { Actions { deleteEntity() } } } }

    Entity { id("crow") template(true) color("1f2937") Type { triangle(48, "down") }
        var("hp", 3)
        Movement { speed(2) AutoMove { direction(1, 0) } }
        Events {
            OnElapsedTime { Actions { randomAction { Actions { setVelocity(200, 0) } Actions { setVelocity(-200, 0) } } } }
            OnCollision { Actions { bounceBack(0) } }
            OnElapsedTime(random(1.8, 3.6)) { repeatCount(-1) Actions { spawnHere("pinecone") } }
            OnCollision("acorn") { Actions { changeVar("hp", -1) deleteCollider() flash("ffffff", 0.1) sound("bump") } }
            OnCollision("chestnut") { Actions { changeVar("hp", -2) deleteCollider() flash("ffffff", 0.1) sound("thud") } }
            OnCollision("coconut") { Actions { changeVar("hp", -3) shake(4, 0.2) sound("thud") } }
            OnElapsedTime(0) { repeatCount(-1) Actions { whenVar("hp", "<=", 0) { Actions {
                sound("pop") burst(16, "374151") updateBar("crows", 1) deleteEntity() } } } }
        } }

    Entity { id("pinecone") template(true) color("92400e") Type { triangle(22, "down") }
        Movement { speed(3.2) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("squirrel") { Actions {
                sound("hurt") updateBar("lives", -1) flash("squirrel", "ff0000", 0.25) shake(5, 0.3) burst(10, "92400e") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } } } }

    Entity { id("flock") startPosition(0, 0) color("00000000") Type { circle(1) }
        Events { OnElapsedTime(2.8) { repeatCount(-1) Actions { spawn("crow", random(0, 750), random(640, 1100)) } } } }
}
