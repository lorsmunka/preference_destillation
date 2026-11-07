# Snowball Fort: tap the snow pile to pack snowballs, tap the sky to throw them at the rival kids
# who pop up behind their wall. Your own fort walls crumble under their iceballs.
Window {
    size(9, 12)
    background("e0f2fe")
    music("bg_6")
    gravity(0)

    lives(3, "-- SNOWED UNDER --", "lose")
    Bar("snowballs") { initial(3) }
    Bar("hits") {
        initial(0)
        OnValue(5) { Actions { sound("buzz") shake(4, 0.3) spawn("boulder", random(100, 720), 1010) } }
        OnValue(10) { Actions { sound("buzz") shake(4, 0.3) spawn("boulder", random(100, 720), 1010) } }
        OnValue(15) { Actions { sound("win") shake(6, 0.5) burst("kid", 30, "ffffff") gameOver("-- KING OF THE FORT --") } }
    }

    Entity { id("sky") startPosition(0, 450) color("00000000") Type { rectangle(900, 750) }
        Events { OnTapped("sky") { Actions { broadcast("throw") } } } }

    Entity { id("pile") startPosition(760, 320) color("ffffff") Type { rectangle(120, 70) }
        Events {
            OnTapped("pile") {
                Actions {
                    whenBar("snowballs", "<", 6) { Actions { updateBar("snowballs", 1) sound("click") burst(5, "bfdbfe") } }
                }
            }
        } }

    Entity { id("kid") startPosition(415, 30) color("2563eb") Type { rectangle(70, 70) }
        Movement { speed(7) Control { movementControl(2) } }
        Events {
            OnSignal("throw") {
                Actions {
                    whenBar("snowballs", "==", 0) { Actions { sound("buzz") flash("pile", "60a5fa", 0.3) } }
                    whenBar("snowballs", ">", 0) { Actions { spawnHere("snowball") updateBar("snowballs", -1) sound("jump") } }
                }
            }
            OnElapsedTime {
                Actions { spawn("wall", 120, 170) spawn("wall", 560, 170) }
            }
        } }

    Entity { id("snowball") template(true) color("ffffff") Type { circle(12) }
        Movement { speed(6.5) AutoMove { direction(0, 1) } }
        Events { OnCollision { Actions { deleteEntity() } } } }

    Entity { id("wall") template(true) color("f8fafc") Type { rectangle(220, 60) }
        var("hp", 4)
        Events {
            OnCollision("iceball") {
                Actions {
                    deleteCollider()
                    changeVar("hp", -1)
                    sound("bump")
                    burst(8, "e2e8f0")
                    whenVar("hp", "==", 3) { Actions { setOpacity(0.8) } }
                    whenVar("hp", "==", 2) { Actions { setOpacity(0.6) } }
                    whenVar("hp", "==", 1) { Actions { setOpacity(0.35) } }
                    whenVar("hp", "<=", 0) { Actions { sound("glass") deleteEntity() } }
                }
            }
        } }

    # the rival kids are declared before their wall so the wall hides them while they duck
    Entity { id("r1") startPosition(140, 905) color("dc2626") Type { rectangle(60, 60) }
        var("up", 0)
        Events {
            OnElapsedTime(random(2, 5)) { repeatCount(-1) Actions { whenVar("up", "==", 0) { Actions {
                changeVar("up", 1) setPosition("r1", 140, 975, 0.2)
                delay(0.7) { Actions { whenVar("up", "==", 1) { Actions { spawnHere("iceball") sound("blip") } } } }
                delay(1.8) { Actions { whenVar("up", "==", 1) { Actions { changeVar("up", -1) setPosition("r1", 140, 905, 0.2) } } } } } } } }
            OnCollision("snowball") { Actions { whenVar("up", "==", 1) { Actions {
                deleteCollider() updateBar("hits", 1) sound("pop") flash("ffffff", 0.2) burst(10, "ffffff")
                changeVar("up", -1) setPosition("r1", 140, 905, 0.2) } } } }
        } }
    Entity { id("r2") startPosition(420, 905) color("ea580c") Type { rectangle(60, 60) }
        var("up", 0)
        Events {
            OnElapsedTime(random(2, 5)) { repeatCount(-1) Actions { whenVar("up", "==", 0) { Actions {
                changeVar("up", 1) setPosition("r2", 420, 975, 0.2)
                delay(0.7) { Actions { whenVar("up", "==", 1) { Actions { spawnHere("iceball") sound("blip") } } } }
                delay(1.8) { Actions { whenVar("up", "==", 1) { Actions { changeVar("up", -1) setPosition("r2", 420, 905, 0.2) } } } } } } } }
            OnCollision("snowball") { Actions { whenVar("up", "==", 1) { Actions {
                deleteCollider() updateBar("hits", 1) sound("pop") flash("ffffff", 0.2) burst(10, "ffffff")
                changeVar("up", -1) setPosition("r2", 420, 905, 0.2) } } } }
        } }
    Entity { id("r3") startPosition(700, 905) color("db2777") Type { rectangle(60, 60) }
        var("up", 0)
        Events {
            OnElapsedTime(random(2, 5)) { repeatCount(-1) Actions { whenVar("up", "==", 0) { Actions {
                changeVar("up", 1) setPosition("r3", 700, 975, 0.2)
                delay(0.7) { Actions { whenVar("up", "==", 1) { Actions { spawnHere("iceball") sound("blip") } } } }
                delay(1.8) { Actions { whenVar("up", "==", 1) { Actions { changeVar("up", -1) setPosition("r3", 700, 905, 0.2) } } } } } } } }
            OnCollision("snowball") { Actions { whenVar("up", "==", 1) { Actions {
                deleteCollider() updateBar("hits", 1) sound("pop") flash("ffffff", 0.2) burst(10, "ffffff")
                changeVar("up", -1) setPosition("r3", 700, 905, 0.2) } } } }
        } }

    Entity { id("rivalWall") startPosition(0, 900) color("cbd5e1") Type { rectangle(900, 70) } }

    Entity { id("iceball") template(true) color("93c5fd") Type { circle(14) }
        Movement { speed(3.2) AutoMove { direction(0, -1) } }
        Events {
            OnElapsedTime { Actions { setVelocity(random(-80, 80), -320) } }
            OnCollision("kid") { Actions {
                sound("hurt") updateBar("lives", -1) flash("kid", "ff0000", 0.3) shake(5, 0.3) burst(10, "93c5fd") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        } }

    Entity { id("boulder") template(true) color("f1f5f9") Type { circle(42) }
        var("hp", 3)
        Movement { speed(1.2) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("snowball") { Actions {
                deleteCollider() changeVar("hp", -1) sound("thud") flash("93c5fd", 0.15)
                whenVar("hp", "==", 2) { Actions { setScale(0.8) } }
                whenVar("hp", "==", 1) { Actions { setScale(0.6) } }
                whenVar("hp", "<=", 0) { Actions { burst(24, "ffffff") updateBar("hits", 1) deleteEntity() } } } }
            OnCollision("wall") { Actions { deleteCollider() sound("glass") shake(5, 0.3) burst(16, "e2e8f0") } }
            OnCollision("kid") { Actions {
                sound("hurt") updateBar("lives", -1) flash("kid", "ff0000", 0.3) shake(8, 0.4) deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        } }
}
