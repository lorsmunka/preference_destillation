# Storm Bosses: a paper plane against three tantrum-throwing weather bosses, one per level.
# Drag the plane, tap the sky to fire; the ammo bar refills by itself.
Window {
    size(8, 13)
    background("1e3a5f")
    music("bg_1")
    gravity(0)

    lives(5, "-- THE PLANE GOT SOGGY --", "lose")
    Bar("ammo") { initial(8) }
    Bar("stars") { initial(0) }

    # Level 1: the Grumpy Cloud rains on you and telegraphs lightning by darkening
    Level {
        Entity { id("sky") startPosition(0, 400) color("00000000") Type { rectangle(800, 900) }
            Events { OnTapped("sky") { Actions { broadcast("fire") } } } }

        Entity { id("plane") startPosition(370, 80) color("f8fafc") Type { triangle(60, "up") }
            Movement { speed(7) Control { movementControl(0) } }
            Events {
                OnSignal("fire") { Actions { whenBar("ammo", ">", 0) { Actions { spawnHere("pellet") updateBar("ammo", -1) sound("laser") } } } }
                OnElapsedTime(0.45) { repeatCount(-1) Actions { whenBar("ammo", "<", 8) { Actions { updateBar("ammo", 1) } } } }
            } }

        Entity { id("pellet") template(true) color("fde047") Type { circle(7) }
            Movement { speed(9) AutoMove { direction(0, 1) } }
            Events { OnCollision { Actions { deleteEntity() } } } }

        Entity { id("cloud") startPosition(40, 1080) color("cbd5e1") Type { circle(90) }
            var("hp", 20)
            Movement { speed(1.2) Path { waypoint(40, 1080) waypoint(580, 1080) } }
            Events {
                Sequence {
                    Phase(5, 1.8) { Actions { spawnHere("drop") } }
                    Phase(4, 2.6) { Actions { setColor("475569") delay(0.7) { Actions {
                        spawnHere("zap") setColor("cbd5e1") sound("thud") shake(4, 0.2) } } } }
                    Phase(-1, 1.4) { Actions { randomAction {
                        Actions { spawnHere("drop") spawnHere("drop") }
                        Actions { setColor("475569") delay(0.6) { Actions { spawnHere("zap") setColor("cbd5e1") sound("thud") } } } } } }
                }
                OnCollision("pellet") { Actions {
                    deleteCollider() changeVar("hp", -1) updateBar("stars", 1) flash("ffffff", 0.1)
                    whenVar("hp", "==", 10) { Actions { setScale(0.8) updateSpeed(2.2) sound("buzz") } }
                    whenVar("hp", "==", 0) { Actions { sound("win2") burst(40, "cbd5e1") setOpacity(0.2)
                        delay(1.5) { Actions { nextLevel() } } } } } }
            } }

        Entity { id("drop") template(true) color("60a5fa") Type { circle(9) }
            Movement { speed(4) AutoMove { direction(0, -1) } }
            Events {
                OnElapsedTime { Actions { setVelocity(random(-60, 60), -400) } }
                OnCollision("plane") { Actions { sound("hurt") updateBar("lives", -1) flash("plane", "ff0000", 0.3) shake(4, 0.2) deleteEntity() } }
                OnCollision { Actions { deleteEntity() } } } }

        Entity { id("zap") template(true) color("fde047") Type { rectangle(14, 120) }
            Movement { speed(9) AutoMove { direction(0, -1) } }
            Events {
                OnCollision("plane") { Actions { sound("hurt") updateBar("lives", -1) flash("plane", "ff0000", 0.3) shake(7, 0.3) deleteEntity() } }
                OnCollision { Actions { deleteEntity() } } } }
    }

    # Level 2: the Twister wanders, flings homing debris and sucks the plane up with gusts
    Level {
        Entity { id("sky") startPosition(0, 400) color("00000000") Type { rectangle(800, 900) }
            Events { OnTapped("sky") { Actions { broadcast("fire") } } } }

        Entity { id("plane") startPosition(370, 80) color("f8fafc") Type { triangle(60, "up") }
            Movement { speed(7) Control { movementControl(0) } }
            Events {
                OnSignal("fire") { Actions { whenBar("ammo", ">", 0) { Actions { spawnHere("pellet") updateBar("ammo", -1) sound("laser") } } } }
                OnElapsedTime(0.45) { repeatCount(-1) Actions { whenBar("ammo", "<", 8) { Actions { updateBar("ammo", 1) } } } }
            } }

        Entity { id("pellet") template(true) color("fde047") Type { circle(7) }
            Movement { speed(9) AutoMove { direction(0, 1) } }
            Events { OnCollision { Actions { deleteEntity() } } } }

        Entity { id("twister") startPosition(60, 1100) color("a8a29e") Type { triangle(160, "down") }
            var("hp", 26)
            Movement { speed(1.5) Path {
                waypoint(60, 1100) waypoint(560, 1000) waypoint(320, 860) waypoint(600, 1130) waypoint(80, 900) } }
            Events {
                Sequence {
                    Phase(3, 2) { Actions { spawnHere("debris") } }
                    Phase(-1, 1.5) { Actions { randomAction {
                        Actions { spawnHere("debris") }
                        Actions { spawnHere("leaf") spawnHere("leaf") }
                        Actions { spawnHere("debris") spawnHere("leaf") } } } }
                }
                OnElapsedTime(6) { repeatCount(-1) Actions {
                    sound("buzz") shake(3, 1.2) flash("e7e5e4", 1.2)
                    setVelocity("plane", 0, 150)
                    delay(1.2) { Actions { setVelocity("plane", 0, 0) } } } }
                OnCollision("pellet") { Actions {
                    deleteCollider() changeVar("hp", -1) updateBar("stars", 1) flash("ffffff", 0.1)
                    whenVar("hp", "==", 13) { Actions { setColor("57534e") updateSpeed(2.4) sound("buzz") } }
                    whenVar("hp", "==", 0) { Actions { sound("win2") burst(40, "a8a29e") setOpacity(0.2)
                        delay(1.5) { Actions { nextLevel() } } } } } }
                OnCollision("plane") { Actions { sound("hurt") updateBar("lives", -1) flash("plane", "ff0000", 0.3) shake(6, 0.3) } }
            } }

        Entity { id("debris") template(true) color("78716c") Type { polygon(4, 16) }
            Movement { speed(1.6) chase("plane") }
            Events {
                OnCollision("pellet") { Actions { deleteCollider() sound("pop") burst(6, "78716c") deleteEntity() } }
                OnCollision("plane") { Actions { sound("hurt") updateBar("lives", -1) flash("plane", "ff0000", 0.3) shake(4, 0.2) deleteEntity() } }
                OnElapsedTime(5) { Actions { burst(6, "a8a29e") deleteEntity() } } } }

        Entity { id("leaf") template(true) color("65a30d") Type { diamond(28, 16) }
            Movement { speed(2.5) AutoMove { direction(1, -1) } }
            Events {
                OnElapsedTime { Actions { setVelocity(random(-220, 220), random(-200, -120)) } }
                OnCollision { Actions { bounceBack(0) } }
                OnCollision("plane") { Actions { sound("hurt") updateBar("lives", -1) flash("plane", "ff0000", 0.3) deleteEntity() } }
                OnElapsedTime(6) { Actions { setOpacity(0.3) delay(0.5) { Actions { deleteEntity() } } } } } }
    }

    # Level 3: the Hail King fans hail, drops ice slab walls with one gap and sends snowflakes after you
    Level {
        Entity { id("sky") startPosition(0, 400) color("00000000") Type { rectangle(800, 900) }
            Events { OnTapped("sky") { Actions { broadcast("fire") } } } }

        Entity { id("plane") startPosition(370, 80) color("f8fafc") Type { triangle(60, "up") }
            Movement { speed(7) Control { movementControl(0) } }
            Events {
                OnSignal("fire") { Actions { whenBar("ammo", ">", 0) { Actions { spawnHere("pellet") updateBar("ammo", -1) sound("laser") } } } }
                OnElapsedTime(0.45) { repeatCount(-1) Actions { whenBar("ammo", "<", 8) { Actions { updateBar("ammo", 1) } } } }
            } }

        Entity { id("pellet") template(true) color("fde047") Type { circle(7) }
            Movement { speed(9) AutoMove { direction(0, 1) } }
            Events { OnCollision { Actions { deleteEntity() } } } }

        Entity { id("king") startPosition(100, 1100) color("e0f2fe") Type { star(8, 80, 46) }
            var("hp", 34)
            Movement { speed(1) Path { waypoint(100, 1100) waypoint(540, 1100) } }
            Events {
                Sequence {
                    Phase(4, 2) { Actions { sound("blip") spawnHere("hail") spawnHere("hail") spawnHere("hail") } }
                    Phase(2, 3) { Actions { broadcast("slabs") } }
                    Phase(-1, 1.5) { Actions { randomAction {
                        Actions { sound("blip") spawnHere("hail") spawnHere("hail") spawnHere("hail") }
                        Actions { sound("bubble") spawnHere("flake") }
                        Actions { broadcast("slabs") } } } }
                }
                OnCollision("pellet") { Actions {
                    deleteCollider() changeVar("hp", -1) updateBar("stars", 1) flash("ffffff", 0.1)
                    whenVar("hp", "==", 17) { Actions { broadcast("blizzard") } }
                    whenVar("hp", "==", 0) { Actions { sound("win") shake(10, 0.8) burst(50, "e0f2fe")
                        gameOver("-- CLEAR SKIES AT LAST --") } } } }
                OnSignal("blizzard") { Actions { sound("buzz") shake(6, 0.6) setColor("93c5fd") updateSpeed(1.8) spawnHere("flake") spawnHere("flake") } }
            } }

        Entity { id("frost") startPosition(0, 1290) color("00000000") Type { rectangle(800, 10) }
            Events { OnSignal("slabs") { Actions { sound("thud") randomAction {
                Actions { spawn("slab", 160, 1250) spawn("slab", 320, 1250) spawn("slab", 480, 1250) spawn("slab", 640, 1250) }
                Actions { spawn("slab", 0, 1250) spawn("slab", 320, 1250) spawn("slab", 480, 1250) spawn("slab", 640, 1250) }
                Actions { spawn("slab", 0, 1250) spawn("slab", 160, 1250) spawn("slab", 480, 1250) spawn("slab", 640, 1250) }
                Actions { spawn("slab", 0, 1250) spawn("slab", 160, 1250) spawn("slab", 320, 1250) spawn("slab", 640, 1250) }
                Actions { spawn("slab", 0, 1250) spawn("slab", 160, 1250) spawn("slab", 320, 1250) spawn("slab", 480, 1250) } } } } } }

        Entity { id("hail") template(true) color("f1f5f9") Type { circle(12) }
            Movement { speed(3.5) AutoMove { direction(0, -1) } }
            Events {
                OnElapsedTime { Actions { setVelocity(random(-160, 160), -340) } }
                OnCollision("plane") { Actions { sound("hurt") updateBar("lives", -1) flash("plane", "ff0000", 0.3) shake(4, 0.2) deleteEntity() } }
                OnCollision { Actions { deleteEntity() } } } }

        Entity { id("slab") template(true) color("bae6fd") Type { rectangle(150, 24) }
            Movement { speed(2.6) AutoMove { direction(0, -1) } }
            Events {
                OnCollision("plane") { Actions { sound("glass") updateBar("lives", -1) flash("plane", "ff0000", 0.3) shake(7, 0.4) deleteEntity() } }
                OnCollision { Actions { deleteEntity() } } } }

        Entity { id("flake") template(true) color("bae6fd") Type { star(6, 18, 8) }
            Movement { speed(1.4) chase("plane") }
            Events {
                OnCollision("pellet") { Actions { deleteCollider() sound("pop") burst(8, "bae6fd") deleteEntity() } }
                OnCollision("plane") { Actions { sound("hurt") updateBar("lives", -1) flash("plane", "ff0000", 0.3) deleteEntity() } }
                OnElapsedTime(5) { Actions { burst(6, "e0f2fe") deleteEntity() } } } }
    }
}
