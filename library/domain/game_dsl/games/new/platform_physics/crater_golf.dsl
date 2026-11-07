# Golf on the moon: three holes, one shared bag of strokes.
# The ball stops dead where it lands in the dust and is played from there.
Window {
    size(10, 7)
    background("0b1026")
    music("bg_ambient")
    gravity(240)

    Bar("strokes") { initial(15) }

    Level {
        Entity {
            id("ball")
            startPosition(110, 110)
            color("f8fafc")
            Type { circle(12) }
            Movement { speed(0) AutoMove { direction(1, 0) } }
            var("rest", 0)
            Events {
                OnSignal("putt") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(50, 120) changeVar("rest", -1) updateBar("strokes", -1) sound("click") } } } }
                OnSignal("chip") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(90, 240) changeVar("rest", -1) updateBar("strokes", -1) sound("click") } } } }
                OnSignal("lob") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(100, 400) changeVar("rest", -1) updateBar("strokes", -1) sound("click") } } } }
                OnSignal("drive") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(180, 300) changeVar("rest", -1) updateBar("strokes", -1) sound("jump") } } } }
                OnSignal("back") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(-80, 200) changeVar("rest", -1) updateBar("strokes", -1) sound("click") } } } }
                OnCollision("dust") { Actions {
                    setVelocity(0, 0)
                    whenVar("rest", "==", 0) { Actions {
                        changeVar("rest", 1) sound("thud") burst(8, "a8a29e")
                        whenBar("strokes", "<=", 0) { Actions { sound("lose") gameOver("-- OUT OF STROKES --") } } } } } }
                OnCollision("pit") { Actions {
                    sound("bump") shake(5, 0.3) updateBar("strokes", -1)
                    setPosition("ball", 110, 110, 0) setVelocity(0, 0) } }
                OnCollision("cup") { Actions { sound("win2") burst(30, "4ade80") shake(5, 0.4) nextLevel() } }
                OnCollision { Actions { sound("bump") bounceBack(0) } }
            }
        }

        Entity { id("tee") startPosition(0, 80) color("64748b") oneWay(true) Type { rectangle(300, 20) } }
        Entity { id("fairway") startPosition(450, 80) color("64748b") oneWay(true) Type { rectangle(300, 20) } }
        Entity { id("cup") startPosition(810, 90) color("22c55e") Type { rectangle(150, 20) } }
        Entity { id("dust") template(true) startPosition(0, 0) color("00000000") Type { rectangle(150, 14) } }
        Entity { id("pit") startPosition(0, 0) color("1e1b4b") Type { rectangle(1000, 30) }
            Events { OnElapsedTime { Actions {
                spawn("dust", 0, 94) spawn("dust", 150, 94) spawn("dust", 450, 94) spawn("dust", 600, 94) } } } }

        Entity { id("btnPutt") startPosition(20, 640) color("cbd5e1") Type { text("PUTT", 34) }
            Events { OnTapped("btnPutt") { Actions { broadcast("putt") } } } }
        Entity { id("btnChip") startPosition(170, 640) color("fde68a") Type { text("CHIP", 34) }
            Events { OnTapped("btnChip") { Actions { broadcast("chip") } } } }
        Entity { id("btnLob") startPosition(320, 640) color("a5f3fc") Type { text("LOB", 34) }
            Events { OnTapped("btnLob") { Actions { broadcast("lob") } } } }
        Entity { id("btnDrive") startPosition(450, 640) color("fca5a5") Type { text("DRIVE", 34) }
            Events { OnTapped("btnDrive") { Actions { broadcast("drive") } } } }
        Entity { id("btnBack") startPosition(620, 640) color("c4b5fd") Type { text("BACK", 34) }
            Events { OnTapped("btnBack") { Actions { broadcast("back") } } } }
        Entity { id("holeLbl") startPosition(820, 646) color("64748b") Type { text("HOLE 1/3", 26) } }
    }

    Level {
        Entity {
            id("ball")
            startPosition(60, 110)
            color("f8fafc")
            Type { circle(12) }
            Movement { speed(0) AutoMove { direction(1, 0) } }
            var("rest", 0)
            Events {
                OnSignal("putt") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(50, 120) changeVar("rest", -1) updateBar("strokes", -1) sound("click") } } } }
                OnSignal("chip") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(90, 240) changeVar("rest", -1) updateBar("strokes", -1) sound("click") } } } }
                OnSignal("lob") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(100, 400) changeVar("rest", -1) updateBar("strokes", -1) sound("click") } } } }
                OnSignal("drive") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(180, 300) changeVar("rest", -1) updateBar("strokes", -1) sound("jump") } } } }
                OnSignal("back") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(-80, 200) changeVar("rest", -1) updateBar("strokes", -1) sound("click") } } } }
                OnCollision("dust") { Actions {
                    setVelocity(0, 0)
                    whenVar("rest", "==", 0) { Actions {
                        changeVar("rest", 1) sound("thud") burst(8, "a8a29e")
                        whenBar("strokes", "<=", 0) { Actions { sound("lose") gameOver("-- OUT OF STROKES --") } } } } } }
                OnCollision("pit") { Actions {
                    sound("bump") shake(5, 0.3) updateBar("strokes", -1)
                    setPosition("ball", 60, 110, 0) setVelocity(0, 0) } }
                # the mesa's cliff face sends a short shot straight back
                OnCollision("mesa") { Actions { sound("thud") bounceBack(0) } }
                OnCollision("boulder") { Actions { sound("bump") bounceBack() burst(6, "94a3b8") } }
                OnCollision("cup") { Actions { sound("win2") burst(30, "4ade80") shake(5, 0.4) nextLevel() } }
                OnCollision { Actions { sound("bump") bounceBack(0) } }
            }
        }

        Entity { id("tee") startPosition(0, 80) color("64748b") oneWay(true) Type { rectangle(450, 20) } }
        Entity { id("mesa") startPosition(600, 0) color("475569") oneWay(true) Type { rectangle(300, 320) } }
        Entity { id("cup") startPosition(750, 310) color("22c55e") Type { rectangle(150, 20) } }
        Entity { id("boulder") startPosition(450, 200) color("94a3b8")
            Type { circle(30) }
            Movement { speed(1) Path { waypoint(450, 200) waypoint(450, 480) dwell(0.6) } } }
        Entity { id("dust") template(true) startPosition(0, 0) color("00000000") Type { rectangle(150, 14) } }
        Entity { id("pit") startPosition(0, 0) color("1e1b4b") Type { rectangle(1000, 30) }
            Events { OnElapsedTime { Actions {
                spawn("dust", 0, 94) spawn("dust", 150, 94) spawn("dust", 300, 94) spawn("dust", 600, 314) } } } }

        Entity { id("btnPutt") startPosition(20, 640) color("cbd5e1") Type { text("PUTT", 34) }
            Events { OnTapped("btnPutt") { Actions { broadcast("putt") } } } }
        Entity { id("btnChip") startPosition(170, 640) color("fde68a") Type { text("CHIP", 34) }
            Events { OnTapped("btnChip") { Actions { broadcast("chip") } } } }
        Entity { id("btnLob") startPosition(320, 640) color("a5f3fc") Type { text("LOB", 34) }
            Events { OnTapped("btnLob") { Actions { broadcast("lob") } } } }
        Entity { id("btnDrive") startPosition(450, 640) color("fca5a5") Type { text("DRIVE", 34) }
            Events { OnTapped("btnDrive") { Actions { broadcast("drive") } } } }
        Entity { id("btnBack") startPosition(620, 640) color("c4b5fd") Type { text("BACK", 34) }
            Events { OnTapped("btnBack") { Actions { broadcast("back") } } } }
        Entity { id("holeLbl") startPosition(820, 646) color("64748b") Type { text("HOLE 2/3", 26) } }
    }

    Level {
        Entity {
            id("ball")
            startPosition(100, 110)
            color("f8fafc")
            Type { circle(12) }
            Movement { speed(0) AutoMove { direction(1, 0) } }
            var("rest", 0)
            Events {
                OnSignal("putt") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(50, 120) changeVar("rest", -1) updateBar("strokes", -1) sound("click") } } } }
                OnSignal("chip") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(90, 240) changeVar("rest", -1) updateBar("strokes", -1) sound("click") } } } }
                OnSignal("lob") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(100, 400) changeVar("rest", -1) updateBar("strokes", -1) sound("click") } } } }
                OnSignal("drive") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(180, 300) changeVar("rest", -1) updateBar("strokes", -1) sound("jump") } } } }
                OnSignal("back") { Actions { whenVar("rest", "==", 1) { Actions {
                    applyImpulse(-80, 200) changeVar("rest", -1) updateBar("strokes", -1) sound("click") } } } }
                OnCollision("dust") { Actions {
                    setVelocity(0, 0)
                    whenVar("rest", "==", 0) { Actions {
                        changeVar("rest", 1) sound("thud") burst(8, "a8a29e")
                        whenBar("strokes", "<=", 0) { Actions { sound("lose") gameOver("-- OUT OF STROKES --") } } } } } }
                OnCollision("pit") { Actions {
                    sound("bump") shake(5, 0.3) updateBar("strokes", -1)
                    setPosition("ball", 100, 110, 0) setVelocity(0, 0) } }
                OnCollision("rover") { Actions { sound("clang") bounceBack() flash("rover", "ffffff", 0.2) } }
                OnCollision("cup") { Actions {
                    sound("win") burst(40, "4ade80") shake(8, 0.6) gameOver("-- CRATER CHAMPION --") } }
                OnCollision { Actions { sound("bump") bounceBack(0) } }
            }
        }

        Entity { id("tee") startPosition(0, 80) color("64748b") oneWay(true) Type { rectangle(300, 20) } }
        Entity { id("island") startPosition(420, 240) color("64748b") oneWay(true) Type { rectangle(200, 20) } }
        Entity { id("farside") startPosition(700, 80) color("64748b") oneWay(true) Type { rectangle(300, 20) } }
        Entity { id("cup") startPosition(850, 90) color("22c55e") Type { rectangle(150, 20) } }
        Entity { id("rover") startPosition(700, 100) color("e2e8f0")
            Type { rectangle(60, 40) }
            Movement { speed(0.8) Path { waypoint(700, 100) waypoint(800, 100) dwell(1) } } }
        Entity { id("dust") template(true) startPosition(0, 0) color("00000000") Type { rectangle(150, 14) } }
        Entity { id("pit") startPosition(0, 0) color("1e1b4b") Type { rectangle(1000, 30) }
            Events { OnElapsedTime { Actions {
                spawn("dust", 0, 94) spawn("dust", 150, 94) spawn("dust", 420, 254) spawn("dust", 470, 254) spawn("dust", 700, 94) } } } }

        Entity { id("btnPutt") startPosition(20, 640) color("cbd5e1") Type { text("PUTT", 34) }
            Events { OnTapped("btnPutt") { Actions { broadcast("putt") } } } }
        Entity { id("btnChip") startPosition(170, 640) color("fde68a") Type { text("CHIP", 34) }
            Events { OnTapped("btnChip") { Actions { broadcast("chip") } } } }
        Entity { id("btnLob") startPosition(320, 640) color("a5f3fc") Type { text("LOB", 34) }
            Events { OnTapped("btnLob") { Actions { broadcast("lob") } } } }
        Entity { id("btnDrive") startPosition(450, 640) color("fca5a5") Type { text("DRIVE", 34) }
            Events { OnTapped("btnDrive") { Actions { broadcast("drive") } } } }
        Entity { id("btnBack") startPosition(620, 640) color("c4b5fd") Type { text("BACK", 34) }
            Events { OnTapped("btnBack") { Actions { broadcast("back") } } } }
        Entity { id("holeLbl") startPosition(820, 646) color("64748b") Type { text("HOLE 3/3", 26) } }
    }
}
