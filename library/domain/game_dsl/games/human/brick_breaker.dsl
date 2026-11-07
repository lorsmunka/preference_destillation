Window {
    size(10, 8)
    background("0a0e27")
    music("bg_3")
    gravity(0)

    Bar("cleared") {
        initial(0)
        OnValue(24) { Actions { sound("win2") nextLevel() } }
        OnValue(40) { Actions { sound("win") gameOver("-- WALLS DOWN --") } }
    }

    Level {
        Entity { id("titleL1") startPosition(820, 25) color("64748b")
            Type { text("STAGE 1/2", 26) } }
        Entity {
            id("paddle")
            startPosition(440, 30)
            color("00BFFF")
            Type { rectangle(120, 25) }
            Movement { speed(1000) Control { movementControl(2) } }
        }
        Entity {
            id("ball")
            startPosition(490, 400)
            color("FF4500")
            Type { circle(12) }
            Movement { speed(2.55) AutoMove { direction(0, -1) } }
            Events {
                Sequence {
                    Phase(1, 12) { Actions { updateSpeed(3.4) } }
                    Phase(1, 12) { Actions { updateSpeed(4.25) } }
                    Phase(1, 12) { Actions { updateSpeed(5.1) } }
                }
                OnCollision("paddle") {
                    Actions { sound("pop") bounceBack()
                        flash("paddle", "ffffff", 0.1)
                        burst(6, "ffffff") }
                }
                OnCollision("brick1") { Actions { bounceBack() } }
                OnCollision("brick2") { Actions { bounceBack() } }
                OnCollision("brick3") { Actions { bounceBack() } }
                OnCollision("brick4") { Actions { bounceBack() } }
                OnCollision("brick5") { Actions { bounceBack() } }
                OnCollision("brick6") { Actions { bounceBack() } }
                OnCollision("brick7") { Actions { bounceBack() } }
                OnCollision("brick8") { Actions { bounceBack() } }
                OnCollision("brick9") { Actions { bounceBack() } }
                OnCollision("brick10") { Actions { bounceBack() } }
                OnCollision("brick11") { Actions { bounceBack() } }
                OnCollision("brick12") { Actions { bounceBack() } }
                OnCollision("brick13") { Actions { bounceBack() } }
                OnCollision("brick14") { Actions { bounceBack() } }
                OnCollision("brick15") { Actions { bounceBack() } }
                OnCollision("brick16") { Actions { bounceBack() } }
                OnCollision("brick17") { Actions { bounceBack() } }
                OnCollision("brick18") { Actions { bounceBack() } }
                OnCollision("brick19") { Actions { bounceBack() } }
                OnCollision("brick20") { Actions { bounceBack() } }
                OnCollision("brick21") { Actions { bounceBack() } }
                OnCollision("brick22") { Actions { bounceBack() } }
                OnCollision("brick23") { Actions { bounceBack() } }
                OnCollision("brick24") { Actions { bounceBack() } }
                OnCollision { Actions { bounceBack() } }
            }
        }
        Entity { id("bottomEdge") startPosition(0, 0) color("0a0e27")
            Type { rectangle(1000, 15) }
            Events { OnCollision("ball") {
                Actions {
                    sound("hurt")
                    shake(6, 0.35)
                    burst(16, "FF4500")
                    restart()
                } } } }

        Entity { id("brick1") startPosition(20, 720) color("FF6347")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "FF6347")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick2") startPosition(140, 720) color("FFD700")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "FFD700")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick3") startPosition(260, 720) color("00FA9A")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "00FA9A")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick4") startPosition(380, 720) color("8A2BE2")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "8A2BE2")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick5") startPosition(500, 720) color("FF6347")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "FF6347")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick6") startPosition(620, 720) color("FFD700")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "FFD700")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick7") startPosition(740, 720) color("00FA9A")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "00FA9A")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick8") startPosition(860, 720) color("8A2BE2")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "8A2BE2")
                updateBar("cleared", 1) deleteEntity() } } } }

        Entity { id("brick9") startPosition(20, 680) color("FFD700")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "FFD700")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick10") startPosition(140, 680) color("00FA9A")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "00FA9A")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick11") startPosition(260, 680) color("8A2BE2")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "8A2BE2")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick12") startPosition(380, 680) color("FF6347")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "FF6347")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick13") startPosition(500, 680) color("FFD700")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "FFD700")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick14") startPosition(620, 680) color("00FA9A")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "00FA9A")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick15") startPosition(740, 680) color("8A2BE2")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "8A2BE2")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick16") startPosition(860, 680) color("FF6347")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "FF6347")
                updateBar("cleared", 1) deleteEntity() } } } }

        Entity { id("brick17") startPosition(20, 640) color("00FA9A")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "00FA9A")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick18") startPosition(140, 640) color("8A2BE2")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "8A2BE2")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick19") startPosition(260, 640) color("FF6347")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "FF6347")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick20") startPosition(380, 640) color("FFD700")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "FFD700")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick21") startPosition(500, 640) color("00FA9A")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "00FA9A")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick22") startPosition(620, 640) color("8A2BE2")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "8A2BE2")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick23") startPosition(740, 640) color("FF6347")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "FF6347")
                updateBar("cleared", 1) deleteEntity() } } } }
        Entity { id("brick24") startPosition(860, 640) color("FFD700")
            Type { rectangle(110, 30) } Events { OnCollision("ball") {
            Actions { sound("glass") flash("ffffff", 0.1) burst(10, "FFD700")
                updateBar("cleared", 1) deleteEntity() } } } }
    }

    Level {
        Entity { id("titleL2") startPosition(820, 25) color("64748b")
            Type { text("STAGE 2/2", 26) } }
        Entity {
            id("paddle")
            startPosition(440, 30)
            color("00BFFF")
            Type { rectangle(120, 25) }
            Movement { speed(1000) Control { movementControl(2) } }
        }
        Entity {
            id("ball")
            startPosition(490, 400)
            color("FF4500")
            Type { circle(12) }
            Movement { speed(3.4) AutoMove { direction(0, -1) } }
            Events {
                Sequence {
                    Phase(1, 12) { Actions { updateSpeed(4.25) } }
                    Phase(1, 12) { Actions { updateSpeed(5.1) } }
                    Phase(1, 12) { Actions { updateSpeed(5.95) } }
                }
                OnCollision("paddle") {
                    Actions { sound("pop") bounceBack()
                        flash("paddle", "ffffff", 0.1)
                        burst(6, "ffffff") }
                }
                OnCollision("brick1") { Actions { bounceBack() } }
                OnCollision("brick2") { Actions { bounceBack() } }
                OnCollision("brick3") { Actions { bounceBack() } }
                OnCollision("brick4") { Actions { bounceBack() } }
                OnCollision("brick5") { Actions { bounceBack() } }
                OnCollision("brick6") { Actions { bounceBack() } }
                OnCollision("brick7") { Actions { bounceBack() } }
                OnCollision("brick8") { Actions { bounceBack() } }
                OnCollision("brick9") { Actions { bounceBack() } }
                OnCollision("brick10") { Actions { bounceBack() } }
                OnCollision("brick11") { Actions { bounceBack() } }
                OnCollision("brick12") { Actions { bounceBack() } }
                OnCollision("brick13") { Actions { bounceBack() } }
                OnCollision("brick14") { Actions { bounceBack() } }
                OnCollision("brick15") { Actions { bounceBack() } }
                OnCollision("brick16") { Actions { bounceBack() } }
                OnCollision { Actions { bounceBack() } }
            }
        }
        Entity { id("bottomEdge") startPosition(0, 0) color("0a0e27")
            Type { rectangle(1000, 15) }
            Events { OnCollision("ball") {
                Actions {
                    sound("hurt")
                    shake(6, 0.35)
                    burst(16, "FF4500")
                    restart()
                } } } }

        Entity { id("brick1") startPosition(20, 680) color("64748b")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "64748b")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
        Entity { id("brick2") startPosition(140, 680) color("64748b")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "64748b")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
        Entity { id("brick3") startPosition(260, 680) color("64748b")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "64748b")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
        Entity { id("brick4") startPosition(380, 680) color("64748b")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "64748b")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
        Entity { id("brick5") startPosition(500, 680) color("64748b")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "64748b")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
        Entity { id("brick6") startPosition(620, 680) color("64748b")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "64748b")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
        Entity { id("brick7") startPosition(740, 680) color("64748b")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "64748b")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
        Entity { id("brick8") startPosition(860, 680) color("64748b")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "64748b")
                    updateBar("cleared", 1) deleteEntity() } } } } } }

        Entity { id("brick9") startPosition(20, 640) color("334155")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "334155")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
        Entity { id("brick10") startPosition(140, 640) color("334155")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "334155")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
        Entity { id("brick11") startPosition(260, 640) color("334155")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "334155")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
        Entity { id("brick12") startPosition(380, 640) color("334155")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "334155")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
        Entity { id("brick13") startPosition(500, 640) color("334155")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "334155")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
        Entity { id("brick14") startPosition(620, 640) color("334155")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "334155")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
        Entity { id("brick15") startPosition(740, 640) color("334155")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "334155")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
        Entity { id("brick16") startPosition(860, 640) color("334155")
            Type { rectangle(110, 30) } var("hp", 2)
            Events { OnCollision("ball") { Actions { changeVar("hp", -1)
                flash("ffffff", 0.1)
                whenVar("hp", "==", 1) { Actions { sound("clang") setColor("94a3b8")
                    burst(8, "94a3b8") } }
                whenVar("hp", "<=", 0) { Actions { sound("glass") burst(12, "334155")
                    updateBar("cleared", 1) deleteEntity() } } } } } }
    }
}
