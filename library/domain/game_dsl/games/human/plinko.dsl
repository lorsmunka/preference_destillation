Window {
    size(6, 12)
    background("0a0e27")
    gravity(600)

    Bar("score") { initial(0) }

    Entity {
        id("dropPad")
        startPosition(0, 1130)
        color("fbbf24")
        Type { rectangle(600, 50) }
        Events {
            OnTapped("dropPad") {
                Actions {
                    sound("click")
                    spawn("ball", random(120, 480), 1080)
                    flash("ffffff", 0.12)
                }
            }
        }
    }
    Entity { id("dropLabel") startPosition(220, 1138) color("0a0e27")
        Type { text("TAP TO DROP", 28) } }

    Entity {
        id("ball")
        template(true)
        startPosition(0, 0)
        color("fbbf24")
        Type { circle(14) }
        Movement { speed(0.85) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("slot1") { Actions { sound("coin") updateBar("score", 100)
                flash("ffffff", 0.15) burst(10, "94a3b8")
                whenBar("score", ">=", 1500) { Actions { broadcast("won") } }
                deleteEntity() } }
            OnCollision("slot2") { Actions { sound("coin") updateBar("score", 250)
                flash("ffffff", 0.15) burst(12, "4ade80")
                whenBar("score", ">=", 1500) { Actions { broadcast("won") } }
                deleteEntity() } }
            OnCollision("slot3") { Actions { sound("bell") updateBar("score", 500)
                flash("ffffff", 0.18) burst(18, "fbbf24")
                whenBar("score", ">=", 1500) { Actions { broadcast("won") } }
                shake(5, 0.3) deleteEntity() } }
            OnCollision("slot4") { Actions { sound("coin") updateBar("score", 250)
                flash("ffffff", 0.15) burst(12, "4ade80")
                whenBar("score", ">=", 1500) { Actions { broadcast("won") } }
                deleteEntity() } }
            OnCollision("slot5") { Actions { sound("coin") updateBar("score", 100)
                flash("ffffff", 0.15) burst(10, "94a3b8")
                whenBar("score", ">=", 1500) { Actions { broadcast("won") } }
                deleteEntity() } }
            OnCollision("p01") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p02") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p03") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p04") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p05") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p06") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p07") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p08") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p09") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p10") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p11") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p12") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p13") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p14") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p15") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p16") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p17") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p18") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p19") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p20") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p21") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p22") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p23") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p24") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p25") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p26") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("p27") { Actions { sound("blip") bounceBack() burst(4, "fde68a") } }
            OnCollision("d1") { Actions { bounceBack() } }
            OnCollision("d2") { Actions { bounceBack() } }
            OnCollision("d3") { Actions { bounceBack() } }
            OnCollision("d4") { Actions { bounceBack() } }
            OnCollision { Actions { bounceBack() } }
            OnSignal("won") {
                Actions {
                    sound("win")
                    shake(8, 0.6)
                    burst(30, "fbbf24")
                    gameOver("-- TYCOON --")
                }
            }
        }
    }

    Entity { id("p01") startPosition(100, 900) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p02") startPosition(200, 900) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p03") startPosition(300, 900) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p04") startPosition(400, 900) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p05") startPosition(500, 900) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }

    Entity { id("p06") startPosition(150, 800) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p07") startPosition(250, 800) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p08") startPosition(350, 800) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p09") startPosition(450, 800) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }

    Entity { id("p10") startPosition(100, 700) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p11") startPosition(200, 700) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p12") startPosition(300, 700) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p13") startPosition(400, 700) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p14") startPosition(500, 700) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }

    Entity { id("p15") startPosition(150, 600) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p16") startPosition(250, 600) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p17") startPosition(350, 600) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p18") startPosition(450, 600) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }

    Entity { id("p19") startPosition(100, 500) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p20") startPosition(200, 500) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p21") startPosition(300, 500) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p22") startPosition(400, 500) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p23") startPosition(500, 500) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }

    Entity { id("p24") startPosition(150, 400) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p25") startPosition(250, 400) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p26") startPosition(350, 400) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }
    Entity { id("p27") startPosition(450, 400) color("e2e8f0")
        Type { circle(8) } Events { OnCollision("ball") {
        Actions { flash("ffffff", 0.08) } } } }

    Entity { id("d1") startPosition(120, 80) color("64748b")
        Type { rectangle(4, 70) } solid(true) }
    Entity { id("d2") startPosition(240, 80) color("64748b")
        Type { rectangle(4, 70) } solid(true) }
    Entity { id("d3") startPosition(360, 80) color("64748b")
        Type { rectangle(4, 70) } solid(true) }
    Entity { id("d4") startPosition(480, 80) color("64748b")
        Type { rectangle(4, 70) } solid(true) }

    Entity { id("slot1") startPosition(0, 30) color("94a3b8")
        Type { rectangle(120, 50) } }
    Entity { id("slot2") startPosition(120, 30) color("4ade80")
        Type { rectangle(120, 50) } }
    Entity { id("slot3") startPosition(240, 30) color("fbbf24")
        Type { rectangle(120, 50) } }
    Entity { id("slot4") startPosition(360, 30) color("4ade80")
        Type { rectangle(120, 50) } }
    Entity { id("slot5") startPosition(480, 30) color("94a3b8")
        Type { rectangle(120, 50) } }

    Entity { id("lbl1") startPosition(35, 45) color("0a0e27")
        Type { text("100", 22) } }
    Entity { id("lbl2") startPosition(150, 45) color("0a0e27")
        Type { text("250", 22) } }
    Entity { id("lbl3") startPosition(270, 45) color("0a0e27")
        Type { text("500", 22) } }
    Entity { id("lbl4") startPosition(390, 45) color("0a0e27")
        Type { text("250", 22) } }
    Entity { id("lbl5") startPosition(510, 45) color("0a0e27")
        Type { text("100", 22) } }
}
