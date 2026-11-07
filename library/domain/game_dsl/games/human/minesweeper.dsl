Window {
    size(8, 12)
    background("0f172a")
    gravity(0)

    Bar("cleared") {
        initial(0)
        OnValue(20) { Actions { sound("win2") nextLevel() } }
        OnValue(40) { Actions { sound("win2") nextLevel() } }
        OnValue(60) { Actions { sound("win") gameOver("-- ALL BOARDS CLEARED --") } }
    }

    Level {
        Entity { id("titleL1") startPosition(282, 1080) color("e2e8f0")
            Type { text("BOARD 1/3", 48) } }
        Entity { id("c00") startPosition(70, 870) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c00") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c00", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c01") startPosition(215, 870) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c01") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c01", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c02") startPosition(360, 870) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c02") { Actions { setText("c02", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c03") startPosition(505, 870) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c03") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c03", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c04") startPosition(650, 870) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c04") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c04", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c10") startPosition(70, 720) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c10") { Actions { setText("c10", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c11") startPosition(215, 720) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c11") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c11", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c12") startPosition(360, 720) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c12") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c12", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c13") startPosition(505, 720) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c13") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c13", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c14") startPosition(650, 720) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c14") { Actions { setText("c14", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c20") startPosition(70, 570) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c20") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c20", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c21") startPosition(215, 570) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c21") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c21", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c22") startPosition(360, 570) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c22") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c22", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c23") startPosition(505, 570) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c23") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c23", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c24") startPosition(650, 570) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c24") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c24", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c30") startPosition(70, 420) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c30") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c30", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c31") startPosition(215, 420) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c31") { Actions { setText("c31", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c32") startPosition(360, 420) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c32") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c32", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c33") startPosition(505, 420) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c33") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c33", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c34") startPosition(650, 420) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c34") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c34", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c40") startPosition(70, 270) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c40") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c40", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c41") startPosition(215, 270) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c41") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c41", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c42") startPosition(360, 270) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c42") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c42", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c43") startPosition(505, 270) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c43") { Actions { setText("c43", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c44") startPosition(650, 270) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c44") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c44", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
    }

    Level {
        Entity { id("titleL2") startPosition(282, 1080) color("e2e8f0")
            Type { text("BOARD 2/3", 48) } }
        Entity { id("c00") startPosition(70, 870) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c00") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c00", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c01") startPosition(215, 870) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c01") { Actions { setText("c01", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c02") startPosition(360, 870) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c02") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c02", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c03") startPosition(505, 870) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c03") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c03", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c04") startPosition(650, 870) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c04") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c04", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c10") startPosition(70, 720) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c10") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c10", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c11") startPosition(215, 720) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c11") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c11", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c12") startPosition(360, 720) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c12") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c12", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c13") startPosition(505, 720) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c13") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c13", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c14") startPosition(650, 720) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c14") { Actions { setText("c14", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c20") startPosition(70, 570) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c20") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c20", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c21") startPosition(215, 570) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c21") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c21", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c22") startPosition(360, 570) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c22") { Actions { setText("c22", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c23") startPosition(505, 570) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c23") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c23", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c24") startPosition(650, 570) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c24") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c24", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c30") startPosition(70, 420) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c30") { Actions { setText("c30", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c31") startPosition(215, 420) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c31") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c31", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c32") startPosition(360, 420) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c32") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c32", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c33") startPosition(505, 420) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c33") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c33", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c34") startPosition(650, 420) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c34") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c34", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c40") startPosition(70, 270) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c40") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c40", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c41") startPosition(215, 270) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c41") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c41", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c42") startPosition(360, 270) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c42") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c42", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c43") startPosition(505, 270) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c43") { Actions { setText("c43", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c44") startPosition(650, 270) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c44") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c44", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
    }

    Level {
        Entity { id("titleL3") startPosition(282, 1080) color("e2e8f0")
            Type { text("BOARD 3/3", 48) } }
        Entity { id("c00") startPosition(70, 870) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c00") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c00", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c01") startPosition(215, 870) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c01") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c01", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c02") startPosition(360, 870) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c02") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c02", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c03") startPosition(505, 870) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c03") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c03", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c04") startPosition(650, 870) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c04") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c04", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c10") startPosition(70, 720) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c10") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c10", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c11") startPosition(215, 720) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c11") { Actions { setText("c11", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c12") startPosition(360, 720) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c12") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c12", "3") setColor("f97316")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("f97316", 0.2) burst(6, "f97316") } } } } } }
        Entity { id("c13") startPosition(505, 720) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c13") { Actions { setText("c13", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c14") startPosition(650, 720) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c14") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c14", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c20") startPosition(70, 570) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c20") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c20", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c21") startPosition(215, 570) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c21") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c21", "3") setColor("f97316")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("f97316", 0.2) burst(6, "f97316") } } } } } }
        Entity { id("c22") startPosition(360, 570) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c22") { Actions { setText("c22", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c23") startPosition(505, 570) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c23") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c23", "3") setColor("f97316")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("f97316", 0.2) burst(6, "f97316") } } } } } }
        Entity { id("c24") startPosition(650, 570) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c24") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c24", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c30") startPosition(70, 420) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c30") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c30", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c31") startPosition(215, 420) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c31") { Actions { setText("c31", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c32") startPosition(360, 420) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c32") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c32", "3") setColor("f97316")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("f97316", 0.2) burst(6, "f97316") } } } } } }
        Entity { id("c33") startPosition(505, 420) color("475569")
            Type { text("[?]", 64) }
            Events { OnTapped("c33") { Actions { setText("c33", "X")
                sound("lose") setColor("ef4444") shake(7, 0.4) burst(16, "ef4444")
                gameOver("-- BOOM --") } } } }
        Entity { id("c34") startPosition(650, 420) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c34") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c34", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c40") startPosition(70, 270) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c40") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c40", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c41") startPosition(215, 270) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c41") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c41", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c42") startPosition(360, 270) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c42") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c42", "2") setColor("4ade80")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("4ade80", 0.2) burst(6, "4ade80") } } } } } }
        Entity { id("c43") startPosition(505, 270) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c43") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c43", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
        Entity { id("c44") startPosition(650, 270) color("475569")
            Type { text("[?]", 64) } var("revealed", 0)
            Events { OnTapped("c44") { Actions { whenVar("revealed", "==", 0) {
                Actions { setText("c44", "1") setColor("60a5fa")
                    sound("click") updateBar("cleared", 1) changeVar("revealed", 1)
                    flash("60a5fa", 0.2) burst(6, "60a5fa") } } } } } }
    }
}
