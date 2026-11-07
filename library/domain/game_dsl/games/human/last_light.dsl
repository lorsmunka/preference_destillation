Window {
    size(8, 12)
    background("05070f")
    music("bg_3")
    gravity(0)

    lives(5, "-- THE LIGHT FADES --", "lose")
    Bar("xp") {
        initial(0)
        OnValue(3) { Actions { broadcast("lvl") } }
        OnValue(8) { Actions { broadcast("lvl") } }
        OnValue(15) { Actions { broadcast("lvl") } }
        OnValue(24) { Actions { broadcast("lvl") } }
        OnValue(35) { Actions { broadcast("lvl") } }
        OnValue(48) { Actions { broadcast("lvl") } }
        OnValue(63) { Actions { broadcast("lvl") } }
        OnValue(80) { Actions { broadcast("lvl") } }
        OnValue(100) { Actions { broadcast("lvl") } }
        OnValue(125) { Actions { broadcast("lvl") } }
        OnValue(155) { Actions { broadcast("lvl") } }
        OnValue(190) { Actions { broadcast("lvl") } }
    }
    Bar("night") { initial(1) }
    Bar("boss") {
        initial(0)
        OnValue(20) { Actions { broadcast("r1") } }
        OnValue(50) { Actions { broadcast("r2") } }
        OnValue(130) { Actions { broadcast("r3") } }
        OnValue(60) { Actions { broadcast("r4") } }
        OnValue(0) { Actions { broadcast("bossdown") } }
    }

    Entity { id("moon") startPosition(640, 1030) color("30e2e8f0")
        Type { circle(56) }
        Events {
            OnSignal("eclipse") { Actions { setColor("90991b1b") } }
            OnSignal("dawn") { Actions {
                whenBar("night", "==", 3) { Actions { setColor("fde68a") } } } }
        } }

    Entity { id("g1") startPosition(170, 390) color("1e293b") Type { rectangle(54, 84) } solid(true) }
    Entity { id("g2") startPosition(576, 390) color("1e293b") Type { rectangle(54, 84) } solid(true) }
    Entity { id("g3") startPosition(170, 780) color("1e293b") Type { rectangle(54, 84) } solid(true) }
    Entity { id("g4") startPosition(576, 780) color("1e293b") Type { rectangle(54, 84) } solid(true) }

    Entity {
        id("hero")
        startPosition(380, 560)
        color("fde68a")
        Type { circle(20) }
        Movement { speed(0) Control { movementControl(0) } }
        var("pulse", 1)
        var("bolt", 0)
        var("seek", 0)
        var("ember", 0)
        var("inv", 0)
        Events {
            OnElapsedTime(1.1) { repeatCount(-1) Actions {
                whenVar("pulse", "==", 1) { Actions { spawnHere("p1") } }
                whenVar("pulse", "==", 2) { Actions { spawnHere("p2") } }
                whenVar("pulse", "==", 3) { Actions { spawnHere("p3") } }
            } }
            OnElapsedTime(0.8) { repeatCount(-1) Actions {
                whenVar("bolt", "==", 1) { Actions {
                    spawnHere("b1") spawnHere("b2") spawnHere("b3") spawnHere("b4") } }
                whenVar("bolt", "==", 2) { Actions {
                    spawnHere("b1") spawnHere("b2") spawnHere("b3") spawnHere("b4")
                    spawnHere("b5") spawnHere("b6") spawnHere("b7") spawnHere("b8") } }
            } }
            OnElapsedTime(0.4) { repeatCount(-1) Actions {
                whenVar("bolt", "==", 3) { Actions {
                    spawnHere("b1") spawnHere("b2") spawnHere("b3") spawnHere("b4")
                    spawnHere("b5") spawnHere("b6") spawnHere("b7") spawnHere("b8") } }
            } }
            OnElapsedTime(1.5) { repeatCount(-1) Actions {
                whenVar("seek", ">=", 1) { Actions { spawnHere("mi") } } } }
            OnElapsedTime(1.3) { repeatCount(-1) Actions {
                whenVar("seek", ">=", 2) { Actions { spawnHere("mi") } } } }
            OnElapsedTime(1.7) { repeatCount(-1) Actions {
                whenVar("seek", ">=", 3) { Actions { spawnHere("mi") } } } }
            OnElapsedTime(0.45) { repeatCount(-1) Actions {
                whenVar("ember", ">=", 1) { Actions { spawnHere("em") } } } }
            OnElapsedTime(0.42) { repeatCount(-1) Actions {
                whenVar("ember", ">=", 2) { Actions { spawnHere("em") } } } }
            OnElapsedTime(0.5) { repeatCount(-1) Actions {
                whenVar("ember", ">=", 3) { Actions { spawnHere("em") } } } }

            OnSignal("ouch") { Actions { whenVar("inv", "==", 0) { Actions {
                changeVar("inv", 1)
                updateBar("lives", -1)
                sound("hurt")
                shake(7, 0.3)
                flash("ef4444", 1)
                delay(1) { Actions { changeVar("inv", -1) } }
            } } } }

            OnCollision("gm") { Actions { updateBar("xp", 1) deleteCollider() } }
            OnCollision("mg") { Actions {
                sound("sparkle") broadcast("magnet") deleteCollider() } }
            OnCollision("st") { Actions {
                sound("thud") shake(10, 0.6) burst(40, "fde68a")
                broadcast("nova") deleteCollider() } }

            OnSignal("lvl") { Actions { setEnable(false) sound("bell") spawn("vl", 0, 0) } }
            OnSignal("lvl") { Actions {
                whenVar("pulse", "<", 3) { Actions { broadcast("oP") } }
                whenVar("bolt", "<", 3) { Actions { broadcast("oB") } }
                whenVar("seek", "<", 3) { Actions { broadcast("oM") } }
                whenVar("ember", "<", 3) { Actions { broadcast("oE") } }
                broadcast("oH")
            } }
            OnSignal("gP") { Actions { changeVar("pulse", 1) broadcast("go") } }
            OnSignal("gB") { Actions { changeVar("bolt", 1) broadcast("go") } }
            OnSignal("gM") { Actions { changeVar("seek", 1) broadcast("go") } }
            OnSignal("gE") { Actions { changeVar("ember", 1) broadcast("go") } }
            OnSignal("gH") { Actions { updateBar("lives", 2) broadcast("go") } }
            OnSignal("go") { Actions {
                setEnable(true)
                sound("sparkle")
                changeVar("inv", 1)
                delay(1.5) { Actions { changeVar("inv", -1) } } } }
        }
    }

    Entity { id("p1") template(true) color("6038bdf8") Type { ring(100, 90) }
        Events { OnElapsedTime(0.2) { Actions { deleteEntity() } } } }
    Entity { id("p2") template(true) color("6038bdf8") Type { ring(140, 128) }
        Events { OnElapsedTime(0.2) { Actions { deleteEntity() } } } }
    Entity { id("p3") template(true) color("707dd3fc") Type { ring(185, 170) }
        Events { OnElapsedTime(0.2) { Actions { deleteEntity() } } } }

    Entity { id("b1") template(true) color("a5f3fc") Type { circle(7) }
        Movement { speed(7) AutoMove { direction(0, 1) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnElapsedTime(1.3) { Actions { deleteEntity() } } } }
    Entity { id("b2") template(true) color("a5f3fc") Type { circle(7) }
        Movement { speed(7) AutoMove { direction(0, -1) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnElapsedTime(1.3) { Actions { deleteEntity() } } } }
    Entity { id("b3") template(true) color("a5f3fc") Type { circle(7) }
        Movement { speed(7) AutoMove { direction(1, 0) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnElapsedTime(1.3) { Actions { deleteEntity() } } } }
    Entity { id("b4") template(true) color("a5f3fc") Type { circle(7) }
        Movement { speed(7) AutoMove { direction(-1, 0) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnElapsedTime(1.3) { Actions { deleteEntity() } } } }
    Entity { id("b5") template(true) color("a5f3fc") Type { circle(7) }
        Movement { speed(7) AutoMove { direction(0.7, 0.7) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnElapsedTime(1.3) { Actions { deleteEntity() } } } }
    Entity { id("b6") template(true) color("a5f3fc") Type { circle(7) }
        Movement { speed(7) AutoMove { direction(-0.7, 0.7) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnElapsedTime(1.3) { Actions { deleteEntity() } } } }
    Entity { id("b7") template(true) color("a5f3fc") Type { circle(7) }
        Movement { speed(7) AutoMove { direction(0.7, -0.7) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnElapsedTime(1.3) { Actions { deleteEntity() } } } }
    Entity { id("b8") template(true) color("a5f3fc") Type { circle(7) }
        Movement { speed(7) AutoMove { direction(-0.7, -0.7) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnElapsedTime(1.3) { Actions { deleteEntity() } } } }

    Entity { id("mi") template(true) color("c084fc") Type { triangle(18, "up") }
        Movement { speed(4.6) chase("gh") }
        Events {
            OnElapsedTime { Actions { randomAction { Actions { } Actions { retarget("bt") } } } }
            OnElapsedTime(3.2) { Actions { deleteEntity() } }
        } }

    Entity { id("em") template(true) color("fb923c") Type { star(5, 13, 6) }
        Events { OnElapsedTime(2.2) { Actions { deleteEntity() } } } }

    Entity { id("gh") template(true) color("65a30d") Type { circle(18) }
        var("hp", 2)
        Movement { speed(1.1) chase("hero") }
        Events {
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnSignal("lvl") { Actions { setEnable(false) } }
            OnSignal("go") { Actions { setEnable(true) } }
            OnElapsedTime { Actions { whenBar("night", ">=", 2) { Actions { changeVar("hp", 1) } }
                whenBar("night", ">=", 3) { Actions { changeVar("hp", 1) } } } }
            OnCollision("p1") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p2") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p3") { Actions { changeVar("hp", -2) flash("fff", 0.08) } }
            OnCollision("b1") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b2") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b3") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b4") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b5") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b6") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b7") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b8") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("mi") { Actions { changeVar("hp", -2) burst(6, "c084fc") deleteCollider() } }
            OnCollision("em") { Actions { changeVar("hp", -1) flash("fb923c", 0.1) } }
            OnSignal("nova") { Actions { changeVar("hp", -6) } }
            OnSignal("dawn") { Actions { burst(8, "fde68a") deleteEntity() } }
            OnElapsedTime(45) { Actions { deleteEntity() } }
            OnElapsedTime(0) { repeatCount(-1) Actions { whenVar("hp", "<=", 0) { Actions {
                sound("pop") burst(10, "65a30d") spawnHere("gm")
                randomAction { Actions { } Actions { } Actions { } Actions { } Actions { }
                    Actions { } Actions { } Actions { } Actions { } Actions { randomAction {
                    Actions { spawnHere("mg") }
                    Actions { spawnHere("st") } Actions { } } } }
                deleteEntity() } } } }
        } }

    Entity { id("bt") template(true) color("a855f7") Type { triangle(30, "down") }
        var("hp", 1)
        Movement { speed(2) chase("hero") }
        Events {
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnSignal("lvl") { Actions { setEnable(false) } }
            OnSignal("go") { Actions { setEnable(true) } }
            OnCollision("p1") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p2") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p3") { Actions { changeVar("hp", -2) flash("fff", 0.08) } }
            OnCollision("b1") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b2") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b3") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b4") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b5") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b6") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b7") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b8") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("mi") { Actions { changeVar("hp", -2) burst(6, "c084fc") deleteCollider() } }
            OnCollision("em") { Actions { changeVar("hp", -1) flash("fb923c", 0.1) } }
            OnSignal("nova") { Actions { changeVar("hp", -6) } }
            OnSignal("dawn") { Actions { burst(8, "fde68a") deleteEntity() } }
            OnElapsedTime(45) { Actions { deleteEntity() } }
            OnElapsedTime(0) { repeatCount(-1) Actions { whenVar("hp", "<=", 0) { Actions {
                sound("pop") burst(10, "a855f7") spawnHere("gm")
                randomAction { Actions { } Actions { } Actions { } Actions { } Actions { }
                    Actions { } Actions { } Actions { } Actions { } Actions { randomAction {
                    Actions { spawnHere("mg") }
                    Actions { spawnHere("st") } Actions { } } } }
                deleteEntity() } } } }
        } }

    Entity { id("br") template(true) color("b91c1c") Type { polygon(5, 32) }
        var("hp", 9)
        Movement { speed(0.6) chase("hero") }
        Events {
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnSignal("lvl") { Actions { setEnable(false) } }
            OnSignal("go") { Actions { setEnable(true) } }
            OnElapsedTime { Actions { whenBar("night", ">=", 2) { Actions { changeVar("hp", 1) } }
                whenBar("night", ">=", 3) { Actions { changeVar("hp", 1) } } } }
            OnCollision("p1") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p2") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p3") { Actions { changeVar("hp", -2) flash("fff", 0.08) } }
            OnCollision("b1") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b2") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b3") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b4") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b5") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b6") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b7") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b8") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("mi") { Actions { changeVar("hp", -2) burst(6, "c084fc") deleteCollider() } }
            OnCollision("em") { Actions { changeVar("hp", -1) flash("fb923c", 0.1) } }
            OnSignal("nova") { Actions { changeVar("hp", -6) } }
            OnSignal("dawn") { Actions { burst(8, "fde68a") deleteEntity() } }
            OnElapsedTime(45) { Actions { deleteEntity() } }
            OnElapsedTime(0) { repeatCount(-1) Actions { whenVar("hp", "<=", 0) { Actions {
                sound("thud") shake(4, 0.2) burst(16, "b91c1c")
                spawnHere("gm") spawnHere("gm") spawnHere("gm")
                deleteEntity() } } } }
        } }

    Entity { id("sp") template(true) color("f97316") Type { circle(26) }
        var("hp", 5)
        Movement { speed(0.85) chase("hero") }
        Events {
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnSignal("lvl") { Actions { setEnable(false) } }
            OnSignal("go") { Actions { setEnable(true) } }
            OnElapsedTime { Actions { whenBar("night", ">=", 2) { Actions { changeVar("hp", 1) } }
                whenBar("night", ">=", 3) { Actions { changeVar("hp", 1) } } } }
            OnCollision("p1") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p2") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p3") { Actions { changeVar("hp", -2) flash("fff", 0.08) } }
            OnCollision("b1") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b2") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b3") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b4") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b5") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b6") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b7") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b8") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("mi") { Actions { changeVar("hp", -2) burst(6, "c084fc") deleteCollider() } }
            OnCollision("em") { Actions { changeVar("hp", -1) flash("fb923c", 0.1) } }
            OnSignal("nova") { Actions { changeVar("hp", -6) } }
            OnSignal("dawn") { Actions { burst(8, "fde68a") deleteEntity() } }
            OnElapsedTime(45) { Actions { deleteEntity() } }
            OnElapsedTime(0) { repeatCount(-1) Actions { whenVar("hp", "<=", 0) { Actions {
                sound("bubble") burst(14, "f97316")
                spawnHere("gm") spawnHere("gm") spawnHere("bt") spawnHere("bt")
                deleteEntity() } } } }
        } }

    Entity { id("wi") template(true) color("818cf8") Type { star(4, 22, 9) }
        var("hp", 4)
        Movement { speed(1.4) Path { waypoint(30, 1060) waypoint(730, 1060) dwell(0.8) } }
        Events {
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnSignal("lvl") { Actions { setEnable(false) } }
            OnSignal("go") { Actions { setEnable(true) } }
            OnElapsedTime { Actions { whenBar("night", ">=", 2) { Actions { changeVar("hp", 1) } }
                whenBar("night", ">=", 3) { Actions { changeVar("hp", 1) } } } }
            OnElapsedTime(2.6) { repeatCount(-1) Actions { spawnHere("sr") } }
            OnCollision("p1") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p2") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p3") { Actions { changeVar("hp", -2) flash("fff", 0.08) } }
            OnCollision("b1") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b2") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b3") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b4") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b5") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b6") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b7") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b8") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("mi") { Actions { changeVar("hp", -2) burst(6, "c084fc") deleteCollider() } }
            OnCollision("em") { Actions { changeVar("hp", -1) flash("fb923c", 0.1) } }
            OnSignal("nova") { Actions { changeVar("hp", -6) } }
            OnSignal("dawn") { Actions { burst(8, "fde68a") deleteEntity() } }
            OnElapsedTime(45) { Actions { deleteEntity() } }
            OnElapsedTime(0) { repeatCount(-1) Actions { whenVar("hp", "<=", 0) { Actions {
                sound("glass") burst(12, "818cf8") spawnHere("gm") spawnHere("gm")
                randomAction { Actions { } Actions { } Actions { } Actions { randomAction {
                    Actions { spawnHere("mg") }
                    Actions { spawnHere("st") } } } }
                deleteEntity() } } } }
        } }

    Entity { id("kl") template(true) color("f1f5f9") Type { circle(12) }
        Movement { speed(1.9) chase("hero") }
        Events {
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnElapsedTime { Actions { updateSpeed(random(1.3, 2.4)) } }
            OnCollision("p1") { Actions { burst(5, "f1f5f9") deleteEntity() } }
            OnCollision("p2") { Actions { burst(5, "f1f5f9") deleteEntity() } }
            OnCollision("p3") { Actions { burst(5, "f1f5f9") deleteEntity() } }
            OnSignal("dawn") { Actions { deleteEntity() } }
            OnElapsedTime(4) { Actions { deleteEntity() } }
        } }
    Entity { id("sr") template(true) color("fb7185") Type { circle(9) }
        Movement { speed(1.6) chase("hero") }
        Events {
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnCollision("p1") { Actions { burst(5, "f1f5f9") deleteEntity() } }
            OnCollision("p2") { Actions { burst(5, "f1f5f9") deleteEntity() } }
            OnCollision("p3") { Actions { burst(5, "f1f5f9") deleteEntity() } }
            OnSignal("dawn") { Actions { deleteEntity() } }
            OnElapsedTime(3.5) { Actions { deleteEntity() } }
        } }
    Entity { id("o1") template(true) color("c4b5fd") Type { circle(10) }
        Movement { speed(2.6) AutoMove { direction(0, 1) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnSignal("dawn") { Actions { deleteEntity() } } } }
    Entity { id("o2") template(true) color("c4b5fd") Type { circle(10) }
        Movement { speed(2.6) AutoMove { direction(0, -1) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnSignal("dawn") { Actions { deleteEntity() } } } }
    Entity { id("o3") template(true) color("c4b5fd") Type { circle(10) }
        Movement { speed(2.6) AutoMove { direction(1, 0) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnSignal("dawn") { Actions { deleteEntity() } } } }
    Entity { id("o4") template(true) color("c4b5fd") Type { circle(10) }
        Movement { speed(2.6) AutoMove { direction(-1, 0) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnSignal("dawn") { Actions { deleteEntity() } } } }
    Entity { id("o5") template(true) color("c4b5fd") Type { circle(10) }
        Movement { speed(2.6) AutoMove { direction(0.7, 0.7) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnSignal("dawn") { Actions { deleteEntity() } } } }
    Entity { id("o6") template(true) color("c4b5fd") Type { circle(10) }
        Movement { speed(2.6) AutoMove { direction(-0.7, 0.7) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnSignal("dawn") { Actions { deleteEntity() } } } }
    Entity { id("o7") template(true) color("c4b5fd") Type { circle(10) }
        Movement { speed(2.6) AutoMove { direction(0.7, -0.7) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnSignal("dawn") { Actions { deleteEntity() } } } }
    Entity { id("o8") template(true) color("c4b5fd") Type { circle(10) }
        Movement { speed(2.6) AutoMove { direction(-0.7, -0.7) } }
        Events { OnCollision { Actions { deleteEntity() } }
            OnCollision("hero") { Actions { broadcast("ouch") deleteEntity() } }
            OnSignal("dawn") { Actions { deleteEntity() } } } }
    Entity { id("wh") template(true) color("30f43f5e") Type { rectangle(800, 50) }
        Events {
            OnSignal("lvl") { Actions { setEnable(false) } }
            OnSignal("go") { Actions { setEnable(true) } }
            OnElapsedTime(0.9) { Actions { spawnHere("bh") deleteEntity() } }
            OnSignal("dawn") { Actions { deleteEntity() } } } }
    Entity { id("bh") template(true) color("e0f43f5e") Type { rectangle(800, 50) }
        Events { OnElapsedTime { Actions { sound("laser") shake(4, 0.2) } }
            OnCollision("hero") { Actions { broadcast("ouch") } }
            OnElapsedTime(0.35) { Actions { deleteEntity() } } } }
    Entity { id("wv") template(true) color("30f43f5e") Type { rectangle(50, 1200) }
        Events {
            OnSignal("lvl") { Actions { setEnable(false) } }
            OnSignal("go") { Actions { setEnable(true) } }
            OnElapsedTime(0.9) { Actions { spawnHere("bv") deleteEntity() } }
            OnSignal("dawn") { Actions { deleteEntity() } } } }
    Entity { id("bv") template(true) color("e0f43f5e") Type { rectangle(50, 1200) }
        Events { OnElapsedTime { Actions { sound("laser") shake(4, 0.2) } }
            OnCollision("hero") { Actions { broadcast("ouch") } }
            OnElapsedTime(0.35) { Actions { deleteEntity() } } } }

    Entity { id("rf") template(true) color("c0a855f7") Type { ring(18, 11) }
        Events {
            OnSignal("lvl") { Actions { setEnable(false) } }
            OnSignal("go") { Actions { setEnable(true) } }
            OnElapsedTime(0.8) { Actions {
                whenBar("night", "==", 1) { Actions { randomAction {
                    Actions { spawnHere("gh") } Actions { spawnHere("gh") }
                    Actions { spawnHere("gh") } Actions { spawnHere("bt") } } } }
                whenBar("night", "==", 2) { Actions { randomAction {
                    Actions { spawnHere("gh") } Actions { spawnHere("gh") } Actions { spawnHere("bt") }
                    Actions { spawnHere("br") } Actions { spawnHere("wi") } } } }
                whenBar("night", "==", 3) { Actions { randomAction {
                    Actions { spawnHere("gh") } Actions { spawnHere("bt") } Actions { spawnHere("bt") }
                    Actions { spawnHere("br") } Actions { spawnHere("sp") } Actions { spawnHere("wi") } } } }
                deleteEntity() } }
            OnSignal("dawn") { Actions { deleteEntity() } }
        } }

    Entity { id("nt") template(true) color("0000") Type { rectangle(0, 0) }
        Events {
            OnSignal("lvl") { Actions { setEnable(false) } }
            OnSignal("go") { Actions { setEnable(true) } }
            Sequence {
                Phase(10, 1.6) { Actions {
                    randomAction {
                        Actions { spawn("rf", random(160, 740), 1100) }
                        Actions { spawn("rf", random(240, 740), 40) }
                        Actions { spawn("rf", 30, random(240, 1040)) }
                        Actions { spawn("rf", 740, random(60, 1040)) } }
                    whenBar("night", ">=", 3) { Actions { randomAction {
                        Actions { spawn("rf", random(160, 740), 1100) }
                        Actions { spawn("rf", 740, random(60, 1040)) } } } } } }
                Phase(16, 1.1) { Actions {
                    randomAction {
                        Actions { spawn("rf", random(160, 740), 1100) }
                        Actions { spawn("rf", random(240, 740), 40) }
                        Actions { spawn("rf", 30, random(240, 1040)) }
                        Actions { spawn("rf", 740, random(60, 1040)) } }
                    whenBar("night", ">=", 2) { Actions { randomAction { Actions { } Actions {
                        spawn("rf", random(160, 740), 1100) } } } }
                    whenBar("night", ">=", 3) { Actions { spawn("rf", 30, random(240, 1040)) } } } }
                Phase(24, 0.75) { Actions {
                    randomAction {
                        Actions { spawn("rf", random(160, 740), 1100) }
                        Actions { spawn("rf", random(240, 740), 40) }
                        Actions { spawn("rf", 30, random(240, 1040)) }
                        Actions { spawn("rf", 740, random(60, 1040)) } }
                    whenBar("night", ">=", 2) { Actions { spawn("rf", 30, random(240, 1040)) } }
                    whenBar("night", ">=", 3) { Actions { spawn("rf", 740, random(60, 1040)) } } } }
                Phase(1, 2) { Actions { broadcast("boss") } }
                Phase(-1, 2.4) { Actions {
                    randomAction {
                        Actions { spawn("rf", random(160, 740), 1100) }
                        Actions { spawn("rf", random(240, 740), 40) }
                        Actions { spawn("rf", 30, random(240, 1040)) }
                        Actions { spawn("rf", 740, random(60, 1040)) } } } }
            }
            OnSignal("dawn") { Actions { deleteEntity() } }
        } }

    Entity { id("wd") template(true) color("e2e8f0") Type { star(7, 62, 32) }
        var("hp", 0)
        var("rage", 0)
        Movement { speed(0.5) chase("hero") }
        Events {
            OnCollision("hero") { Actions { broadcast("ouch") } }
            OnSignal("lvl") { Actions { setEnable(false) } }
            OnSignal("go") { Actions { setEnable(true) } }
            OnElapsedTime { Actions { updateBar("boss", 50) } }
            OnElapsedTime(3) { repeatCount(-1) Actions {
                flash("ef4444", 0.5) sound("laser") spawnHere("kl") spawnHere("kl") spawnHere("kl") } }
            OnElapsedTime(1.7) { repeatCount(-1) Actions {
                whenVar("rage", "==", 1) { Actions { spawnHere("kl") } } } }
            OnSignal("r1") { Actions {
                changeVar("rage", 1) updateSpeed(0.8) setColor("fca5a5")
                sound("buzz") shake(6, 0.4) } }
            OnSignal("bossdown") { Actions { burst(40, "e2e8f0") deleteEntity() } }
            OnCollision("p1") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p2") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p3") { Actions { changeVar("hp", -2) flash("fff", 0.08) } }
            OnCollision("b1") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b2") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b3") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b4") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b5") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b6") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b7") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b8") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("mi") { Actions { changeVar("hp", -2) burst(6, "c084fc") deleteCollider() } }
            OnCollision("em") { Actions { changeVar("hp", -1) flash("fb923c", 0.1) } }
            OnSignal("nova") { Actions { changeVar("hp", -6) } }
            OnElapsedTime(0) { repeatCount(-1) Actions { whenVar("hp", "<", 0) { Actions {
                changeVar("hp", 1) whenBar("boss", ">", 0) { Actions { updateBar("boss", -1) } } } } } }
        } }

    Entity { id("mo") template(true) color("f59e0b") Type { polygon(6, 58) }
        var("hp", 0)
        var("rage", 0)
        Movement { speed(1.1) Path {
            waypoint(80, 900) waypoint(600, 900) waypoint(600, 330) waypoint(80, 330) dwell(0.6) } }
        Events {
            OnCollision("hero") { Actions { broadcast("ouch") } }
            OnSignal("lvl") { Actions { setEnable(false) } }
            OnSignal("go") { Actions { setEnable(true) } }
            OnElapsedTime { Actions { updateBar("boss", 100) } }
            OnElapsedTime(3.5) { repeatCount(-1) Actions {
                flash("ef4444", 0.5) sound("bubble") spawnHere("bt") spawnHere("bt") } }
            OnElapsedTime(2.2) { repeatCount(-1) Actions { spawnHere("sr") } }
            OnElapsedTime(1.1) { repeatCount(-1) Actions {
                whenVar("rage", "==", 1) { Actions { spawnHere("sr") } } } }
            OnSignal("r2") { Actions {
                changeVar("rage", 1) updateSpeed(1.8) setColor("ef4444")
                sound("buzz") shake(6, 0.4) } }
            OnSignal("bossdown") { Actions { burst(40, "f59e0b") deleteEntity() } }
            OnCollision("p1") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p2") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p3") { Actions { changeVar("hp", -2) flash("fff", 0.08) } }
            OnCollision("b1") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b2") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b3") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b4") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b5") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b6") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b7") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b8") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("mi") { Actions { changeVar("hp", -2) burst(6, "c084fc") deleteCollider() } }
            OnCollision("em") { Actions { changeVar("hp", -1) flash("fb923c", 0.1) } }
            OnSignal("nova") { Actions { changeVar("hp", -6) } }
            OnElapsedTime(0) { repeatCount(-1) Actions { whenVar("hp", "<", 0) { Actions {
                changeVar("hp", 1) whenBar("boss", ">", 0) { Actions { updateBar("boss", -1) } } } } } }
        } }

    Entity { id("ec") template(true) color("7c3aed") Type { ring(84, 52) }
        var("hp", 0)
        var("ph", 0)
        Movement { speed(0.22) chase("hero") }
        Events {
            OnCollision("hero") { Actions { broadcast("ouch") } }
            OnSignal("lvl") { Actions { setEnable(false) } }
            OnSignal("go") { Actions { setEnable(true) } }
            OnElapsedTime { Actions { updateBar("boss", 200) broadcast("eclipse") } }
            OnElapsedTime(2.4) { repeatCount(-1) Actions {
                flash("ef4444", 0.5) sound("laser")
                spawnHere("o1") spawnHere("o2") spawnHere("o3") spawnHere("o4")
                spawnHere("o5") spawnHere("o6") spawnHere("o7") spawnHere("o8") } }
            OnElapsedTime(3.1) { repeatCount(-1) Actions {
                whenVar("ph", ">=", 1) { Actions { randomAction {
                    Actions { spawn("wh", 0, random(60, 1000)) }
                    Actions { spawn("wv", random(0, 740), 0) } } } } } }
            OnElapsedTime(1.2) { repeatCount(-1) Actions {
                whenVar("ph", ">=", 2) { Actions {
                    spawnHere("o5") spawnHere("o6") spawnHere("o7") spawnHere("o8") } } } }
            OnSignal("r3") { Actions {
                changeVar("ph", 1) sound("buzz") shake(8, 0.5) } }
            OnSignal("r4") { Actions {
                changeVar("ph", 1) updateSpeed(0.4) setColor("be123c")
                sound("buzz") shake(10, 0.6) } }
            OnSignal("bossdown") { Actions { burst(60, "7c3aed") deleteEntity() } }
            OnCollision("p1") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p2") { Actions { changeVar("hp", -1) flash("fff", 0.08) } }
            OnCollision("p3") { Actions { changeVar("hp", -2) flash("fff", 0.08) } }
            OnCollision("b1") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b2") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b3") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b4") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b5") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b6") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b7") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("b8") { Actions { changeVar("hp", -1) flash("fff", 0.08) deleteCollider() } }
            OnCollision("mi") { Actions { changeVar("hp", -2) burst(6, "c084fc") deleteCollider() } }
            OnCollision("em") { Actions { changeVar("hp", -1) flash("fb923c", 0.1) } }
            OnSignal("nova") { Actions { changeVar("hp", -6) } }
            OnElapsedTime(0) { repeatCount(-1) Actions { whenVar("hp", "<", 0) { Actions {
                changeVar("hp", 1) whenBar("boss", ">", 0) { Actions { updateBar("boss", -1) } } } } } }
        } }

    Entity { id("gm") template(true) color("67e8f9") Type { diamond(14, 20) }
        Movement { speed(0) chase("hero") }
        Events {
            OnSignal("magnet") { Actions { updateSpeed(9) } }
            OnSignal("dawn") { Actions { updateSpeed(9) } }
            OnElapsedTime(14) { Actions { deleteEntity() } }
        } }
    Entity { id("mg") template(true) color("60a5fa") Type { ring(15, 8) }
        Events { OnElapsedTime(10) { Actions { deleteEntity() } } } }
    Entity { id("st") template(true) color("fde047") Type { star(5, 18, 8) }
        Events { OnElapsedTime(10) { Actions { deleteEntity() } } } }

    Entity { id("cu") startPosition(400, 860) color("0000") Type { rectangle(0, 0) }
        Movement { GridMove(150) }
        Events {
            OnSignal("oP") { Actions { gridStep(0, -1) spawnHere("cP") } }
            OnSignal("oB") { Actions { gridStep(0, -1) spawnHere("cB") } }
            OnSignal("oM") { Actions { gridStep(0, -1) spawnHere("cM") } }
            OnSignal("oE") { Actions { gridStep(0, -1) spawnHere("cE") } }
            OnSignal("oH") { Actions { gridStep(0, -1) spawnHere("cH") } }
            OnSignal("go") { Actions { setPosition("cu", 400, 860) } } } }
    Entity { id("vl") template(true) color("e6030509") Type { rectangle(800, 2020) }
        Events { OnElapsedTime { Actions { spawnHere("hd") } }
            OnSignal("go") { Actions { deleteEntity() } } } }
    Entity { id("hd") template(true) color("fde68a") Type { text("LEVEL UP", 60) }
        Events { OnSignal("go") { Actions { deleteEntity() } } } }
    Entity { id("cP") template(true) color("3338bdf8") Type { rectangle(600, 125) }
        Events { OnElapsedTime { Actions { spawnHere("tP") } }
            OnTapped { Actions { broadcast("gP") } }
            OnSignal("go") { Actions { deleteEntity() } } } }
    Entity { id("tP") template(true) color("38bdf8") Type { text("PULSE+
a wider ring", 40) }
        Events { OnSignal("go") { Actions { deleteEntity() } } } }
    Entity { id("cB") template(true) color("33a5f3fc") Type { rectangle(600, 125) }
        Events { OnElapsedTime { Actions { spawnHere("tB") } }
            OnTapped { Actions { broadcast("gB") } }
            OnSignal("go") { Actions { deleteEntity() } } } }
    Entity { id("tB") template(true) color("a5f3fc") Type { text("BOLTS+
shots all around", 40) }
        Events { OnSignal("go") { Actions { deleteEntity() } } } }
    Entity { id("cM") template(true) color("33c084fc") Type { rectangle(600, 125) }
        Events { OnElapsedTime { Actions { spawnHere("tM") } }
            OnTapped { Actions { broadcast("gM") } }
            OnSignal("go") { Actions { deleteEntity() } } } }
    Entity { id("tM") template(true) color("c084fc") Type { text("SEEKER+
homing darts", 40) }
        Events { OnSignal("go") { Actions { deleteEntity() } } } }
    Entity { id("cE") template(true) color("33fb923c") Type { rectangle(600, 125) }
        Events { OnElapsedTime { Actions { spawnHere("tE") } }
            OnTapped { Actions { broadcast("gE") } }
            OnSignal("go") { Actions { deleteEntity() } } } }
    Entity { id("tE") template(true) color("fb923c") Type { text("EMBERS+
a burning trail", 40) }
        Events { OnSignal("go") { Actions { deleteEntity() } } } }
    Entity { id("cH") template(true) color("33f472b6") Type { rectangle(600, 125) }
        Events { OnElapsedTime { Actions { spawnHere("tH") } }
            OnTapped { Actions { broadcast("gH") } }
            OnSignal("go") { Actions { deleteEntity() } } } }
    Entity { id("tH") template(true) color("f472b6") Type { text("HEAL
+2 lives", 40) }
        Events { OnSignal("go") { Actions { deleteEntity() } } } }

    Entity { id("n0") template(true) color("fde68a") Type { text("LAST LIGHT
", 64) }
        Events { OnElapsedTime(2.8) { Actions { deleteEntity() } } } }
    Entity { id("s0") template(true) color("94a3b8") Type { text("
survive three nights", 28) }
        Events { OnElapsedTime(2.8) { Actions { deleteEntity() } } } }
    Entity { id("n1") template(true) color("fde68a") Type { text("NIGHT I", 48) }
        Events { OnElapsedTime(2.5) { Actions { deleteEntity() } } } }
    Entity { id("n2") template(true) color("fde68a") Type { text("NIGHT II", 48) }
        Events { OnElapsedTime(2.5) { Actions { deleteEntity() } } } }
    Entity { id("n3") template(true) color("fde68a") Type { text("NIGHT III", 48) }
        Events { OnElapsedTime(2.5) { Actions { deleteEntity() } } } }
    Entity { id("w1") template(true) color("fca5a5") Type { text("THE WARDEN", 48) }
        Events { OnElapsedTime(2.5) { Actions { deleteEntity() } } } }
    Entity { id("w2") template(true) color("fca5a5") Type { text("THE HIVE MOTHER", 48) }
        Events { OnElapsedTime(2.5) { Actions { deleteEntity() } } } }
    Entity { id("w3") template(true) color("fca5a5") Type { text("THE ECLIPSE", 48) }
        Events { OnElapsedTime(2.5) { Actions { deleteEntity() } } } }
    Entity { id("dn") template(true) color("fde68a") Type { text("DAWN", 48) }
        Events { OnElapsedTime(2.5) { Actions { deleteEntity() } } } }
    Entity { id("dk") template(true) color("0000") Type { rectangle(0, 0) }
        Events {
            OnSignal("lvl") { Actions { setEnable(false) } }
            OnSignal("go") { Actions { setEnable(true) } }
            OnElapsedTime(5) { Actions { broadcast("dusk") deleteEntity() } } } }

    Entity {
        id("dir")
        startPosition(400, 700)
        color("0000")
        Type { rectangle(0, 0) }
        Events {
            OnElapsedTime { Actions {
                spawnHere("n0") spawnHere("s0")
                delay(3) { Actions { broadcast("dusk") } } } }
            OnSignal("dusk") { Actions {
                sound("bell")
                whenBar("night", "==", 1) { Actions { spawnHere("n1") } }
                whenBar("night", "==", 2) { Actions { spawnHere("n2") } }
                whenBar("night", "==", 3) { Actions { spawnHere("n3") } }
                spawn("nt", 0, 0) } }
            OnSignal("boss") { Actions {
                sound("buzz") shake(8, 0.6)
                whenBar("night", "==", 1) { Actions { spawnHere("w1") spawn("wd", 340, 1000) } }
                whenBar("night", "==", 2) { Actions { spawnHere("w2") spawn("mo", 80, 900) } }
                whenBar("night", "==", 3) { Actions { spawnHere("w3") spawn("ec", 316, 940) } } } }
            OnSignal("bossdown") { Actions {
                broadcast("dawn")
                sound("win2") shake(10, 0.8) spawnHere("dn")
                whenBar("night", "==", 3) { Actions { delay(3) { Actions {
                    sound("win") gameOver("-- DAWN BREAKS --") } } } }
                whenBar("night", "<", 3) { Actions { updateBar("night", 1) spawn("dk", 0, 0) } } } }
        }
    }

    joystick("hero", 90, 3.2)
}
