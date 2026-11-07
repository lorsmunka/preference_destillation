Window {
    size(8, 12)
    worldSize(20, 30)
    cameraFollow("hero")
    background("6f9d47")
    music("bg_nature")
    gravity(0)

    Bar("☀") {
        initial(120)
        OnValue(90) { Actions { broadcast("wolf") } }
        OnValue(60) { Actions { broadcast("wolf") } }
        OnValue(30) { Actions { broadcast("wolf") } }
        OnValue(15) { Actions { broadcast("late") } }
        OnValue(0) { Actions { broadcast("dark") } }
    }
    Bar("🏠") {
        initial(0)
        OnValue(5) { Actions { broadcast("win") } }
    }
    Bar("🐴") { initial(0) }

    Entity { id("w1") startPosition(-240, 1640) color("557f37") Type { circle(600) } }
    Entity { id("w2") startPosition(-100, 2300) color("557f37") Type { circle(500) } }
    Entity { id("m1") startPosition(20, 640) color("86b257") Type { circle(340) } }
    Entity { id("p1") startPosition(1180, 60) color("68953f") Type { circle(220) } }
    Entity { id("p2") startPosition(40, 140) color("68953f") Type { circle(170) } }
    Entity { id("q1") startPosition(1220, 2440) color("86b257") Type { circle(340) } }
    Entity { id("q2") startPosition(1320, 2540) color("98c065") Type { circle(240) } }
    Entity { id("q3") startPosition(1420, 2640) color("abcd77") Type { circle(140) } }
    Entity { id("sf") startPosition(1180, 1840) color("8f7a45") Type { rectangle(780, 560) } }
    Entity { id("r1") startPosition(1195, 1860) color("ffffff") Type { text("🌻🌻🌻🌻🌻🌻🌻🌻🌻🌻", 52) } }
    Entity { id("r2") startPosition(1225, 1952) color("ffffff") Type { text("🌻🌻🌻🌻🌻🌻🌻🌻🌻🌻", 52) } }
    Entity { id("r3") startPosition(1195, 2044) color("ffffff") Type { text("🌻🌻🌻🌻🌻🌻🌻🌻🌻🌻", 52) } }
    Entity { id("r4") startPosition(1225, 2136) color("ffffff") Type { text("🌻🌻🌻🌻🌻🌻🌻🌻🌻🌻", 52) } }
    Entity { id("r5") startPosition(1195, 2228) color("ffffff") Type { text("🌻🌻🌻🌻🌻🌻🌻🌻🌻🌻", 52) } }
    Entity { id("r6") startPosition(1225, 2320) color("ffffff") Type { text("🌻🌻🌻🌻🌻🌻🌻🌻🌻🌻", 52) } }
    Entity { id("o1") startPosition(1440, 790) color("3f8fcf") Type { circle(210) } }
    Entity { id("o2") startPosition(1475, 825) color("56a6e0") Type { circle(175) } }
    Entity { id("o3") startPosition(1560, 1060) color("5d9e45") Type { circle(16) } }
    Entity { id("o4") startPosition(1700, 900) color("5d9e45") Type { circle(13) } }
    Entity { id("o5") startPosition(1620, 930) color("5d9e45") Type { circle(10) } }
    Entity { id("rv") startPosition(0, 1450) color("3f8fcf") Type { rectangle(2000, 150) } }
    Entity { id("ra") startPosition(0, 1440) color("d9c58f") Type { rectangle(2000, 12) } }
    Entity { id("rb") startPosition(0, 1598) color("d9c58f") Type { rectangle(2000, 12) } }
    Entity { id("pa") startPosition(945, 520) color("dcc590") Type { rectangle(110, 940) } }
    Entity { id("pb") startPosition(945, 1600) color("dcc590") Type { rectangle(110, 190) } }
    Entity { id("pc") startPosition(300, 1700) color("dcc590") Type { rectangle(900, 90) } }
    Entity { id("ba") startPosition(910, 1425) color("a8743f") Type { rectangle(180, 200) } }
    Entity { id("bb") startPosition(910, 1470) color("8a5a2e") Type { rectangle(180, 5) } }
    Entity { id("bc") startPosition(910, 1520) color("8a5a2e") Type { rectangle(180, 5) } }
    Entity { id("be") startPosition(910, 1570) color("8a5a2e") Type { rectangle(180, 5) } }
    Entity { id("bf") startPosition(910, 1425) color("6b4423") Type { rectangle(10, 200) } }
    Entity { id("bg") startPosition(1080, 1425) color("6b4423") Type { rectangle(10, 200) } }
    Entity { id("bh") startPosition(160, 1425) color("a8743f") Type { rectangle(180, 200) } }
    Entity { id("bk") startPosition(160, 1470) color("8a5a2e") Type { rectangle(180, 5) } }
    Entity { id("bl") startPosition(160, 1520) color("8a5a2e") Type { rectangle(180, 5) } }
    Entity { id("bm") startPosition(160, 1570) color("8a5a2e") Type { rectangle(180, 5) } }
    Entity { id("bi") startPosition(160, 1425) color("6b4423") Type { rectangle(10, 200) } }
    Entity { id("bj") startPosition(330, 1425) color("6b4423") Type { rectangle(10, 200) } }
    Entity { id("pd") startPosition(760, 120) color("d8b57a") Type { rectangle(480, 400) } }
    Entity { id("fa") startPosition(760, 120) color("fff1dc") Type { rectangle(480, 10) } }
    Entity { id("fb") startPosition(760, 120) color("fff1dc") Type { rectangle(10, 400) } }
    Entity { id("fc") startPosition(1230, 120) color("fff1dc") Type { rectangle(10, 400) } }
    Entity { id("fd") startPosition(760, 510) color("fff1dc") Type { rectangle(170, 10) } }
    Entity { id("fe") startPosition(1070, 510) color("fff1dc") Type { rectangle(170, 10) } }
    Entity { id("tr") startPosition(1110, 150) color("8a5a2e") Type { rectangle(90, 34) } }
    Entity { id("ts") startPosition(1116, 156) color("56a6e0") Type { rectangle(78, 22) } }
    Entity { id("bn") startPosition(500, 190) color("9c3a31") Type { rectangle(220, 200) } }
    Entity { id("bo") startPosition(500, 290) color("b84a3d") Type { rectangle(220, 100) } }
    Entity { id("bp") startPosition(500, 286) color("6e2621") Type { rectangle(220, 8) } }
    Entity { id("bq") startPosition(590, 270) color("f1e4c4") Type { rectangle(40, 40) } }
    Entity { id("br") startPosition(598, 278) color("6e2621") Type { rectangle(24, 24) } }
    Entity { id("y1") startPosition(560, 120) color("e3bd57") Type { circle(28) } }
    Entity { id("y2") startPosition(620, 110) color("e3bd57") Type { circle(24) } }
    Entity { id("dc") startPosition(40, 2760) color("2c3b25") Type { circle(110) } }

    Entity { id("x1") startPosition(0, 1450) color("0000") Type { rectangle(170, 150) } solid(true) }
    Entity { id("x2") startPosition(340, 1450) color("0000") Type { rectangle(570, 150) } solid(true) }
    Entity { id("x3") startPosition(1090, 1450) color("0000") Type { rectangle(910, 150) } solid(true) }
    Entity { id("x4") startPosition(1502, 852) color("0000") Type { rectangle(296, 296) } solid(true) }

    Entity { id("hz") startPosition(780, 140) color("0000") Type { rectangle(440, 360) }
        Events { OnCollision("hero") { Actions { broadcast("home") } } } }
    Entity { id("s1") startPosition(970, 300) color("0000") Type { rectangle(0, 0) }
        Movement { speed(6) chase("hero", 0.75) } }
    Entity { id("s2") startPosition(970, 300) color("0000") Type { rectangle(0, 0) }
        Movement { speed(6) chase("s1", 0.6) } }
    Entity { id("s3") startPosition(970, 300) color("0000") Type { rectangle(0, 0) }
        Movement { speed(6) chase("s2", 0.6) } }
    Entity { id("s4") startPosition(970, 300) color("0000") Type { rectangle(0, 0) }
        Movement { speed(6) chase("s3", 0.6) } }
    Entity { id("s5") startPosition(970, 300) color("0000") Type { rectangle(0, 0) }
        Movement { speed(6) chase("s4", 0.6) } }
    Entity { id("g1") startPosition(820, 180) color("0000") Type { rectangle(0, 0) } }
    Entity { id("g2") startPosition(950, 180) color("0000") Type { rectangle(0, 0) } }
    Entity { id("g3") startPosition(1080, 240) color("0000") Type { rectangle(0, 0) } }
    Entity { id("g4") startPosition(860, 360) color("0000") Type { rectangle(0, 0) } }
    Entity { id("g5") startPosition(1020, 380) color("0000") Type { rectangle(0, 0) } }
    Entity { id("h1") startPosition(380, 1000) color("0000") Type { rectangle(0, 0) } }
    Entity { id("h2") startPosition(1860, 1220) color("0000") Type { rectangle(0, 0) } }
    Entity { id("h3") startPosition(450, 2500) color("0000") Type { rectangle(0, 0) } }
    Entity { id("h4") startPosition(1550, 2100) color("0000") Type { rectangle(0, 0) } }
    Entity { id("h5") startPosition(1550, 2760) color("0000") Type { rectangle(0, 0) } }
    Entity { id("dn") startPosition(120, 2820) color("0000") Type { rectangle(0, 0) } }
    Entity { id("ck") color("0000") Type { rectangle(0, 0) }
        Events { OnElapsedTime(1) { repeatCount(-1) Actions { updateBar("☀", -1) } }
            OnSignal("win") { Actions { deleteEntity() } } } }

    Entity { id("t1") template(true) color("ffffff") Type { text("🌳", 84) } }
    Entity { id("t2") template(true) color("ffffff") Type { text("🌲", 84) } }
    Entity { id("fw") template(true) color("ffffff") Type { text("🌼", 30) } }
    Entity { id("fp") template(true) color("ffffff") Type { text("🌸", 28) } }
    Entity { id("rk") template(true) color("ffffff") Type { text("🪨", 40) } }
    Entity { id("rd") template(true) color("ffffff") Type { text("🌾", 40) } }
    Entity { id("ap") template(true) color("ffffff") Type { text("🍎", 24) } }
    Entity { id("dk") template(true) color("001b1f4b") Type { rectangle(2000, 3000) }
        Events { Sequence {
            Phase(1, 25) { Actions { setColor("1aff9a3c") } }
            Phase(1, 12) { Actions { setColor("26ff7a4a") } }
            Phase(1, 12) { Actions { setColor("30e0607a") } }
            Phase(1, 12) { Actions { setColor("3c9a4c8a") } }
            Phase(1, 12) { Actions { setColor("4a553f8f") } }
            Phase(1, 12) { Actions { setColor("58303a80") } }
            Phase(1, 10) { Actions { setColor("66202a6a") } }
            Phase(1, 8) { Actions { setColor("74182356") } }
            Phase(1, 8) { Actions { setColor("82141d4a") } }
            Phase(1, 7) { Actions { setColor("900f183e") } } } } }

    Entity { id("fo") template(true) color("ffffff") Type { text("🐎", 34) } facing("left")
        Movement { speed(5) chase("zz") }
        var("f", 0)
        Events {
            OnElapsedTime(0.4) { repeatCount(-1) Actions {
                whenVar("f", "==", 0) { Actions { burst(4, "fde68a") } } } }
            OnCollision("hero") { Actions { whenVar("f", "==", 0) { Actions {
                changeVar("f", 1)
                whenBar("🐴", "==", 0) { Actions { retarget("s1") } }
                whenBar("🐴", "==", 1) { Actions { retarget("s2") } }
                whenBar("🐴", "==", 2) { Actions { retarget("s3") } }
                whenBar("🐴", "==", 3) { Actions { retarget("s4") } }
                whenBar("🐴", "==", 4) { Actions { retarget("s5") } }
                updateBar("🐴", 1) sound("pickup") burst(16, "ff8fab") spawnHere("ht") } } } }
            OnSignal("home") { Actions { whenVar("f", "==", 1) { Actions {
                changeVar("f", 1)
                whenBar("🏠", "==", 0) { Actions { retarget("g1") } }
                whenBar("🏠", "==", 1) { Actions { retarget("g2") } }
                whenBar("🏠", "==", 2) { Actions { retarget("g3") } }
                whenBar("🏠", "==", 3) { Actions { retarget("g4") } }
                whenBar("🏠", "==", 4) { Actions { retarget("g5") } }
                updateBar("🐴", -1) updateBar("🏠", 1) sound("bell") burst(20, "ff8fab") spawnHere("ht") } } } }
            OnSignal("spook") { Actions { whenVar("f", "==", 1) { Actions {
                changeVar("f", -1) updateBar("🐴", -1)
                randomAction {
                    Actions { retarget("h1") } Actions { retarget("h2") } Actions { retarget("h3") }
                    Actions { retarget("h4") } Actions { retarget("h5") } } } } } }
        } }

    Entity { id("wf") template(true) color("ffffff") Type { text("🐺", 50) }
        Movement { speed(2.6) chase("hero") }
        Events {
            OnCollision("hero") { Actions { retarget("dn") delay(6) { Actions { retarget("hero") } } } }
            OnCollision("hz") { Actions { retarget("dn") delay(6) { Actions { retarget("hero") } } } }
            OnSignal("win") { Actions { deleteEntity() } } } }

    Entity { id("hero") template(true) color("ffffff") Type { text("🐎", 58) } facing("left")
        Movement { speed(0) Control { movementControl(0) } }
        Events {
            OnCollision("wf") { Actions { whenBar("🐴", ">", 0) { Actions {
                broadcast("spook") sound("hurt") shake(8, 0.4) spawnHere("kt") } } } }
            OnSignal("wolf") { Actions { spawnHere("wt") sound("buzz") } }
            OnSignal("late") { Actions { spawnHere("lt") } } } }

    Entity { id("ht") template(true) color("ffffff") Type { text("💛", 30) }
        Movement { speed(0.8) AutoMove { direction(0, 1) } }
        Events { OnElapsedTime(1) { Actions { deleteEntity() } } } }
    Entity { id("wt") template(true) color("ffe4e4") Type { text("🐺 a wolf is out!



", 32) }
        Events { OnElapsedTime(3) { Actions { deleteEntity() } } } }
    Entity { id("lt") template(true) color("fff3c4") Type { text("the sun is setting -- hurry home!



", 30) }
        Events { OnElapsedTime(3) { Actions { deleteEntity() } } } }
    Entity { id("kt") template(true) color("ffe4e4") Type { text("the foals ran off!



", 32) }
        Events { OnElapsedTime(2.5) { Actions { deleteEntity() } } } }
    Entity { id("tp") template(true) color("8c1a2410") Type { rectangle(720, 250) }
        Events { OnElapsedTime(5) { Actions { deleteEntity() } } } }
    Entity { id("ti") template(true) color("fff7e6") Type { text("SUNDOWN HERD
", 60) }
        Events { OnElapsedTime(5) { Actions { deleteEntity() } } } }
    Entity { id("su") template(true) color("ffe2b8") Type { text("
five foals wandered off", 30) }
        Events { OnElapsedTime(5) { Actions { deleteEntity() } } } }
    Entity { id("hi") template(true) color("ffffff") Type { text("


find them ✨ and lead them home before dark", 24) }
        Events { OnElapsedTime(5) { Actions { deleteEntity() } } } }
    Entity { id("wn") template(true) color("fff7e6") Type { text("ALL FIVE ARE HOME 💛", 52) } }

    Entity { id("bd") color("0000") Type { rectangle(0, 0) }
        Events { OnElapsedTime { Actions {
            loopCount(24) spawn("t1", random(0, 860), random(1700, 2900))
            loopCount(20) spawn("t2", random(0, 860), random(1700, 2900))
            loopCount(5) spawn("t1", random(1880, 1960), random(0, 700))
            loopCount(5) spawn("t2", random(0, 120), random(0, 600))
            loopCount(14) spawn("fw", random(60, 640), random(680, 1300))
            loopCount(12) spawn("fp", random(60, 640), random(680, 1300))
            loopCount(6) spawn("rk", random(1300, 1800), random(2520, 2900))
            loopCount(9) spawn("t1", random(1330, 1900), random(80, 640))
            loopCount(10) spawn("ap", random(1330, 1920), random(80, 660))
            loopCount(5) spawn("rd", random(1460, 1820), random(760, 800))
            loopCount(8) spawn("rd", random(0, 1960), random(1380, 1415)) } } } }

    Entity { id("dir") startPosition(1000, 760) color("0000") Type { rectangle(0, 0) }
        Events {
            OnElapsedTime { Actions {
                spawn("dk", 0, 0)
                spawn("fo", random(120, 600), random(700, 1300))
                spawn("fo", random(1800, 1920), random(1100, 1350))
                spawn("fo", random(100, 800), random(2200, 2850))
                spawn("fo", random(1250, 1850), random(1870, 2350))
                spawn("fo", random(1380, 1720), random(2620, 2880))
                spawn("hero", 970, 300)
                spawnHere("tp") spawnHere("ti") spawnHere("su") spawnHere("hi") } }
            OnSignal("wolf") { Actions { spawn("wf", 120, 2840) } }
            OnSignal("win") { Actions { spawnHere("tp") spawnHere("wn") sound("win") burst(80, "ff8fab")
                delay(3) { Actions {
                    whenBar("☀", ">=", 75) { Actions { gameOver("-- ALL HOME BEFORE SUNSET ⭐⭐⭐ --") } }
                    whenBar("☀", "<", 75) { Actions { whenBar("☀", ">=", 45) { Actions { gameOver("-- ALL HOME AT SUNSET ⭐⭐ --") } } } }
                    whenBar("☀", "<", 45) { Actions { gameOver("-- ALL HOME BY DARK ⭐ --") } } } } } }
            OnSignal("dark") { Actions { sound("lose") gameOver("-- NIGHT FELL BEFORE ALL WERE HOME --") } } } }

    joystick("hero", 100, 4)
}
