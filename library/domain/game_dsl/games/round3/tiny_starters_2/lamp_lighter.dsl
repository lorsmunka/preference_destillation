Window {
    size(8, 8)
    background("0b1020")
    winAt(4, "THE ROOM IS BRIGHT", "win")
    Entity {
        id("lampA") startPosition(100, 560) color("334155") Type { circle(50) } var("on", 0)
        Events {
            OnTapped("lampA") { Actions { whenVar("on", "==", 0) { Actions { changeVar("on", 1) setColor("fde047") updateBar(1) sound("sparkle") } } } }
            OnCollision("bat") { Actions { whenVar("on", "==", 1) { Actions { changeVar("on", -1) setColor("334155") updateBar(-1) sound("blip") } } } }
        }
    }
    Entity {
        id("lampB") startPosition(600, 560) color("334155") Type { circle(50) } var("on", 0)
        Events {
            OnTapped("lampB") { Actions { whenVar("on", "==", 0) { Actions { changeVar("on", 1) setColor("fde047") updateBar(1) sound("sparkle") } } } }
            OnCollision("bat") { Actions { whenVar("on", "==", 1) { Actions { changeVar("on", -1) setColor("334155") updateBar(-1) sound("blip") } } } }
        }
    }
    Entity {
        id("lampC") startPosition(100, 200) color("334155") Type { circle(50) } var("on", 0)
        Events {
            OnTapped("lampC") { Actions { whenVar("on", "==", 0) { Actions { changeVar("on", 1) setColor("fde047") updateBar(1) sound("sparkle") } } } }
            OnCollision("bat") { Actions { whenVar("on", "==", 1) { Actions { changeVar("on", -1) setColor("334155") updateBar(-1) sound("blip") } } } }
        }
    }
    Entity {
        id("lampD") startPosition(600, 200) color("334155") Type { circle(50) } var("on", 0)
        Events {
            OnTapped("lampD") { Actions { whenVar("on", "==", 0) { Actions { changeVar("on", 1) setColor("fde047") updateBar(1) sound("sparkle") } } } }
            OnCollision("bat") { Actions { whenVar("on", "==", 1) { Actions { changeVar("on", -1) setColor("334155") updateBar(-1) sound("blip") } } } }
        }
    }
    Entity {
        id("bat") startPosition(350, 100) color("7c3aed") Type { triangle(40, "up") }
        Movement { speed(3) Path { waypoint(350, 100) waypoint(350, 650) waypoint(150, 400) waypoint(600, 400) } }
    }
    Entity {
        id("hint") startPosition(180, 740) color("94a3b8") Type { text("Light all 4 lamps", 36) }
        Events { OnElapsedTime(40) { Actions { gameOver("THE BAT WON") } } }
    }
}
