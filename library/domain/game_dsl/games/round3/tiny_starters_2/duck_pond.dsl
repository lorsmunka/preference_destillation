Window {
    size(8, 10)
    background("65a30d")
    music("bg_nature")
    winAt(3, "ALL DUCKLINGS SWIM", "win")
    Entity { id("pond") startPosition(280, 800) color("38bdf8") Type { rectangle(240, 160) } }
    Entity {
        id("mother") startPosition(380, 100) color("fef08a") Type { circle(30) }
        Movement { speed(5) Control { movementControl(0) } }
    }
    Entity {
        id("duckA") startPosition(300, 60) color("fde047") Type { circle(16) }
        Movement { speed(4) chase("mother", 0.5) }
        Events { OnCollision("pond") { Actions { sound("bubble") updateBar(1) setEnable(false) } } }
    }
    Entity {
        id("duckB") startPosition(460, 60) color("facc15") Type { circle(16) }
        Movement { speed(3) chase("mother", 0.8) }
        Events { OnCollision("pond") { Actions { sound("bubble") updateBar(1) setEnable(false) } } }
    }
    Entity {
        id("duckC") startPosition(380, 20) color("eab308") Type { circle(16) }
        Movement { speed(3.5) chase("mother", 1.1) }
        Events { OnCollision("pond") { Actions { sound("bubble") updateBar(1) setEnable(false) } } }
    }
    Entity {
        id("fox") startPosition(100, 450) color("ea580c") Type { triangle(50, "right") }
        Movement { speed(3) Path { waypoint(100, 450) waypoint(700, 450) } }
        Events {
            OnCollision("duckA") { Actions { sound("lose") gameOver("THE FOX GOT A DUCKLING") } }
            OnCollision("duckB") { Actions { sound("lose") gameOver("THE FOX GOT A DUCKLING") } }
            OnCollision("duckC") { Actions { sound("lose") gameOver("THE FOX GOT A DUCKLING") } }
        }
    }
}
