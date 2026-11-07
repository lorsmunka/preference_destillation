Window {
    size(8, 12)
    background("2a1a12")
    music("bg_6")
    gravity(0)

    dpad("theseus")

    Entity { id("status") startPosition(630, 230) color("fcd34d") Type { text("Find the sword", 22) }
        Events {
            OnSignal("armed") { Actions { setText("status", "Hunt it down") } }
            OnSignal("slain") { Actions { setText("status", "Gate open!") } }
        } }

    Entity {
        id("theseus")
        startPosition(12, 412)
        color("38bdf8")
        Type { circle(38) }
        var("armed", 0)
        Movement { GridMove(100, 0.3) { direction(0, 0) } }
        Events {
            OnSignal("armed") { Actions { changeVar("armed", 1) setColor("e0f2fe") flash("fde68a", 0.4) burst(14, "fde68a") } }
            OnCollision("minotaur") {
                Actions {
                    whenVar("armed", "==", 1) { Actions {
                        sound("thud") shake(10, 0.6) burst(30, "b91c1c")
                        deleteEntity("minotaur") broadcast("slain") } }
                    whenVar("armed", "==", 0) { Actions {
                        sound("death") shake(8, 0.5) gameOver("-- THE MINOTAUR FOUND YOU --") } }
                }
            }
            OnCollision("exit") { Actions { sound("win") burst(24, "4ade80") gameOver("-- OUT OF THE LABYRINTH --") } }
        }
    }

    # 8 x 8 labyrinth of 100 px cells, rows from y 400 to 1200.
    Entity { id("wall1") startPosition(100, 400) color("57534e") solid(true) Type { rectangle(300, 100) } }
    Entity { id("wall2") startPosition(500, 400) color("57534e") solid(true) Type { rectangle(200, 100) } }
    Entity { id("wall3") startPosition(100, 600) color("57534e") solid(true) Type { rectangle(500, 100) } }
    Entity { id("wall4") startPosition(700, 600) color("57534e") solid(true) Type { rectangle(100, 200) } }
    Entity { id("wall5") startPosition(100, 700) color("57534e") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wall6") startPosition(100, 800) color("57534e") solid(true) Type { rectangle(300, 100) } }
    Entity { id("wall7") startPosition(500, 800) color("57534e") solid(true) Type { rectangle(100, 300) } }
    Entity { id("wall8") startPosition(0, 1000) color("57534e") solid(true) Type { rectangle(200, 100) } }
    Entity { id("wall9") startPosition(300, 1000) color("57534e") solid(true) Type { rectangle(100, 200) } }
    Entity { id("wall10") startPosition(600, 1000) color("57534e") solid(true) Type { rectangle(100, 100) } }

    Entity { id("gate") startPosition(400, 500) color("92400e") Type { rectangle(100, 100) } door("slain", "clang") }
    Entity { id("exit") startPosition(410, 410) color("4ade80") Type { rectangle(80, 80) }
        Events { OnElapsedTime(1.2) { repeatCount(-1) Actions { burst(4, "86efac") } } } }

    Entity { id("sword") startPosition(20, 1120) color("e5e7eb") Type { cross(60, 60, 12) }
        Events { OnCollision("theseus") { Actions { sound("clang") broadcast("armed") deleteEntity() } } } }

    # Torches frighten the beast: it slows to a crawl for six seconds.
    Entity { id("torch1") startPosition(230, 730) color("f97316") Type { triangle(40, "up") }
        Events { OnCollision("theseus") { Actions { sound("sparkle") burst(12, "f97316") broadcast("torch") deleteEntity() } } } }
    Entity { id("torch2") startPosition(630, 930) color("f97316") Type { triangle(40, "up") }
        Events { OnCollision("theseus") { Actions { sound("sparkle") burst(12, "f97316") broadcast("torch") deleteEntity() } } } }

    Entity {
        id("minotaur")
        startPosition(715, 1115)
        color("b91c1c")
        Type { rectangle(70, 70) }
        Movement { speed(1.1) chase("theseus") }
        Events {
            OnSignal("torch") { Actions { updateSpeed(0.4) setColor("7f1d1d")
                delay(6) { Actions { updateSpeed(1.1) setColor("b91c1c") } } } }
            OnElapsedTime(2.5) { repeatCount(-1) Actions { burst(5, "7f1d1d") } }
        }
    }

    # Blades slide out of the walls across three corridors.
    Entity { id("blade1") startPosition(20, 720) color("cbd5e1") Type { rectangle(60, 60) }
        hostile("theseus", "-- CUT BY A BLADE --", "hurt")
        Movement { speed(1) Path { waypoint(20, 720) waypoint(120, 720) dwell(0.8) } } }
    Entity { id("blade2") startPosition(320, 820) color("cbd5e1") Type { rectangle(60, 60) }
        hostile("theseus", "-- CUT BY A BLADE --", "hurt")
        Movement { speed(1.4) Path { waypoint(320, 820) waypoint(320, 1020) dwell(1) } } }
    Entity { id("blade3") startPosition(520, 420) color("cbd5e1") Type { rectangle(60, 60) }
        hostile("theseus", "-- CUT BY A BLADE --", "hurt")
        Movement { speed(1.4) Path { waypoint(520, 420) waypoint(520, 620) dwell(1) } } }
}
