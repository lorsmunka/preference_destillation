# Sluice Gardens: canals cut the garden into strips. Tap a sluice lever to drain a canal --
# but the water has to go somewhere, and it usually floods the canal behind you.
# Levers toggle, so a wrong pull can always be undone. Seeds lie on the canal beds.
Window {
    size(8, 10)
    worldSize(30, 10)
    cameraFollow("gardener")
    background("4d7c0f")
    music("bg_6")
    gravity(0)

    lives(3, "-- THE FLAMINGOS CHASED YOU OFF --", "lose")
    Bar("seeds") {
        initial(0)
        OnValue(4) { Actions { broadcast("planted") } }
        OnValue(9) { Actions { broadcast("planted") } }
    }

    Level {
        Entity {
            id("sign")
            startPosition(60, 940)
            color("ecfccb")
            Type { text("Tap the levers to drain the canals. 4 seeds open the greenhouse.", 24) }
            Events { OnSignal("planted") { Actions { setText("sign", "All seeds found - the greenhouse is open!") } } }
        }

        # Seeds come first so the canal water is drawn over them.
        Entity { id("seedA") startPosition(630, 450) color("fde047") Type { circle(16) }
            Events { OnCollision("gardener") { Actions { sound("pickup") updateBar("seeds", 1) burst(10, "fde047") deleteEntity() } } } }
        Entity { id("seedB") startPosition(1250, 200) color("fde047") Type { circle(16) }
            Events { OnCollision("gardener") { Actions { sound("pickup") updateBar("seeds", 1) burst(10, "fde047") deleteEntity() } } } }
        Entity { id("seedC") startPosition(1880, 750) color("fde047") Type { circle(16) }
            Events { OnCollision("gardener") { Actions { sound("pickup") updateBar("seeds", 1) burst(10, "fde047") deleteEntity() } } } }
        Entity { id("seedD") startPosition(2300, 500) color("fde047") Type { circle(16) }
            Events { OnCollision("gardener") { Actions { sound("pickup") updateBar("seeds", 1) burst(10, "fde047") deleteEntity() } } } }

        Entity {
            id("gardener")
            startPosition(100, 480)
            color("fef3c7")
            Type { rectangle(40, 52) }
            Movement { speed(0) Control { movementControl(0) } }
            Events { OnSignal("planted") { Actions { sound("win2") burst(24, "a3e635") } } }
        }
        joystick("gardener", 80, 3)

        Entity {
            id("canalA")
            startPosition(500, 0)
            color("2563eb")
            solid(true)
            Type { rectangle(300, 1000) }
            Events {
                OnSignal("drainA") { Actions { sound("bubble") burst(24, "93c5fd") setEnable(false) } }
                OnSignal("fillA") { Actions { sound("bubble") setEnable(true) } }
            }
        }
        Entity {
            id("canalB")
            startPosition(1100, 0)
            color("2563eb")
            solid(true)
            Type { rectangle(350, 1000) }
            Events {
                OnSignal("drainB") { Actions { sound("bubble") burst(24, "93c5fd") setEnable(false) } }
                OnSignal("fillB") { Actions { sound("bubble") setEnable(true) } }
            }
        }
        Entity {
            id("canalC")
            startPosition(1750, 0)
            color("2563eb")
            solid(true)
            Type { rectangle(300, 1000) }
            Events {
                OnSignal("drainC") { Actions { sound("bubble") burst(24, "93c5fd") setEnable(false) } }
                OnSignal("fillC") { Actions { sound("bubble") setEnable(true) } }
            }
        }

        Entity {
            id("leverA")
            startPosition(300, 800)
            color("a16207")
            Type { rectangle(30, 70) }
            var("on", 0)
            Events {
                OnTapped("leverA") { Actions { sound("click") changeVar("on", 1)
                    whenVar("on", "==", 1) { Actions { setColor("22c55e") broadcast("drainA") } }
                    whenVar("on", "==", 2) { Actions { changeVar("on", -2) setColor("a16207") broadcast("fillA") } } } }
            }
        }
        Entity {
            id("leverB")
            startPosition(950, 820)
            color("a16207")
            Type { rectangle(30, 70) }
            var("on", 0)
            Events {
                OnTapped("leverB") { Actions { sound("click") changeVar("on", 1)
                    whenVar("on", "==", 1) { Actions { setColor("22c55e") broadcast("drainB") broadcast("fillA") } }
                    whenVar("on", "==", 2) { Actions { changeVar("on", -2) setColor("a16207") broadcast("fillB") broadcast("drainA") } } } }
            }
        }
        Entity {
            id("leverC")
            startPosition(1600, 120)
            color("a16207")
            Type { rectangle(30, 70) }
            var("on", 0)
            Events {
                OnTapped("leverC") { Actions { sound("click") changeVar("on", 1)
                    whenVar("on", "==", 1) { Actions { setColor("22c55e") broadcast("drainC") broadcast("fillB") } }
                    whenVar("on", "==", 2) { Actions { changeVar("on", -2) setColor("a16207") broadcast("fillC") broadcast("drainB") } } } }
            }
        }

        Entity {
            id("flamingo")
            color("ffffff")
            Type { text("🦩", 54) }
            facing("left")
            Movement { speed(2) Path { waypoint(2100, 300) waypoint(2500, 300) waypoint(2500, 700) waypoint(2100, 700) } }
            Events { OnCollision("gardener") { Actions { sound("hurt") shake(6, 0.3) flash("gardener", "f472b6", 0.3) updateBar("lives", -1) setPosition("gardener", 2120, 900, 0) } } }
        }

        Entity {
            id("greenGate")
            startPosition(2600, 0)
            color("166534")
            solid(true)
            Type { rectangle(40, 1000) }
            Events { OnSignal("planted") { Actions { sound("clang") burst(20, "86efac") setEnable(false) } } }
        }
        Entity {
            id("greenhouse")
            startPosition(2760, 420)
            color("bbf7d0")
            Type { polygon(6, 70) }
            Events { OnCollision("gardener") { Actions { sound("bell") flash("ffffff", 0.3) nextLevel() } } }
        }
    }

    Level {
        Entity {
            id("sign")
            startPosition(60, 940)
            color("ecfccb")
            Type { text("Five more seeds. One lever sits on a canal bed.", 24) }
            Events { OnSignal("planted") { Actions { setText("sign", "The orchard greenhouse is open!") } } }
        }

        Entity { id("seed1") startPosition(850, 300) color("fde047") Type { circle(16) }
            Events { OnCollision("gardener") { Actions { sound("pickup") updateBar("seeds", 1) burst(10, "fde047") deleteEntity() } } } }
        Entity { id("seed2") startPosition(550, 300) color("fde047") Type { circle(16) }
            Events { OnCollision("gardener") { Actions { sound("pickup") updateBar("seeds", 1) burst(10, "fde047") deleteEntity() } } } }
        Entity { id("seed3") startPosition(1150, 600) color("fde047") Type { circle(16) }
            Events { OnCollision("gardener") { Actions { sound("pickup") updateBar("seeds", 1) burst(10, "fde047") deleteEntity() } } } }
        Entity { id("seed4") startPosition(2000, 250) color("fde047") Type { circle(16) }
            Events { OnCollision("gardener") { Actions { sound("pickup") updateBar("seeds", 1) burst(10, "fde047") deleteEntity() } } } }
        Entity { id("seed5") startPosition(2050, 800) color("fde047") Type { circle(16) }
            Events { OnCollision("gardener") { Actions { sound("pickup") updateBar("seeds", 1) burst(10, "fde047") deleteEntity() } } } }

        # The bed lever only works while canal D is dry.
        Entity {
            id("leverE")
            startPosition(520, 780)
            color("a16207")
            Type { rectangle(30, 70) }
            var("on", 0)
            Events {
                OnElapsedTime { Actions { setEnable(false) } }
                OnSignal("drainD") { Actions { setEnable(true) } }
                OnSignal("fillD") { Actions { setEnable(false) } }
                OnTapped("leverE") { Actions { sound("click") changeVar("on", 1)
                    whenVar("on", "==", 1) { Actions { setColor("22c55e") broadcast("drainE") broadcast("fillD") } }
                    whenVar("on", "==", 2) { Actions { changeVar("on", -2) setColor("a16207") broadcast("fillE") } } } }
            }
        }

        Entity {
            id("gardener")
            startPosition(100, 500)
            color("fef3c7")
            Type { rectangle(40, 52) }
            Movement { speed(0) Control { movementControl(0) } }
            Events { OnSignal("planted") { Actions { sound("win2") burst(24, "a3e635") } } }
        }
        joystick("gardener", 80, 3)

        Entity {
            id("canalD")
            startPosition(400, 0)
            color("1d4ed8")
            solid(true)
            Type { rectangle(300, 1000) }
            Events {
                OnSignal("drainD") { Actions { sound("bubble") burst(24, "93c5fd") setEnable(false) } }
                OnSignal("fillD") { Actions { sound("bubble") setEnable(true) } }
            }
        }
        Entity {
            id("canalE")
            startPosition(1000, 0)
            color("1d4ed8")
            solid(true)
            Type { rectangle(300, 1000) }
            Events {
                OnSignal("drainE") { Actions { sound("bubble") burst(24, "93c5fd") setEnable(false) } }
                OnSignal("fillE") { Actions { sound("bubble") setEnable(true) } }
            }
        }
        Entity {
            id("canalF")
            startPosition(1900, 0)
            color("1d4ed8")
            solid(true)
            Type { rectangle(300, 1000) }
            Events {
                OnSignal("drainF") { Actions { sound("bubble") burst(24, "93c5fd") setEnable(false) } }
                OnSignal("fillF") { Actions { sound("bubble") setEnable(true) } }
            }
        }

        Entity {
            id("leverD")
            startPosition(250, 150)
            color("a16207")
            Type { rectangle(30, 70) }
            var("on", 0)
            Events {
                OnTapped("leverD") { Actions { sound("click") changeVar("on", 1)
                    whenVar("on", "==", 1) { Actions { setColor("22c55e") broadcast("drainD") } }
                    whenVar("on", "==", 2) { Actions { changeVar("on", -2) setColor("a16207") broadcast("fillD") } } } }
            }
        }
        Entity {
            id("leverF")
            startPosition(1700, 500)
            color("a16207")
            Type { rectangle(30, 70) }
            var("on", 0)
            Events {
                OnTapped("leverF") { Actions { sound("click") changeVar("on", 1)
                    whenVar("on", "==", 1) { Actions { setColor("22c55e") broadcast("drainF") broadcast("fillE") } }
                    whenVar("on", "==", 2) { Actions { changeVar("on", -2) setColor("a16207") broadcast("fillF") broadcast("drainE") } } } }
            }
        }

        Entity {
            id("flamingo1")
            color("ffffff")
            Type { text("🦩", 54) }
            facing("left")
            Movement { speed(1.8) Path { waypoint(720, 150) waypoint(720, 850) dwell(0.6) } }
            Events { OnCollision("gardener") { Actions { sound("hurt") shake(6, 0.3) flash("gardener", "f472b6", 0.3) updateBar("lives", -1) setPosition("gardener", 820, 940, 0) } } }
        }
        Entity {
            id("flamingo2")
            color("ffffff")
            Type { text("🦩", 54) }
            facing("left")
            Movement { speed(2.4) Path { waypoint(1350, 200) waypoint(1800, 800) } }
            Events { OnCollision("gardener") { Actions { sound("hurt") shake(6, 0.3) flash("gardener", "f472b6", 0.3) updateBar("lives", -1) setPosition("gardener", 1350, 900, 0) } } }
        }
        Entity {
            id("flamingo3")
            color("ffffff")
            Type { text("🦩", 54) }
            facing("left")
            Movement { speed(2.6) Path { waypoint(2250, 200) waypoint(2550, 500) waypoint(2250, 800) } }
            Events { OnCollision("gardener") { Actions { sound("hurt") shake(6, 0.3) flash("gardener", "f472b6", 0.3) updateBar("lives", -1) setPosition("gardener", 2250, 940, 0) } } }
        }

        Entity {
            id("greenGate")
            startPosition(2600, 0)
            color("166534")
            solid(true)
            Type { rectangle(40, 1000) }
            Events { OnSignal("planted") { Actions { sound("clang") burst(20, "86efac") setEnable(false) } } }
        }
        Entity {
            id("greenhouse")
            startPosition(2760, 420)
            color("bbf7d0")
            Type { polygon(6, 70) }
            Events { OnCollision("gardener") { Actions { sound("win") shake(5, 0.4) burst(30, "86efac") gameOver("-- THE GARDENS ARE IN BLOOM --") } } }
        }
    }
}
