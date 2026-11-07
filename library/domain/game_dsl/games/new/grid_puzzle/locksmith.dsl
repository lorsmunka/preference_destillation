# Locksmith. A key blank drops toward the lock on its own; shift it, turn it and push it so it
# slides all the way into the keyhole and touches the gold pin at the bottom. A key that lands
# anywhere else bends: it costs a pick and a fresh blank drops from the top.
Window {
    size(7, 12)
    background("1c1917")
    music("bg_4")
    gravity(0)

    Bar("picks") {
        initial(5)
        OnValue(0) { Actions { sound("lose") gameOver("-- OUT OF PICKS --") } }
    }

    Level {
        Entity { id("title1") startPosition(20, 1140) color("d6d3d1") Type { text("LOCK 1/3: an L-shaped keyhole", 32) } }
        Entity { id("railW") startPosition(0, 200) color("292524") Type { rectangle(50, 1000) } solid(true) }
        Entity { id("railE") startPosition(650, 200) color("292524") Type { rectangle(50, 1000) } solid(true) }
        Entity { id("lock1a") startPosition(50, 200) color("57534e") Type { rectangle(600, 60) } solid(true) }
        Entity { id("lock1b") startPosition(50, 260) color("57534e") Type { rectangle(240, 180) } solid(true) }
        Entity { id("lock1c") startPosition(410, 260) color("57534e") Type { rectangle(240, 180) } solid(true) }

        Entity {
            id("pin")
            startPosition(368, 278)
            color("facc15")
            Type { circle(12) }
            Events {
                OnCollision("key") {
                    Actions {
                        sound("win2")
                        burst(20, "fde68a")
                        flash("key", "fde68a", 0.3)
                        nextLevel()
                    }
                }
            }
        }

        Entity {
            id("key")
            startPosition(290, 1040)
            color("fbbf24")
            Type { tetromino(60, "L") }
            Movement { GridMove(60, 0.6) { direction(0, -1) } }
            Events {
                OnSignal("slideL") { Actions { gridStep(-1, 0) } }
                OnSignal("slideR") { Actions { gridStep(1, 0) } }
                OnSignal("twist") { Actions { rotate(1) } }
                OnSignal("push") { Actions { gridStep(0, -1) } }
                OnGridBlocked {
                    Actions {
                        sound("thud")
                        shake(5, 0.3)
                        burst(10, "fbbf24")
                        updateBar("picks", -1)
                        setPosition("key", 290, 1040, 0)
                    }
                }
            }
        }

        Entity { id("btnLeft") startPosition(50, 30) color("44403c") Type { rectangle(140, 140) }
            Events { OnTapped("btnLeft") { Actions { sound("click") flash("ffffff", 0.08) broadcast("slideL") } } } }
        Entity { id("btnTurn") startPosition(200, 30) color("57534e") Type { rectangle(140, 140) }
            Events { OnTapped("btnTurn") { Actions { sound("blip") flash("ffffff", 0.08) broadcast("twist") } } } }
        Entity { id("btnPush") startPosition(350, 30) color("57534e") Type { rectangle(140, 140) }
            Events { OnTapped("btnPush") { Actions { sound("click") flash("ffffff", 0.08) broadcast("push") } } } }
        Entity { id("btnRight") startPosition(500, 30) color("44403c") Type { rectangle(140, 140) }
            Events { OnTapped("btnRight") { Actions { sound("click") flash("ffffff", 0.08) broadcast("slideR") } } } }
        Entity { id("tagLeft") startPosition(95, 80) color("e7e5e4") Type { text("<", 44) } }
        Entity { id("tagTurn") startPosition(225, 85) color("e7e5e4") Type { text("TURN", 32) } }
        Entity { id("tagPush") startPosition(375, 85) color("e7e5e4") Type { text("PUSH", 32) } }
        Entity { id("tagRight") startPosition(555, 80) color("e7e5e4") Type { text(">", 44) } }
    }

    Level {
        Entity { id("title2") startPosition(20, 1140) color("d6d3d1") Type { text("LOCK 2/3: mind the tumbler", 32) } }
        Entity { id("railW") startPosition(0, 200) color("292524") Type { rectangle(50, 1000) } solid(true) }
        Entity { id("railE") startPosition(650, 200) color("292524") Type { rectangle(50, 1000) } solid(true) }
        Entity { id("lock2a") startPosition(50, 200) color("57534e") Type { rectangle(600, 60) } solid(true) }
        Entity { id("lock2b") startPosition(50, 260) color("57534e") Type { rectangle(360, 180) } solid(true) }
        Entity { id("lock2c") startPosition(470, 260) color("57534e") Type { rectangle(180, 60) } solid(true) }
        Entity { id("lock2d") startPosition(530, 320) color("57534e") Type { rectangle(120, 120) } solid(true) }

        # A tumbler bar sweeps along the face of the lock; a key that drops onto it is wasted.
        Entity {
            id("tumbler")
            startPosition(50, 440)
            color("94a3b8")
            Type { rectangle(120, 60) }
            solid(true)
            Movement { speed(1) Path { waypoint(50, 440) waypoint(530, 440) dwell(0.4) } }
        }

        Entity {
            id("pin")
            startPosition(428, 278)
            color("facc15")
            Type { circle(12) }
            Events {
                OnCollision("key") {
                    Actions {
                        sound("win2")
                        burst(20, "fde68a")
                        flash("key", "fde68a", 0.3)
                        nextLevel()
                    }
                }
            }
        }

        Entity {
            id("key")
            startPosition(230, 1040)
            color("a3e635")
            Type { tetromino(60, "T") }
            Movement { GridMove(60, 0.6) { direction(0, -1) } }
            Events {
                OnSignal("slideL") { Actions { gridStep(-1, 0) } }
                OnSignal("slideR") { Actions { gridStep(1, 0) } }
                OnSignal("twist") { Actions { rotate(1) } }
                OnSignal("push") { Actions { gridStep(0, -1) } }
                OnGridBlocked {
                    Actions {
                        sound("thud")
                        shake(5, 0.3)
                        burst(10, "a3e635")
                        updateBar("picks", -1)
                        setPosition("key", 230, 1040, 0)
                    }
                }
            }
        }

        Entity { id("btnLeft") startPosition(50, 30) color("44403c") Type { rectangle(140, 140) }
            Events { OnTapped("btnLeft") { Actions { sound("click") flash("ffffff", 0.08) broadcast("slideL") } } } }
        Entity { id("btnTurn") startPosition(200, 30) color("57534e") Type { rectangle(140, 140) }
            Events { OnTapped("btnTurn") { Actions { sound("blip") flash("ffffff", 0.08) broadcast("twist") } } } }
        Entity { id("btnPush") startPosition(350, 30) color("57534e") Type { rectangle(140, 140) }
            Events { OnTapped("btnPush") { Actions { sound("click") flash("ffffff", 0.08) broadcast("push") } } } }
        Entity { id("btnRight") startPosition(500, 30) color("44403c") Type { rectangle(140, 140) }
            Events { OnTapped("btnRight") { Actions { sound("click") flash("ffffff", 0.08) broadcast("slideR") } } } }
        Entity { id("tagLeft") startPosition(95, 80) color("e7e5e4") Type { text("<", 44) } }
        Entity { id("tagTurn") startPosition(225, 85) color("e7e5e4") Type { text("TURN", 32) } }
        Entity { id("tagPush") startPosition(375, 85) color("e7e5e4") Type { text("PUSH", 32) } }
        Entity { id("tagRight") startPosition(555, 80) color("e7e5e4") Type { text(">", 44) } }
    }

    Level {
        Entity { id("title3") startPosition(20, 1140) color("d6d3d1") Type { text("LOCK 3/3: two keyholes, one pin", 32) } }
        Entity { id("railW") startPosition(0, 200) color("292524") Type { rectangle(50, 1000) } solid(true) }
        Entity { id("railE") startPosition(650, 200) color("292524") Type { rectangle(50, 1000) } solid(true) }
        Entity { id("lock3a") startPosition(50, 200) color("57534e") Type { rectangle(600, 60) } solid(true) }
        Entity { id("lock3b") startPosition(50, 260) color("57534e") Type { rectangle(120, 60) } solid(true) }
        Entity { id("lock3c") startPosition(230, 260) color("57534e") Type { rectangle(420, 60) } solid(true) }
        Entity { id("lock3d") startPosition(50, 320) color("57534e") Type { rectangle(60, 120) } solid(true) }
        Entity { id("lock3e") startPosition(230, 320) color("57534e") Type { rectangle(180, 120) } solid(true) }
        Entity { id("lock3f") startPosition(530, 320) color("57534e") Type { rectangle(120, 60) } solid(true) }
        Entity { id("lock3g") startPosition(590, 380) color("57534e") Type { rectangle(60, 60) } solid(true) }

        Entity {
            id("pin")
            startPosition(188, 278)
            color("facc15")
            Type { circle(12) }
            Events {
                OnCollision("key") {
                    Actions {
                        sound("win")
                        shake(6, 0.5)
                        burst(30, "fde68a")
                        gameOver("-- EVERY LOCK OPENS FOR YOU --")
                    }
                }
            }
        }

        Entity {
            id("key")
            startPosition(290, 1040)
            color("f472b6")
            Type { tetromino(60, "S") }
            Movement { GridMove(60, 0.45) { direction(0, -1) } }
            Events {
                OnSignal("slideL") { Actions { gridStep(-1, 0) } }
                OnSignal("slideR") { Actions { gridStep(1, 0) } }
                OnSignal("twist") { Actions { rotate(1) } }
                OnSignal("push") { Actions { gridStep(0, -1) } }
                OnGridBlocked {
                    Actions {
                        sound("thud")
                        shake(5, 0.3)
                        burst(10, "f472b6")
                        updateBar("picks", -1)
                        setPosition("key", 290, 1040, 0)
                    }
                }
            }
        }

        Entity { id("btnLeft") startPosition(50, 30) color("44403c") Type { rectangle(140, 140) }
            Events { OnTapped("btnLeft") { Actions { sound("click") flash("ffffff", 0.08) broadcast("slideL") } } } }
        Entity { id("btnTurn") startPosition(200, 30) color("57534e") Type { rectangle(140, 140) }
            Events { OnTapped("btnTurn") { Actions { sound("blip") flash("ffffff", 0.08) broadcast("twist") } } } }
        Entity { id("btnPush") startPosition(350, 30) color("57534e") Type { rectangle(140, 140) }
            Events { OnTapped("btnPush") { Actions { sound("click") flash("ffffff", 0.08) broadcast("push") } } } }
        Entity { id("btnRight") startPosition(500, 30) color("44403c") Type { rectangle(140, 140) }
            Events { OnTapped("btnRight") { Actions { sound("click") flash("ffffff", 0.08) broadcast("slideR") } } } }
        Entity { id("tagLeft") startPosition(95, 80) color("e7e5e4") Type { text("<", 44) } }
        Entity { id("tagTurn") startPosition(225, 85) color("e7e5e4") Type { text("TURN", 32) } }
        Entity { id("tagPush") startPosition(375, 85) color("e7e5e4") Type { text("PUSH", 32) } }
        Entity { id("tagRight") startPosition(555, 80) color("e7e5e4") Type { text(">", 44) } }
    }
}
