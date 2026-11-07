# Drag the monkey to walk; swipe up, left or right anywhere to jump (one jump per 1.5 s).
# The river resets the temple you are in; a spider, snake or bat bite ends the run.
Window {
    size(8, 10)
    background("052e16")
    music("bg_2")
    gravity(900)

    Bar("bananas") { initial(0) }

    Level {
        Entity {
            id("hero")
            startPosition(60, 100)
            color("ffffff")
            facing("left")
            Type { text("🐒", 44) }
            Movement { speed(5) Control { movementControl(2) } }
            var("air", 0)
            Events {
                OnSwipe("up") {
                    Actions {
                        whenVar("air", "==", 0) {
                            Actions {
                                sound("jump")
                                setVelocity(0, 720)
                                burst(6, "bbf7d0")
                                changeVar("air", 1)
                                delay(1.5) { Actions { changeVar("air", -1) } }
                            }
                        }
                    }
                }
                # a sideways hop carries the monkey 260 px, then the push is cancelled
                OnSwipe("right") {
                    Actions {
                        whenVar("air", "==", 0) {
                            Actions {
                                sound("jump")
                                setVelocity(260, 720)
                                changeVar("air", 1)
                                delay(1) { Actions { applyImpulse(-260, 0) } }
                                delay(1.5) { Actions { changeVar("air", -1) } }
                            }
                        }
                    }
                }
                OnSwipe("left") {
                    Actions {
                        whenVar("air", "==", 0) {
                            Actions {
                                sound("jump")
                                setVelocity(-260, 720)
                                changeVar("air", 1)
                                delay(1) { Actions { applyImpulse(260, 0) } }
                                delay(1.5) { Actions { changeVar("air", -1) } }
                            }
                        }
                    }
                }
                OnCollision("river") { Actions { sound("bubble") shake(6, 0.4) burst(20, "38bdf8") restart() } }
                OnCollision("idol") { Actions { sound("win2") flash("fde047", 0.4) burst(24, "fde047") nextLevel() } }
            }
        }

        Entity { id("river") startPosition(0, 0) color("0369a1") Type { rectangle(800, 50) } }
        Entity { id("s1") startPosition(0, 60) color("57534e") solid(true) Type { rectangle(220, 40) } }
        Entity { id("s2") startPosition(300, 250) color("57534e") solid(true) Type { rectangle(180, 30) } }
        Entity { id("s3") startPosition(560, 420) color("57534e") solid(true) Type { rectangle(240, 30) } }
        Entity { id("s4") startPosition(300, 600) color("57534e") solid(true) Type { rectangle(200, 30) } }
        Entity { id("s5") startPosition(0, 760) color("57534e") solid(true) Type { rectangle(280, 30) } }

        # the gate falls once three bananas are in the bag
        Entity { id("gate") startPosition(170, 790) color("a16207") solid(true) Type { rectangle(24, 210) }
            Events { OnSignal("gate") { Actions { sound("clang") burst(20, "a16207") shake(5, 0.3) deleteEntity() } } } }
        Entity { id("idol") startPosition(40, 790) color("fde047") Type { text("🗿", 48) } }

        Entity { id("banana1") startPosition(380, 290) color("fde047") Type { text("🍌", 30) }
            Events { OnCollision("hero") { Actions { sound("coin") updateBar("bananas", 1) burst(10, "fde047")
                whenBar("bananas", ">=", 3) { Actions { broadcast("gate") } } deleteEntity() } } } }
        Entity { id("banana2") startPosition(700, 460) color("fde047") Type { text("🍌", 30) }
            Events { OnCollision("hero") { Actions { sound("coin") updateBar("bananas", 1) burst(10, "fde047")
                whenBar("bananas", ">=", 3) { Actions { broadcast("gate") } } deleteEntity() } } } }
        Entity { id("banana3") startPosition(400, 640) color("fde047") Type { text("🍌", 30) }
            Events { OnCollision("hero") { Actions { sound("coin") updateBar("bananas", 1) burst(10, "fde047")
                whenBar("bananas", ">=", 3) { Actions { broadcast("gate") } } deleteEntity() } } } }

        Entity { id("spider") startPosition(510, 300) color("1c1917") Type { text("🕷", 40) }
            hostile("hero", "-- BITTEN BY A SPIDER --", "lose")
            Movement { speed(1.6) Path { waypoint(510, 300) waypoint(510, 560) dwell(0.5) } } }

        Entity { id("title1") startPosition(560, 960) color("86efac") Type { text("TEMPLE 1/3", 26) } }
        Entity { id("hint") startPosition(250, 140) color("bbf7d0") Type { text("drag to walk, swipe to jump", 22) } }
    }

    Level {
        Entity {
            id("hero")
            startPosition(700, 100)
            color("ffffff")
            facing("left")
            Type { text("🐒", 44) }
            Movement { speed(5) Control { movementControl(2) } }
            var("air", 0)
            Events {
                OnSwipe("up") {
                    Actions {
                        whenVar("air", "==", 0) {
                            Actions {
                                sound("jump")
                                setVelocity(0, 720)
                                burst(6, "bbf7d0")
                                changeVar("air", 1)
                                delay(1.5) { Actions { changeVar("air", -1) } }
                            }
                        }
                    }
                }
                # a sideways hop carries the monkey 260 px, then the push is cancelled
                OnSwipe("right") {
                    Actions {
                        whenVar("air", "==", 0) {
                            Actions {
                                sound("jump")
                                setVelocity(260, 720)
                                changeVar("air", 1)
                                delay(1) { Actions { applyImpulse(-260, 0) } }
                                delay(1.5) { Actions { changeVar("air", -1) } }
                            }
                        }
                    }
                }
                OnSwipe("left") {
                    Actions {
                        whenVar("air", "==", 0) {
                            Actions {
                                sound("jump")
                                setVelocity(-260, 720)
                                changeVar("air", 1)
                                delay(1) { Actions { applyImpulse(260, 0) } }
                                delay(1.5) { Actions { changeVar("air", -1) } }
                            }
                        }
                    }
                }
                OnCollision("river") { Actions { sound("bubble") shake(6, 0.4) burst(20, "38bdf8") restart() } }
                OnCollision("idol") { Actions { sound("win2") flash("fde047", 0.4) burst(24, "fde047") nextLevel() } }
            }
        }

        Entity { id("river") startPosition(0, 0) color("0369a1") Type { rectangle(800, 50) } }
        Entity { id("s1") startPosition(600, 60) color("57534e") solid(true) Type { rectangle(200, 40) } }
        Entity { id("s2") startPosition(330, 200) color("57534e") solid(true) Type { rectangle(180, 30) } }
        Entity { id("s3") startPosition(40, 340) color("57534e") solid(true) Type { rectangle(200, 30) } }
        Entity { id("s4") startPosition(330, 500) color("57534e") solid(true) Type { rectangle(160, 30) } }
        Entity { id("s5") startPosition(580, 650) color("57534e") solid(true) Type { rectangle(220, 30) } }
        Entity { id("s6") startPosition(250, 820) color("57534e") solid(true) Type { rectangle(270, 30) } }

        Entity { id("gate") startPosition(400, 850) color("a16207") solid(true) Type { rectangle(24, 150) }
            Events { OnSignal("gate") { Actions { sound("clang") burst(20, "a16207") shake(5, 0.3) deleteEntity() } } } }
        Entity { id("idol") startPosition(270, 850) color("fde047") Type { text("🗿", 48) } }

        Entity { id("banana1") startPosition(420, 240) color("fde047") Type { text("🍌", 30) }
            Events { OnCollision("hero") { Actions { sound("coin") updateBar("bananas", 1) burst(10, "fde047")
                whenBar("bananas", ">=", 7) { Actions { broadcast("gate") } } deleteEntity() } } } }
        Entity { id("banana2") startPosition(60, 380) color("fde047") Type { text("🍌", 30) }
            Events { OnCollision("hero") { Actions { sound("coin") updateBar("bananas", 1) burst(10, "fde047")
                whenBar("bananas", ">=", 7) { Actions { broadcast("gate") } } deleteEntity() } } } }
        Entity { id("banana3") startPosition(400, 540) color("fde047") Type { text("🍌", 30) }
            Events { OnCollision("hero") { Actions { sound("coin") updateBar("bananas", 1) burst(10, "fde047")
                whenBar("bananas", ">=", 7) { Actions { broadcast("gate") } } deleteEntity() } } } }
        Entity { id("banana4") startPosition(740, 690) color("fde047") Type { text("🍌", 30) }
            Events { OnCollision("hero") { Actions { sound("coin") updateBar("bananas", 1) burst(10, "fde047")
                whenBar("bananas", ">=", 7) { Actions { broadcast("gate") } } deleteEntity() } } } }

        # a snake patrols the third ledge: jump it
        Entity { id("snake") startPosition(40, 370) color("65a30d") facing("left") Type { text("🐍", 34) }
            hostile("hero", "-- BITTEN BY A SNAKE --", "lose")
            Movement { speed(1.2) Path { waypoint(40, 370) waypoint(196, 370) } } }
        Entity { id("spider") startPosition(530, 380) color("1c1917") Type { text("🕷", 40) }
            hostile("hero", "-- BITTEN BY A SPIDER --", "lose")
            Movement { speed(1.8) Path { waypoint(530, 380) waypoint(530, 620) dwell(0.4) } } }

        Entity { id("title2") startPosition(40, 960) color("86efac") Type { text("TEMPLE 2/3", 26) } }
    }

    Level {
        Entity {
            id("hero")
            startPosition(60, 100)
            color("ffffff")
            facing("left")
            Type { text("🐒", 44) }
            Movement { speed(5) Control { movementControl(2) } }
            var("air", 0)
            Events {
                OnSwipe("up") {
                    Actions {
                        whenVar("air", "==", 0) {
                            Actions {
                                sound("jump")
                                setVelocity(0, 720)
                                burst(6, "bbf7d0")
                                changeVar("air", 1)
                                delay(1.5) { Actions { changeVar("air", -1) } }
                            }
                        }
                    }
                }
                # a sideways hop carries the monkey 260 px, then the push is cancelled
                OnSwipe("right") {
                    Actions {
                        whenVar("air", "==", 0) {
                            Actions {
                                sound("jump")
                                setVelocity(260, 720)
                                changeVar("air", 1)
                                delay(1) { Actions { applyImpulse(-260, 0) } }
                                delay(1.5) { Actions { changeVar("air", -1) } }
                            }
                        }
                    }
                }
                OnSwipe("left") {
                    Actions {
                        whenVar("air", "==", 0) {
                            Actions {
                                sound("jump")
                                setVelocity(-260, 720)
                                changeVar("air", 1)
                                delay(1) { Actions { applyImpulse(260, 0) } }
                                delay(1.5) { Actions { changeVar("air", -1) } }
                            }
                        }
                    }
                }
                OnCollision("river") { Actions { sound("bubble") shake(6, 0.4) burst(20, "38bdf8") restart() } }
                # the giant mushroom throws the monkey up to the high ledge
                OnCollision("shroom") { Actions { sound("sparkle") setVelocity(0, 1100) burst(14, "f472b6") } }
            }
        }

        Entity { id("river") startPosition(0, 0) color("0369a1") Type { rectangle(800, 50) } }
        Entity { id("s1") startPosition(0, 60) color("57534e") solid(true) Type { rectangle(200, 40) } }
        Entity { id("s2") startPosition(260, 170) color("57534e") solid(true) Type { rectangle(260, 30) } }
        Entity { id("shroom") startPosition(460, 200) color("f472b6") Type { rectangle(60, 30) }
            Events { OnCollision("hero") { Actions { flash("ffffff", 0.2) } } } }
        Entity { id("s3") startPosition(580, 730) color("57534e") solid(true) Type { rectangle(220, 30) } }
        Entity { id("s4") startPosition(300, 870) color("57534e") solid(true) Type { rectangle(240, 30) } }

        Entity { id("gate") startPosition(430, 900) color("a16207") solid(true) Type { rectangle(24, 100) }
            Events { OnSignal("gate") { Actions { sound("clang") burst(20, "a16207") shake(5, 0.3) deleteEntity() } } } }
        goal("idol", 330, 900, 60, 60, "-- THE GOLDEN IDOL IS YOURS --")

        Entity { id("banana1") startPosition(120, 260) color("fde047") Type { text("🍌", 30) }
            Events { OnCollision("hero") { Actions { sound("coin") updateBar("bananas", 1) burst(10, "fde047")
                whenBar("bananas", ">=", 12) { Actions { broadcast("gate") } } deleteEntity() } } } }
        Entity { id("banana2") startPosition(300, 210) color("fde047") Type { text("🍌", 30) }
            Events { OnCollision("hero") { Actions { sound("coin") updateBar("bananas", 1) burst(10, "fde047")
                whenBar("bananas", ">=", 12) { Actions { broadcast("gate") } } deleteEntity() } } } }
        Entity { id("banana3") startPosition(470, 640) color("fde047") Type { text("🍌", 30) }
            Events { OnCollision("hero") { Actions { sound("coin") updateBar("bananas", 1) burst(10, "fde047")
                whenBar("bananas", ">=", 12) { Actions { broadcast("gate") } } deleteEntity() } } } }
        Entity { id("banana4") startPosition(720, 770) color("fde047") Type { text("🍌", 30) }
            Events { OnCollision("hero") { Actions { sound("coin") updateBar("bananas", 1) burst(10, "fde047")
                whenBar("bananas", ">=", 12) { Actions { broadcast("gate") } } deleteEntity() } } } }
        Entity { id("banana5") startPosition(490, 910) color("fde047") Type { text("🍌", 30) }
            Events { OnCollision("hero") { Actions { sound("coin") updateBar("bananas", 1) burst(10, "fde047")
                whenBar("bananas", ">=", 12) { Actions { broadcast("gate") } } deleteEntity() } } } }

        # a bat sweeps across the mushroom's launch column
        Entity { id("bat") startPosition(300, 480) color("7c3aed") facing("left") Type { text("🦇", 40) }
            hostile("hero", "-- THE BAT GOT YOU --", "lose")
            Movement { speed(2.4) Path { waypoint(300, 480) waypoint(700, 480) } } }

        Entity { id("title3") startPosition(560, 960) color("86efac") Type { text("TEMPLE 3/3", 26) } }
    }
}
