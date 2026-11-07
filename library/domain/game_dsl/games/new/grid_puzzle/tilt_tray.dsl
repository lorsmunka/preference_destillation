# Tilt tray. Every swipe tips the tray: all the marbles roll that way at once until a wall, a
# stud or another marble stops them. A marble that comes to rest on a cup of its own colour drops
# in. Sink every marble on three trays within the tilt budget.
Window {
    size(8, 11)
    background("0f172a")
    music("bg_1")
    gravity(0)

    Bar("pocketed") {
        initial(0)
        OnValue(2) { Actions { sound("win2") nextLevel() } }
        OnValue(5) { Actions { sound("win2") nextLevel() } }
        OnValue(9) { Actions { sound("win") gameOver("-- EVERY MARBLE SUNK --") } }
    }
    Bar("tilts") {
        initial(40)
        OnValue(0) { Actions { sound("lose") gameOver("-- OUT OF TILTS --") } }
    }

    Level {
        Entity { id("title1") startPosition(100, 950) color("cbd5e1") Type { text("TRAY 1/3", 44) } }
        Entity { id("retry1") startPosition(540, 960) color("fca5a5") Type { text("RETRY", 36) }
            Events { OnTapped("retry1") { Actions { sound("click") restart() } } } }
        Entity { id("rimW") startPosition(80, 230) color("1e293b") Type { rectangle(20, 640) } solid(true) }
        Entity { id("rimE") startPosition(700, 230) color("1e293b") Type { rectangle(20, 640) } solid(true) }
        Entity { id("rimN") startPosition(80, 850) color("1e293b") Type { rectangle(640, 20) } solid(true) }
        Entity { id("rimS") startPosition(80, 230) color("1e293b") Type { rectangle(640, 20) } solid(true) }
        Entity { id("stud1a") startPosition(100, 750) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud1b") startPosition(200, 450) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud1c") startPosition(300, 350) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud1d") startPosition(300, 750) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud1e") startPosition(600, 550) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud1f") startPosition(600, 650) color("475569") Type { rectangle(100, 100) } solid(true) }

        Entity { id("rubyCup") template(true) startPosition(0, 0) color("7f1d1d") Type { ring(36, 24) } }
        Entity { id("sapphireCup") template(true) startPosition(0, 0) color("1e3a8a") Type { ring(36, 24) } }
        Entity {
            id("ruby")
            template(true)
            startPosition(0, 0)
            color("ef4444")
            Type { circle(40) }
            solid(true)
            Events {
                OnCollision("rubyCup") {
                    Actions {
                        sound("coin")
                        burst(14, "fca5a5")
                        updateBar("pocketed", 1)
                        deleteEntity()
                    }
                }
            }
        }
        Entity {
            id("sapphire")
            template(true)
            startPosition(0, 0)
            color("3b82f6")
            Type { circle(40) }
            solid(true)
            Events {
                OnCollision("sapphireCup") {
                    Actions {
                        sound("coin")
                        burst(14, "93c5fd")
                        updateBar("pocketed", 1)
                        deleteEntity()
                    }
                }
            }
        }

        # The tray sets out the cups and marbles, and tips on every swipe. Marbles never fuse,
        # so slideAll gets no merge variable and no merge signal (two empty strings); rubies and
        # sapphires are rolled in two rounds so either colour can close up behind the other.
        Entity {
            id("tray")
            startPosition(0, 0)
            color("00000000")
            Type { circle(1) }
            Events {
                OnElapsedTime {
                    Actions {
                        spawn("rubyCup", 514, 664)
                        spawn("sapphireCup", 614, 764)
                        spawn("ruby", 410, 260)
                        spawn("sapphire", 310, 660)
                    }
                }
                OnSwipe("left") { Actions { sound("bubble") updateBar("tilts", -1)
                    slideAll(-1, 0, 100, "ruby", "", "") slideAll(-1, 0, 100, "sapphire", "", "")
                    slideAll(-1, 0, 100, "ruby", "", "") slideAll(-1, 0, 100, "sapphire", "", "") } }
                OnSwipe("right") { Actions { sound("bubble") updateBar("tilts", -1)
                    slideAll(1, 0, 100, "ruby", "", "") slideAll(1, 0, 100, "sapphire", "", "")
                    slideAll(1, 0, 100, "ruby", "", "") slideAll(1, 0, 100, "sapphire", "", "") } }
                OnSwipe("up") { Actions { sound("bubble") updateBar("tilts", -1)
                    slideAll(0, 1, 100, "ruby", "", "") slideAll(0, 1, 100, "sapphire", "", "")
                    slideAll(0, 1, 100, "ruby", "", "") slideAll(0, 1, 100, "sapphire", "", "") } }
                OnSwipe("down") { Actions { sound("bubble") updateBar("tilts", -1)
                    slideAll(0, -1, 100, "ruby", "", "") slideAll(0, -1, 100, "sapphire", "", "")
                    slideAll(0, -1, 100, "ruby", "", "") slideAll(0, -1, 100, "sapphire", "", "") } }
            }
        }
    }

    Level {
        Entity { id("title2") startPosition(100, 950) color("cbd5e1") Type { text("TRAY 2/3", 44) } }
        Entity { id("retry2") startPosition(540, 960) color("fca5a5") Type { text("RETRY", 36) }
            Events { OnTapped("retry2") { Actions { sound("click") restart() } } } }
        Entity { id("rimW") startPosition(80, 230) color("1e293b") Type { rectangle(20, 640) } solid(true) }
        Entity { id("rimE") startPosition(700, 230) color("1e293b") Type { rectangle(20, 640) } solid(true) }
        Entity { id("rimN") startPosition(80, 850) color("1e293b") Type { rectangle(640, 20) } solid(true) }
        Entity { id("rimS") startPosition(80, 230) color("1e293b") Type { rectangle(640, 20) } solid(true) }
        Entity { id("stud2a") startPosition(100, 650) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud2b") startPosition(200, 250) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud2c") startPosition(200, 350) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud2d") startPosition(200, 550) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud2e") startPosition(400, 250) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud2f") startPosition(600, 350) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud2g") startPosition(600, 750) color("475569") Type { rectangle(100, 100) } solid(true) }

        Entity { id("rubyCup") template(true) startPosition(0, 0) color("7f1d1d") Type { ring(36, 24) } }
        Entity { id("sapphireCup") template(true) startPosition(0, 0) color("1e3a8a") Type { ring(36, 24) } }
        Entity {
            id("ruby")
            template(true)
            startPosition(0, 0)
            color("ef4444")
            Type { circle(40) }
            solid(true)
            Events {
                OnCollision("rubyCup") {
                    Actions {
                        sound("coin")
                        burst(14, "fca5a5")
                        updateBar("pocketed", 1)
                        deleteEntity()
                    }
                }
            }
        }
        Entity {
            id("sapphire")
            template(true)
            startPosition(0, 0)
            color("3b82f6")
            Type { circle(40) }
            solid(true)
            Events {
                OnCollision("sapphireCup") {
                    Actions {
                        sound("coin")
                        burst(14, "93c5fd")
                        updateBar("pocketed", 1)
                        deleteEntity()
                    }
                }
            }
        }

        # The tray sets out the cups and marbles, and tips on every swipe. Marbles never fuse,
        # so slideAll gets no merge variable and no merge signal (two empty strings); rubies and
        # sapphires are rolled in two rounds so either colour can close up behind the other.
        Entity {
            id("tray")
            startPosition(0, 0)
            color("00000000")
            Type { circle(1) }
            Events {
                OnElapsedTime {
                    Actions {
                        spawn("rubyCup", 514, 764)
                        spawn("rubyCup", 414, 764)
                        spawn("sapphireCup", 314, 764)
                        spawn("ruby", 210, 460)
                        spawn("ruby", 210, 760)
                        spawn("sapphire", 510, 560)
                    }
                }
                OnSwipe("left") { Actions { sound("bubble") updateBar("tilts", -1)
                    slideAll(-1, 0, 100, "ruby", "", "") slideAll(-1, 0, 100, "sapphire", "", "")
                    slideAll(-1, 0, 100, "ruby", "", "") slideAll(-1, 0, 100, "sapphire", "", "") } }
                OnSwipe("right") { Actions { sound("bubble") updateBar("tilts", -1)
                    slideAll(1, 0, 100, "ruby", "", "") slideAll(1, 0, 100, "sapphire", "", "")
                    slideAll(1, 0, 100, "ruby", "", "") slideAll(1, 0, 100, "sapphire", "", "") } }
                OnSwipe("up") { Actions { sound("bubble") updateBar("tilts", -1)
                    slideAll(0, 1, 100, "ruby", "", "") slideAll(0, 1, 100, "sapphire", "", "")
                    slideAll(0, 1, 100, "ruby", "", "") slideAll(0, 1, 100, "sapphire", "", "") } }
                OnSwipe("down") { Actions { sound("bubble") updateBar("tilts", -1)
                    slideAll(0, -1, 100, "ruby", "", "") slideAll(0, -1, 100, "sapphire", "", "")
                    slideAll(0, -1, 100, "ruby", "", "") slideAll(0, -1, 100, "sapphire", "", "") } }
            }
        }
    }

    Level {
        Entity { id("title3") startPosition(100, 950) color("cbd5e1") Type { text("TRAY 3/3", 44) } }
        Entity { id("retry3") startPosition(540, 960) color("fca5a5") Type { text("RETRY", 36) }
            Events { OnTapped("retry3") { Actions { sound("click") restart() } } } }
        Entity { id("rimW") startPosition(80, 230) color("1e293b") Type { rectangle(20, 640) } solid(true) }
        Entity { id("rimE") startPosition(700, 230) color("1e293b") Type { rectangle(20, 640) } solid(true) }
        Entity { id("rimN") startPosition(80, 850) color("1e293b") Type { rectangle(640, 20) } solid(true) }
        Entity { id("rimS") startPosition(80, 230) color("1e293b") Type { rectangle(640, 20) } solid(true) }
        Entity { id("stud3a") startPosition(100, 350) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud3b") startPosition(100, 550) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud3c") startPosition(200, 250) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud3d") startPosition(200, 550) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud3e") startPosition(400, 750) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud3f") startPosition(500, 350) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud3g") startPosition(500, 550) color("475569") Type { rectangle(100, 100) } solid(true) }
        Entity { id("stud3h") startPosition(600, 350) color("475569") Type { rectangle(100, 100) } solid(true) }

        Entity { id("rubyCup") template(true) startPosition(0, 0) color("7f1d1d") Type { ring(36, 24) } }
        Entity { id("sapphireCup") template(true) startPosition(0, 0) color("1e3a8a") Type { ring(36, 24) } }
        Entity {
            id("ruby")
            template(true)
            startPosition(0, 0)
            color("ef4444")
            Type { circle(40) }
            solid(true)
            Events {
                OnCollision("rubyCup") {
                    Actions {
                        sound("coin")
                        burst(14, "fca5a5")
                        updateBar("pocketed", 1)
                        deleteEntity()
                    }
                }
            }
        }
        Entity {
            id("sapphire")
            template(true)
            startPosition(0, 0)
            color("3b82f6")
            Type { circle(40) }
            solid(true)
            Events {
                OnCollision("sapphireCup") {
                    Actions {
                        sound("coin")
                        burst(14, "93c5fd")
                        updateBar("pocketed", 1)
                        deleteEntity()
                    }
                }
            }
        }

        # The tray sets out the cups and marbles, and tips on every swipe. Marbles never fuse,
        # so slideAll gets no merge variable and no merge signal (two empty strings); rubies and
        # sapphires are rolled in two rounds so either colour can close up behind the other.
        Entity {
            id("tray")
            startPosition(0, 0)
            color("00000000")
            Type { circle(1) }
            Events {
                OnElapsedTime {
                    Actions {
                        spawn("rubyCup", 614, 264)
                        spawn("rubyCup", 314, 764)
                        spawn("sapphireCup", 414, 264)
                        spawn("sapphireCup", 314, 264)
                        spawn("ruby", 110, 660)
                        spawn("ruby", 110, 460)
                        spawn("sapphire", 210, 460)
                        spawn("sapphire", 510, 460)
                    }
                }
                OnSwipe("left") { Actions { sound("bubble") updateBar("tilts", -1)
                    slideAll(-1, 0, 100, "ruby", "", "") slideAll(-1, 0, 100, "sapphire", "", "")
                    slideAll(-1, 0, 100, "ruby", "", "") slideAll(-1, 0, 100, "sapphire", "", "") } }
                OnSwipe("right") { Actions { sound("bubble") updateBar("tilts", -1)
                    slideAll(1, 0, 100, "ruby", "", "") slideAll(1, 0, 100, "sapphire", "", "")
                    slideAll(1, 0, 100, "ruby", "", "") slideAll(1, 0, 100, "sapphire", "", "") } }
                OnSwipe("up") { Actions { sound("bubble") updateBar("tilts", -1)
                    slideAll(0, 1, 100, "ruby", "", "") slideAll(0, 1, 100, "sapphire", "", "")
                    slideAll(0, 1, 100, "ruby", "", "") slideAll(0, 1, 100, "sapphire", "", "") } }
                OnSwipe("down") { Actions { sound("bubble") updateBar("tilts", -1)
                    slideAll(0, -1, 100, "ruby", "", "") slideAll(0, -1, 100, "sapphire", "", "")
                    slideAll(0, -1, 100, "ruby", "", "") slideAll(0, -1, 100, "sapphire", "", "") } }
            }
        }
    }
}
