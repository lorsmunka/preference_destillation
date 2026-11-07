# After hours: returned books keep turning up in the aisles. The librarian walks until a shelf
# stops her, carries one book at a time to the desk (bottom left), and can shove the book
# trolleys around -- but a trolley pushed into a gap between shelves blocks it.
Window {
    size(8, 12)
    background("1c1917")
    music("bg_ambient")
    gravity(0)

    Bar("shelved") {
        initial(0)
    }
    winAt("shelved", 15, "-- EVERY BOOK IS HOME --", "win")
    Bar("piles") {
        initial(0)
        OnValue(10) {
            Actions {
                sound("lose")
                gameOver("-- BURIED IN RETURNS --")
            }
        }
    }
    Bar("clock") {
        initial(120)
        OnValue(0) {
            Actions {
                sound("lose")
                gameOver("-- CLOSING TIME --")
            }
        }
    }

    dpad("librarian")

    Entity { id("shelf1") startPosition(100, 1000) color("7c2d12") Type { rectangle(300, 100) } solid(true) }
    Entity { id("shelf2") startPosition(500, 1000) color("7c2d12") Type { rectangle(200, 100) } solid(true) }
    Entity { id("shelf3") startPosition(0, 800) color("7c2d12") Type { rectangle(300, 100) } solid(true) }
    Entity { id("shelf4") startPosition(400, 800) color("7c2d12") Type { rectangle(300, 100) } solid(true) }
    Entity { id("shelf5") startPosition(100, 600) color("7c2d12") Type { rectangle(200, 100) } solid(true) }
    Entity { id("shelf6") startPosition(400, 600) color("7c2d12") Type { rectangle(400, 100) } solid(true) }

    Entity {
        id("desk")
        startPosition(0, 400)
        color("a16207")
        Type { rectangle(100, 100) }
    }

    Entity {
        id("trolleyA")
        startPosition(505, 715)
        color("64748b")
        Type { rectangle(90, 70) }
        pushable(true)
        Events {
            OnGridStep { Actions { sound("bump") } }
        }
    }
    Entity {
        id("trolleyB")
        startPosition(205, 915)
        color("64748b")
        Type { rectangle(90, 70) }
        pushable(true)
        Events {
            OnGridStep { Actions { sound("bump") } }
        }
    }

    Entity {
        id("book")
        template(true)
        startPosition(0, 0)
        color("60a5fa")
        Type { rectangle(80, 56) }
    }

    Entity {
        id("returnsSlot")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(4, 4.0) {
                    Actions {
                        sound("thud")
                        updateBar("piles", 1)
                        randomAction {
                            Actions { spawnAtEmpty("book", 100, 0, 500, 8, 1) }
                            Actions { spawnAtEmpty("book", 100, 0, 700, 8, 1) }
                            Actions { spawnAtEmpty("book", 100, 0, 900, 8, 1) }
                            Actions { spawnAtEmpty("book", 100, 0, 1100, 8, 1) }
                        }
                    }
                }
                Phase(-1, 3.0) {
                    Actions {
                        sound("thud")
                        updateBar("piles", 1)
                        randomAction {
                            Actions { spawnAtEmpty("book", 100, 0, 500, 8, 1) }
                            Actions { spawnAtEmpty("book", 100, 0, 700, 8, 1) }
                            Actions { spawnAtEmpty("book", 100, 0, 900, 8, 1) }
                            Actions { spawnAtEmpty("book", 100, 0, 1100, 8, 1) }
                        }
                    }
                }
            }
            OnElapsedTime(1) {
                repeatCount(-1)
                Actions { updateBar("clock", -1) }
            }
        }
    }

    Entity {
        id("librarian")
        startPosition(310, 410)
        color("38bdf8")
        Type { circle(40) }
        var("carrying", 0)
        Movement { GridMove(100, 0.22) { direction(0, 0) } }
        Events {
            OnGridBlocked {
                Actions {
                    sound("bump")
                    setDirection("librarian", 0, 0)
                }
            }
            OnCollision("book") {
                Actions {
                    whenVar("carrying", "==", 0) {
                        Actions {
                            sound("pickup")
                            changeVar("carrying", 1)
                            setColor("f59e0b")
                            updateBar("piles", -1)
                            deleteCollider()
                        }
                    }
                }
            }
            OnCollision("desk") {
                Actions {
                    whenVar("carrying", "==", 1) {
                        Actions {
                            sound("coin")
                            burst(10, "fde68a")
                            changeVar("carrying", -1)
                            setColor("38bdf8")
                            updateBar("shelved", 1)
                        }
                    }
                }
            }
        }
    }
}
