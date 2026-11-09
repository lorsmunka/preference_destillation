Window {
    size(8, 10)
    background("14213d")
    music("bg_4")
    gravity(0)
    Bar("coins") { initial(0) }

    Entity {
        id("hero")
        startPosition(60, 40)
        color("4ade80")
        Type { circle(24) }
        var("purse", 0)
        Movement { speed(5) Control { movementControl(0) } }
        Events {
            OnSignal("gotcoin") {
                Actions {
                    changeVar("purse", 1)
                    updateBar("coins", 1)
                }
            }
            OnCollision("toll1") {
                Actions {
                    whenVar("purse", ">=", 3) {
                        Actions {
                            changeVar("purse", -3)
                            updateBar("coins", -3)
                            broadcast("open1")
                        }
                    }
                    whenVar("purse", "<", 3) { Actions { sound("buzz") flash("ef4444", 0.3) } }
                }
            }
            OnCollision("toll2") {
                Actions {
                    whenVar("purse", ">=", 4) {
                        Actions {
                            changeVar("purse", -4)
                            updateBar("coins", -4)
                            broadcast("open2")
                        }
                    }
                    whenVar("purse", "<", 4) { Actions { sound("buzz") flash("ef4444", 0.3) } }
                }
            }
        }
    }

    # zone 1: four coins, the first toll costs 3
    Entity { id("coin1") startPosition(120, 120) color("fde047") Type { circle(16) }
        Events { OnCollision("hero") { Actions { sound("coin") broadcast("gotcoin") deleteEntity() } } } }
    Entity { id("coin2") startPosition(640, 100) color("fde047") Type { circle(16) }
        Events { OnCollision("hero") { Actions { sound("coin") broadcast("gotcoin") deleteEntity() } } } }
    Entity { id("coin3") startPosition(380, 220) color("fde047") Type { circle(16) }
        Events { OnCollision("hero") { Actions { sound("coin") broadcast("gotcoin") deleteEntity() } } } }
    Entity { id("coin4") startPosition(120, 290) color("fde047") Type { circle(16) }
        Events { OnCollision("hero") { Actions { sound("coin") broadcast("gotcoin") deleteEntity() } } } }
    Entity {
        id("sweeper1")
        startPosition(80, 180)
        color("f43f5e")
        Type { rectangle(30, 30) }
        hostile("hero", "-- SWEPT AWAY --", "lose")
        Movement { speed(3) Path { waypoint(80, 180) waypoint(700, 180) } }
    }
    Entity { id("toll1") startPosition(360, 350) color("a16207") Type { rectangle(80, 24) } }
    Entity { id("gate1") startPosition(0, 390) color("92400e") Type { rectangle(800, 30) } door("open1", "clang") }

    # zone 2: five coins, the second toll costs 4
    Entity { id("coin5") startPosition(80, 480) color("fde047") Type { circle(16) } 
        Events { OnCollision("hero") { Actions { sound("coin") broadcast("gotcoin") deleteEntity() } } } }
    Entity { id("coin6") startPosition(660, 520) color("fde047") Type { circle(16) }
        Events { OnCollision("hero") { Actions { sound("coin") broadcast("gotcoin") deleteEntity() } } } }
    Entity { id("coin7") startPosition(400, 600) color("fde047") Type { circle(16) }
        Events { OnCollision("hero") { Actions { sound("coin") broadcast("gotcoin") deleteEntity() } } } }
    Entity { id("coin8") startPosition(100, 650) color("fde047") Type { circle(16) }
        Events { OnCollision("hero") { Actions { sound("coin") broadcast("gotcoin") deleteEntity() } } } }
    Entity { id("coin9") startPosition(690, 660) color("fde047") Type { circle(16) }
        Events { OnCollision("hero") { Actions { sound("coin") broadcast("gotcoin") deleteEntity() } } } }
    Entity {
        id("sweeper2")
        startPosition(700, 560)
        color("f97316")
        Type { rectangle(30, 30) }
        hostile("hero", "-- SWEPT AWAY --", "lose")
        Movement { speed(4) Path { waypoint(700, 560) waypoint(60, 560) dwell(0.4) } }
    }
    Entity { id("toll2") startPosition(360, 700) color("a16207") Type { rectangle(80, 24) } }
    Entity { id("gate2") startPosition(0, 740) color("92400e") Type { rectangle(800, 30) } door("open2", "clang") }

    goal("exit", 350, 880, 100, 70, "-- TOLLS PAID --")
}
