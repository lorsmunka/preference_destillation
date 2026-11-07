# Hive Keeper: tap a flower to send a forager bee; every bee that gets home adds honey and a new comb cell.
# The summer keeps heating the hive and wasps stir it up: swat them and work the fan before the colony swarms.
Window {
    size(8, 12)
    background("1c1408")
    music("bg_1")
    gravity(0)

    Bar("honey") {
        initial(0)
        OnValue(8) {
            Actions {
                setScale("hive", 1.1)
                sound("sparkle")
                burst("hive", 16, "fbbf24")
            }
        }
        OnValue(16) {
            Actions {
                setScale("hive", 1.2)
                sound("sparkle")
                burst("hive", 20, "fbbf24")
            }
        }
    }
    winAt("honey", 24, "-- THE HIVE IS FULL OF HONEY --", "win")

    Bar("heat") {
        initial(2)
        OnValue(5) { Actions { broadcast("cooled") } }
        OnValue(6) { Actions { broadcast("hot") } }
        OnValue(8) { Actions { broadcast("swarming") } }
        OnValue(10) {
            Actions {
                sound("lose")
                shake(9, 0.6)
                burst("hive", 40, "f59e0b")
                gameOver("-- THE COLONY SWARMED AWAY --")
            }
        }
    }

    Wave(20, 10) { Actions { randomAction {
        Actions { spawn("wasp", 10, random(300, 1000)) }
        Actions { spawn("wasp", 760, random(300, 1000)) } } } }
    Wave(30, 6) { Actions { randomAction {
        Actions { spawn("wasp", 10, random(300, 1000)) }
        Actions { spawn("wasp", 760, random(300, 1000)) }
        Actions { spawn("wasp", random(50, 750), 1150) } } } }
    Wave(80, 3.5) { Actions { randomAction {
        Actions { spawn("wasp", 10, random(300, 1000)) }
        Actions { spawn("wasp", 760, random(300, 1000)) }
        Actions { spawn("wasp", random(50, 750), 1150) } } } }

    Entity {
        id("label")
        startPosition(40, 1130)
        color("fde68a")
        Type { text("The hive hums", 34) }
        Events {
            OnSignal("cooled") { Actions { setText("label", "The hive hums") setColor("fde68a") } }
            OnSignal("hot") { Actions { setText("label", "Too hot! Work the fan") setColor("fb923c") } }
            OnSignal("swarming") { Actions { setText("label", "The bees are about to swarm!") setColor("ef4444") } }
        }
    }

    Entity {
        id("hive")
        startPosition(330, 520)
        color("f59e0b")
        Type { polygon(6, 70) }
        Events {
            Sequence {
                Phase(8, 4) { Actions { updateBar("heat", 1) } }
                Phase(-1, 3) { Actions { updateBar("heat", 1) } }
            }
            OnSignal("hot") { Actions { setColor("ea580c") sound("buzz") } }
            OnSignal("cooled") { Actions { setColor("f59e0b") } }
            OnSignal("swarming") { Actions { flash("ef4444", 0.4) shake(5, 0.4) } }
        }
    }

    Entity {
        id("fan")
        startPosition(80, 540)
        color("e2e8f0")
        Type { cross(100, 100, 26) }
        var("spin", 0)
        Events {
            OnTapped("fan") {
                Actions {
                    whenVar("spin", "==", 0) { Actions { whenBar("heat", ">", 0) { Actions {
                        changeVar("spin", 1)
                        updateBar("heat", -1)
                        sound("click")
                        burst(6, "bae6fd")
                        setOpacity(0.35)
                        delay(1.2) { Actions { changeVar("spin", -1) setOpacity(1) } }
                    } } } }
                }
            }
        }
    }

    Entity {
        id("f1")
        startPosition(80, 930)
        color("f472b6")
        Type { star(8, 42, 20) }
        var("nectar", 3)
        Events {
            OnTapped("f1") { Actions {
                whenVar("nectar", ">", 0) { Actions { changeVar("nectar", -1) spawnHere("bee") sound("pop") } }
                whenVar("nectar", "==", 0) { Actions { setColor("57534e") } } } }
            OnElapsedTime(random(4, 6)) { repeatCount(-1) Actions {
                whenVar("nectar", "<", 3) { Actions { changeVar("nectar", 1) setColor("f472b6") burst(4, "f9a8d4") } } } }
        }
    }

    Entity {
        id("f2")
        startPosition(358, 1000)
        color("a78bfa")
        Type { star(8, 42, 20) }
        var("nectar", 3)
        Events {
            OnTapped("f2") { Actions {
                whenVar("nectar", ">", 0) { Actions { changeVar("nectar", -1) spawnHere("bee") sound("pop") } }
                whenVar("nectar", "==", 0) { Actions { setColor("57534e") } } } }
            OnElapsedTime(random(4, 6)) { repeatCount(-1) Actions {
                whenVar("nectar", "<", 3) { Actions { changeVar("nectar", 1) setColor("a78bfa") burst(4, "c4b5fd") } } } }
        }
    }

    Entity {
        id("f3")
        startPosition(636, 930)
        color("facc15")
        Type { star(8, 42, 20) }
        var("nectar", 3)
        Events {
            OnTapped("f3") { Actions {
                whenVar("nectar", ">", 0) { Actions { changeVar("nectar", -1) spawnHere("bee") sound("pop") } }
                whenVar("nectar", "==", 0) { Actions { setColor("57534e") } } } }
            OnElapsedTime(random(4, 6)) { repeatCount(-1) Actions {
                whenVar("nectar", "<", 3) { Actions { changeVar("nectar", 1) setColor("facc15") burst(4, "fef08a") } } } }
        }
    }

    Entity {
        id("bee")
        template(true)
        color("fde047")
        Type { circle(10) }
        Movement { speed(2.2) chase("hive") }
        Events {
            OnCollision("hive") {
                Actions {
                    sound("coin")
                    updateBar("honey", 1)
                    spawnAtEmpty("comb", 50, 100, 80, 12, 2)
                    deleteEntity()
                }
            }
            OnCollision("wasp") { Actions { burst(6, "fde047") deleteEntity() } }
        }
    }

    Entity {
        id("wasp")
        template(true)
        color("dc2626")
        Type { triangle(30, "down") }
        Movement { speed(1.1) chase("hive") }
        Events {
            OnTapped { Actions { sound("pop") burst(10, "dc2626") deleteEntity() } }
            OnCollision("hive") {
                Actions {
                    sound("buzz")
                    shake(4, 0.3)
                    flash("hive", "ef4444", 0.3)
                    deleteEntity()
                    loopCount(2)
                    updateBar("heat", 1)
                }
            }
        }
    }

    Entity {
        id("comb")
        template(true)
        color("d97706")
        Type { polygon(6, 24) }
    }
}
