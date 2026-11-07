# Lava Tiles: the floor of the crater cracks open tile by tile. Orange = about to erupt, red = lava.
Window {
    size(8, 10)
    background("1c0a00")
    music("bg_2")
    gravity(0)

    lives(3, "-- MELTED --", "lose")
    Bar("heat") {
        initial(1)
        OnValue(4) { Actions { broadcast("wisps") } }
    }

    Entity { id("ta1") startPosition(25, 25) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("ha1") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("tb1") startPosition(215, 25) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hb1") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("tc1") startPosition(405, 25) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hc1") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("td1") startPosition(595, 25) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hd1") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("ta2") startPosition(25, 215) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("ha2") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("tb2") startPosition(215, 215) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hb2") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("tc2") startPosition(405, 215) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hc2") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("td2") startPosition(595, 215) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hd2") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("ta3") startPosition(25, 405) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("ha3") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("tb3") startPosition(215, 405) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hb3") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("tc3") startPosition(405, 405) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hc3") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("td3") startPosition(595, 405) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hd3") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("ta4") startPosition(25, 595) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("ha4") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("tb4") startPosition(215, 595) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hb4") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("tc4") startPosition(405, 595) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hc4") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("td4") startPosition(595, 595) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hd4") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("ta5") startPosition(25, 785) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("ha5") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("tb5") startPosition(215, 785) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hb5") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("tc5") startPosition(405, 785) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hc5") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }
    Entity { id("td5") startPosition(595, 785) color("57534e") Type { rectangle(180, 180) }
        var("hot", 0)
        Events { OnSignal("hd5") { Actions { whenVar("hot", "==", 0) { Actions {
            changeVar("hot", 1) setColor("f97316") sound("click")
            delay(1.0) { Actions { setColor("dc2626") spawnHere("lava") } }
            delay(1.9) { Actions { spawnHere("lava") } }
            delay(2.8) { Actions { setColor("57534e") changeVar("hot", -1) } } } } } } } }

    Entity {
        id("picker")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSignal("erupt") {
                Actions {
                    randomAction {
                    Actions { broadcast("ha1") }
                    Actions { broadcast("hb1") }
                    Actions { broadcast("hc1") }
                    Actions { broadcast("hd1") }
                    Actions { broadcast("ha2") }
                    Actions { broadcast("hb2") }
                    Actions { broadcast("hc2") }
                    Actions { broadcast("hd2") }
                    Actions { broadcast("ha3") }
                    Actions { broadcast("hb3") }
                    Actions { broadcast("hc3") }
                    Actions { broadcast("hd3") }
                    Actions { broadcast("ha4") }
                    Actions { broadcast("hb4") }
                    Actions { broadcast("hc4") }
                    Actions { broadcast("hd4") }
                    Actions { broadcast("ha5") }
                    Actions { broadcast("hb5") }
                    Actions { broadcast("hc5") }
                    Actions { broadcast("hd5") }
                    }
                }
            }
        }
    }

    Entity {
        id("hiker")
        startPosition(380, 480)
        color("e0f2fe")
        Type { circle(20) }
        Movement {
            speed(6)
            Control { movementControl(0) }
        }
    }

    Entity {
        id("vent")
        startPosition(0, 965)
        color("7f1d1d")
        Type { rectangle(800, 35) }
        Events {
            OnElapsedTime {
                Actions {
                    ramp("heat", 1, 9, 60)
                    setEnable("wisp1", false)
                    setEnable("wisp2", false)
                }
            }
            OnElapsedTime(1.0) {
                repeatCount(-1)
                Actions {
                    broadcast("erupt")
                    whenBar("heat", ">=", 3) { Actions { broadcast("erupt") } }
                    whenBar("heat", ">=", 6) { Actions { broadcast("erupt") } }
                }
            }
            OnElapsedTime(12) {
                repeatCount(-1)
                Actions { spawn("spring", random(40, 740), random(40, 920)) }
            }
            OnElapsedTime(60) {
                Actions {
                    whenBar("lives", ">", 0) {
                        Actions {
                            sound("win")
                            burst("hiker", 40, "fde68a")
                            shake(5, 0.5)
                            gameOver("-- THE ERUPTION ENDS --")
                        }
                    }
                }
            }
        }
    }

    Entity {
        id("lava")
        template(true)
        startPosition(0, 0)
        color("ccea580c")
        Type { rectangle(180, 180) }
        Events {
            OnElapsedTime { Actions { sound("thud") burst(12, "fdba74") } }
            OnCollision("hiker") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("hiker", "ff0000", 0.4)
                    shake(7, 0.3)
                    burst("hiker", 14, "f97316")
                }
            }
            OnElapsedTime(0.8) { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("spring")
        template(true)
        startPosition(0, 0)
        color("38bdf8")
        Type { polygon(6, 18) }
        Events {
            OnCollision("hiker") {
                Actions {
                    sound("pickup")
                    burst(12, "7dd3fc")
                    whenBar("lives", "<", 3) { Actions { updateBar("lives", 1) flash("hiker", "38bdf8", 0.3) } }
                    deleteEntity()
                }
            }
            OnElapsedTime(5) { Actions { deleteEntity() } }
        }
    }

    # fire wisps wake up once the heat reaches 4
    Entity { id("wisp1") startPosition(0, 0) color("fde047") Type { circle(16) }
        Movement { speed(2.5) Path { waypoint(20, 395) waypoint(750, 395) } }
        Events {
            OnSignal("wisps") { Actions { setEnable(true) sound("buzz") burst(16, "fde047") } }
            OnCollision("hiker") { Actions { sound("hurt") updateBar("lives", -1) flash("hiker", "ff0000", 0.4) shake(6, 0.3) } }
        } }
    Entity { id("wisp2") startPosition(0, 0) color("fde047") Type { circle(16) }
        Movement { speed(2.5) Path { waypoint(395, 930) waypoint(395, 20) } }
        Events {
            OnSignal("wisps") { Actions { setEnable(true) sound("buzz") burst(16, "fde047") } }
            OnCollision("hiker") { Actions { sound("hurt") updateBar("lives", -1) flash("hiker", "ff0000", 0.4) shake(6, 0.3) } }
        } }
}
