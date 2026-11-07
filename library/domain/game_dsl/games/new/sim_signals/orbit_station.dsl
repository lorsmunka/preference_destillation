# Orbit Station: keep a crippled space station alive until the rescue shuttle docks.
# Oxygen, power and water all drain. Tap the scrubber, the solar panel and the recycler to top them up;
# the scrubber and recycler need power above 2, and the solar panel only works on the day side of the orbit.
# Systems break down now and then (tap a broken one 3 times to repair it), and debris drifting into the hull
# breaks something too, so tap the debris away. Any meter at 0 ends the mission.
Window {
    size(9, 16)
    background("020617")
    music("bg_3")
    gravity(0)

    Bar("oxygen") {
        initial(8)
        OnValue(3) { Actions { broadcast("o2Low") } }
        OnValue(0) { Actions { sound("death") shake(8, 0.6) gameOver("-- THE AIR RAN OUT --") } }
    }
    Bar("power") {
        initial(8)
        OnValue(2) { Actions { broadcast("brownout") } }
        OnValue(3) { Actions { broadcast("powerOk") } }
        OnValue(0) { Actions { sound("death") shake(8, 0.6) gameOver("-- THE STATION WENT DARK --") } }
    }
    Bar("water") {
        initial(8)
        OnValue(3) { Actions { broadcast("waterLow") } }
        OnValue(0) { Actions { sound("death") shake(8, 0.6) gameOver("-- THE CREW RAN DRY --") } }
    }
    Bar("daylight") {
        initial(1)
        OnValue(2) { Actions { updateBar("daylight", -2) } }
        OnValue(0) { Actions { broadcast("orbitNight") } }
        OnValue(1) { Actions { broadcast("orbitDay") } }
    }
    Bar("rescue") {
        initial(0)
        OnValue(50) { Actions { broadcast("halfway") } }
        OnValue(85) { Actions { broadcast("shuttleSighted") } }
        OnValue(92) { Actions { setScale("shuttle", 0.7) } }
        OnValue(100) {
            Actions {
                sound("win")
                shake(5, 0.6)
                burst("hull", 40, "a5f3fc")
                gameOver("-- THE SHUTTLE DOCKED, EVERYONE IS SAFE --")
            }
        }
    }

    Wave(40, 10) { Actions { randomAction {
        Actions { spawn("debris", 10, random(860, 1080)) }
        Actions { spawn("debris", 850, random(860, 1080)) } } } }
    Wave(60, 6) { Actions { randomAction {
        Actions { spawn("debris", 10, random(860, 1080)) }
        Actions { spawn("debris", 850, random(860, 1080)) } } } }
    Wave(90, 4) { Actions { randomAction {
        Actions { spawn("debris", 10, random(860, 1080)) }
        Actions { spawn("debris", 850, random(860, 1080)) } } } }

    Entity { id("earth") startPosition(150, 20) color("2563eb") Type { circle(300) }
        Events {
            OnSignal("orbitNight") { Actions { setColor("1e3a8a") } }
            OnSignal("orbitDay") { Actions { setColor("2563eb") } } } }

    Entity { id("sunGlow") startPosition(740, 1420) color("fde047") Type { circle(60) }
        Events {
            Sequence { Phase(-1, 12) { Actions { updateBar("daylight", 1) } } }
            OnSignal("orbitNight") { Actions { setOpacity(0.15) } }
            OnSignal("orbitDay") { Actions { setOpacity(1) } } } }

    Entity { id("radio") startPosition(40, 1450) color("94a3b8") Type { text("Rescue shuttle on its way", 30) }
        Events {
            OnElapsedTime { Actions { ramp("rescue", 0, 100, 150) } }
            OnSignal("halfway") { Actions { setText("radio", "Shuttle halfway here") sound("bell") } }
            OnSignal("shuttleSighted") { Actions { setText("radio", "Shuttle in sight - hold on!") setColor("4ade80") sound("bell") } } } }

    Entity { id("alert") startPosition(40, 1520) color("e2e8f0") Type { text("All systems nominal", 32) }
        Events {
            OnSignal("o2Low") { Actions { setText("alert", "OXYGEN LOW") setColor("f87171") } }
            OnSignal("waterLow") { Actions { setText("alert", "WATER LOW") setColor("f87171") } }
            OnSignal("brownout") { Actions { setText("alert", "BROWNOUT: scrubber and recycler offline") setColor("fbbf24") } }
            OnSignal("powerOk") { Actions { setText("alert", "Power restored") setColor("e2e8f0") } }
            OnSignal("orbitNight") { Actions { setText("alert", "Night side: the solar panel is idle") setColor("93c5fd") } }
            OnSignal("orbitDay") { Actions { setText("alert", "Day side: the solar panel is live") setColor("fde68a") } } } }

    Entity {
        id("hull")
        startPosition(300, 850)
        color("cbd5e1")
        Type { rectangle(300, 220) }
        Events {
            OnElapsedTime(3) { repeatCount(-1) Actions { updateBar("oxygen", -1) } }
            OnElapsedTime(3.5) { repeatCount(-1) Actions { updateBar("power", -1) } }
            OnElapsedTime(4) { repeatCount(-1) Actions { updateBar("water", -1) } }
        }
    }

    Entity { id("cabinLights") startPosition(320, 870) color("80fde68a") Type { rectangle(260, 180) }
        Events {
            OnSignal("brownout") { Actions { setOpacity(0.2) sound("buzz") } }
            OnSignal("powerOk") { Actions { setOpacity(1) } } } }

    Entity {
        id("scrubber")
        startPosition(60, 1180)
        color("22d3ee")
        Type { rectangle(160, 110) }
        var("broken", 0)
        var("repair", 0)
        var("busy", 0)
        Events {
            OnTapped("scrubber") {
                Actions {
                    whenVar("broken", "==", 0) { Actions { whenVar("busy", "==", 0) { Actions {
                        whenBar("power", "<=", 2) { Actions { sound("buzz") flash("475569", 0.3) } }
                        whenBar("power", ">", 2) { Actions { whenBar("oxygen", "<", 10) { Actions {
                            changeVar("busy", 1) updateBar("oxygen", 1) sound("bubble") burst(8, "a5f3fc")
                            delay(0.6) { Actions { changeVar("busy", -1) } } } } } }
                    } } } }
                    whenVar("broken", "==", 1) { Actions {
                        changeVar("repair", 1) sound("clang") burst(6, "fbbf24")
                        whenVar("repair", "==", 3) { Actions {
                            changeVar("repair", -3) changeVar("broken", -1) setColor("22d3ee")
                            setEnable("o2Lamp", false) sound("sparkle") setText("alert", "Scrubber repaired") } }
                    } }
                }
            }
            OnElapsedTime(2) { repeatCount(-1) Actions { whenVar("broken", "==", 1) { Actions { updateBar("oxygen", -1) } } } }
            OnSignal("breakO2") { Actions { whenVar("broken", "==", 0) { Actions {
                changeVar("broken", 1) setColor("dc2626") setEnable("o2Lamp", true) sound("buzz") shake(4, 0.3)
                setText("alert", "SCRUBBER FAULT: tap it 3 times") } } } }
        }
    }

    Entity {
        id("solarPanel")
        startPosition(370, 1180)
        color("3b82f6")
        Type { rectangle(160, 110) }
        var("broken", 0)
        var("repair", 0)
        var("busy", 0)
        Events {
            OnTapped("solarPanel") {
                Actions {
                    whenVar("broken", "==", 0) { Actions { whenVar("busy", "==", 0) { Actions {
                        whenBar("daylight", "==", 0) { Actions { sound("buzz") flash("475569", 0.3) } }
                        whenBar("daylight", "==", 1) { Actions { whenBar("power", "<", 10) { Actions {
                            changeVar("busy", 1) updateBar("power", 1) sound("blip") burst(8, "fde047")
                            delay(0.6) { Actions { changeVar("busy", -1) } } } } } }
                    } } } }
                    whenVar("broken", "==", 1) { Actions {
                        changeVar("repair", 1) sound("clang") burst(6, "fbbf24")
                        whenVar("repair", "==", 3) { Actions {
                            changeVar("repair", -3) changeVar("broken", -1) setColor("3b82f6")
                            setEnable("pwrLamp", false) sound("sparkle") setText("alert", "Solar panel repaired") } }
                    } }
                }
            }
            OnElapsedTime(2) { repeatCount(-1) Actions { whenVar("broken", "==", 1) { Actions { updateBar("power", -1) } } } }
            OnSignal("breakPower") { Actions { whenVar("broken", "==", 0) { Actions {
                changeVar("broken", 1) setColor("dc2626") setEnable("pwrLamp", true) sound("buzz") shake(4, 0.3)
                setText("alert", "SOLAR PANEL FAULT: tap it 3 times") } } } }
            OnSignal("orbitNight") { Actions { whenVar("broken", "==", 0) { Actions { setColor("1e293b") } } } }
            OnSignal("orbitDay") { Actions { whenVar("broken", "==", 0) { Actions { setColor("3b82f6") } } } }
        }
    }

    Entity {
        id("recycler")
        startPosition(680, 1180)
        color("14b8a6")
        Type { rectangle(160, 110) }
        var("broken", 0)
        var("repair", 0)
        var("busy", 0)
        Events {
            OnTapped("recycler") {
                Actions {
                    whenVar("broken", "==", 0) { Actions { whenVar("busy", "==", 0) { Actions {
                        whenBar("power", "<=", 2) { Actions { sound("buzz") flash("475569", 0.3) } }
                        whenBar("power", ">", 2) { Actions { whenBar("water", "<", 10) { Actions {
                            changeVar("busy", 1) updateBar("water", 1) sound("bubble") burst(8, "5eead4")
                            delay(0.6) { Actions { changeVar("busy", -1) } } } } } }
                    } } } }
                    whenVar("broken", "==", 1) { Actions {
                        changeVar("repair", 1) sound("clang") burst(6, "fbbf24")
                        whenVar("repair", "==", 3) { Actions {
                            changeVar("repair", -3) changeVar("broken", -1) setColor("14b8a6")
                            setEnable("h2oLamp", false) sound("sparkle") setText("alert", "Recycler repaired") } }
                    } }
                }
            }
            OnElapsedTime(2) { repeatCount(-1) Actions { whenVar("broken", "==", 1) { Actions { updateBar("water", -1) } } } }
            OnSignal("breakWater") { Actions { whenVar("broken", "==", 0) { Actions {
                changeVar("broken", 1) setColor("dc2626") setEnable("h2oLamp", true) sound("buzz") shake(4, 0.3)
                setText("alert", "RECYCLER FAULT: tap it 3 times") } } } }
        }
    }

    Entity { id("o2Lamp") startPosition(120, 1320) color("ef4444") Type { circle(18) }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnElapsedTime(0.5) { repeatCount(-1) Actions { flash("ffffff", 0.2) } } } }
    Entity { id("pwrLamp") startPosition(430, 1320) color("ef4444") Type { circle(18) }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnElapsedTime(0.5) { repeatCount(-1) Actions { flash("ffffff", 0.2) } } } }
    Entity { id("h2oLamp") startPosition(740, 1320) color("ef4444") Type { circle(18) }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnElapsedTime(0.5) { repeatCount(-1) Actions { flash("ffffff", 0.2) } } } }

    Entity { id("faults") startPosition(0, 1590) color("00000000") Type { circle(1) }
        Events { Sequence {
            Phase(3, 15) { Actions { randomAction {
                Actions { broadcast("breakO2") } Actions { broadcast("breakPower") } Actions { broadcast("breakWater") } } } }
            Phase(-1, 10) { Actions { randomAction {
                Actions { broadcast("breakO2") } Actions { broadcast("breakPower") } Actions { broadcast("breakWater") } } } }
        } } }

    Entity { id("shuttle") startPosition(800, 1000) color("f8fafc") Type { triangle(70, "left") }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("shuttleSighted") { Actions { setEnable(true) setScale(0.4) burst(10, "ffffff") setPosition("shuttle", 610, 960, 22) } } } }

    Entity {
        id("debris")
        template(true)
        color("a8a29e")
        Type { polygon(5, 22) }
        Movement { speed(0.8) chase("hull") }
        Events {
            OnTapped { Actions { sound("pop") burst(10, "d6d3d1") deleteEntity() } }
            OnCollision("hull") { Actions {
                sound("thud") shake(7, 0.4) burst(14, "f97316") flash("hull", "ef4444", 0.3)
                randomAction {
                    Actions { broadcast("breakO2") } Actions { broadcast("breakPower") } Actions { broadcast("breakWater") } }
                deleteEntity() } }
        }
    }
}
