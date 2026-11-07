# Dam Keeper: rain fills the reservoir. The turbine only spins while the water stands at 5 or more,
# so release it in steady pulses: too low and the gate locks, too high and the dam gives way.
Window {
    size(6, 10)
    background("0c1a2b")
    music("bg_nature")
    gravity(0)

    Bar("reservoir") { initial(5)
        OnValue(4) { Actions { setPosition("surface", 0, 290, 0.3) } }
        OnValue(5) { Actions { setPosition("surface", 0, 330, 0.3) } }
        OnValue(6) { Actions { setPosition("surface", 0, 370, 0.3) } }
        OnValue(7) { Actions { setPosition("surface", 0, 410, 0.3) broadcast("calm") } }
        OnValue(8) { Actions { setPosition("surface", 0, 450, 0.3) broadcast("highWater") } }
        OnValue(9) { Actions { setPosition("surface", 0, 490, 0.3) sound("bell") shake(3, 0.3) } }
        OnValue(10) { Actions { sound("death") shake(9, 0.6) burst("dam", 30, "38bdf8") gameOver("-- THE DAM GAVE WAY --") } } }
    Bar("power") { initial(0) }
    winAt("power", 40, "-- THE VALLEY IS LIT --", "win")

    Wave(20, 1.4) { Actions { spawn("drop", random(20, 570), 980) } }
    Wave(25, 0.9) { Actions { spawn("drop", random(20, 570), 980) } }
    Wave(60, 0.45) { Actions { spawn("drop", random(20, 570), 980)
        randomAction { Actions { } Actions { } Actions { spawn("drop", random(20, 570), 980) } } } }

    Entity { id("lake") startPosition(0, 130) color("1e3a8a") Type { rectangle(600, 420) } }
    Entity { id("surface") startPosition(0, 330) color("7dd3fc") Type { rectangle(600, 14) } }
    Entity { id("dam") startPosition(0, 100) color("78716c") Type { rectangle(600, 30) } }
    Entity { id("turbine") startPosition(460, 20) color("94a3b8") Type { star(6, 34, 14) } }

    Entity { id("siren") startPosition(530, 900) color("334155") Type { circle(24) }
        Events {
            OnSignal("highWater") { Actions { setColor("ef4444") sound("beep") } }
            OnSignal("calm") { Actions { setColor("334155") } }
            OnElapsedTime(0.6) { repeatCount(-1) Actions {
                whenBar("reservoir", ">=", 8) { Actions { flash("fecaca", 0.3) } } } } } }

    Entity { id("gate") startPosition(240, 20) color("f59e0b") Type { rectangle(120, 70) }
        var("cool", 0)
        Events { OnTapped("gate") { Actions {
            whenBar("reservoir", "<", 5) { Actions { sound("buzz") flash("ef4444", 0.2) } }
            whenBar("reservoir", ">=", 5) { Actions { whenVar("cool", "==", 0) { Actions {
                changeVar("cool", 1) updateBar("reservoir", -1) updateBar("power", 1)
                sound("blip") burst(10, "7dd3fc") flash("turbine", "fde047", 0.25)
                delay(0.4) { Actions { changeVar("cool", -1) } } } } } } } } } }

    Entity { id("drop") template(true) color("93c5fd") Type { circle(6) }
        Movement { speed(4) AutoMove { direction(0, -1) } }
        Events { OnCollision("lake") { Actions {
            updateBar("reservoir", 1) burst(4, "bfdbfe") deleteEntity() } } } }
}
