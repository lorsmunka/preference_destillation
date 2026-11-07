# Crossroads: you run the traffic lights of one junction. Swipe anywhere to switch which road has green;
# both lights stay red for a short moment first. Cars queue at a red light, and switching while a car
# is still crossing causes a crash. Too many cars waiting at once is gridlock.
Window {
    size(10, 10)
    background("1c2a1c")
    music("bg_5")
    gravity(0)

    lives(3, "-- TOO MANY CRASHES --", "lose")
    Bar("flow") { initial(0) }
    winAt("flow", 30, "-- RUSH HOUR CLEARED --", "win")
    Bar("jam") {
        initial(0)
        OnValue(6) { Actions { broadcast("honk") } }
        OnValue(9) { Actions { sound("lose") shake(6, 0.5) gameOver("-- GRIDLOCK --") } }
    }
    Bar("light") {
        initial(0)
        OnValue(0) { Actions { broadcast("ewGo") } }
        OnValue(1) { Actions { broadcast("allRed") } }
        OnValue(2) { Actions { broadcast("nsGo") } }
        OnValue(3) { Actions { broadcast("allRed") } }
        OnValue(4) { Actions { updateBar("light", -4) } }
    }

    Entity { id("roadEW") startPosition(0, 440) color("3f3f46") Type { rectangle(1000, 120) } }
    Entity { id("roadNS") startPosition(440, 0) color("3f3f46") Type { rectangle(120, 1000) } }
    Entity { id("lineW") startPosition(400, 440) color("e5e7eb") Type { rectangle(10, 120) } }
    Entity { id("lineS") startPosition(440, 400) color("e5e7eb") Type { rectangle(120, 10) } }

    Entity {
        id("lampEW")
        startPosition(370, 590)
        color("22c55e")
        Type { circle(18) }
        Events {
            OnSignal("ewGo") { Actions { setColor("22c55e") } }
            OnSignal("allRed") { Actions { setColor("f59e0b") } }
            OnSignal("nsGo") { Actions { setColor("ef4444") } }
        }
    }
    Entity {
        id("lampNS")
        startPosition(590, 370)
        color("ef4444")
        Type { circle(18) }
        Events {
            OnSignal("nsGo") { Actions { setColor("22c55e") } }
            OnSignal("allRed") { Actions { setColor("f59e0b") } }
            OnSignal("ewGo") { Actions { setColor("ef4444") } }
        }
    }

    Entity {
        id("controller")
        startPosition(700, 700)
        color("94a3b8")
        Type { rectangle(200, 150) }
        Events {
            OnSwipe {
                Actions {
                    whenBar("light", "==", 0) { Actions { updateBar("light", 1) sound("click") delay(0.6) { Actions { updateBar("light", 1) } } } }
                    whenBar("light", "==", 2) { Actions { updateBar("light", 1) sound("click") delay(0.6) { Actions { updateBar("light", 1) } } } }
                }
            }
            OnSignal("honk") { Actions { sound("buzz") flash("ef4444", 0.4) shake(3, 0.3) } }
        }
    }

    Entity {
        id("westGate")
        startPosition(0, 600)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(5, 3) { Actions { spawn("carE", 10, 472) } }
                Phase(10, 2.2) { Actions { spawn("carE", 10, 472) } }
                Phase(-1, 1.7) { Actions { randomAction { Actions { spawn("carE", 10, 472) } Actions { spawn("truckE", 10, 466) } } } }
            }
        }
    }
    Entity {
        id("southGate")
        startPosition(600, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(4, 3.6) { Actions { spawn("carN", 482, 10) } }
                Phase(10, 2.6) { Actions { spawn("carN", 482, 10) } }
                Phase(-1, 1.9) { Actions { spawn("carN", 482, 10) } }
            }
        }
    }

    Entity {
        id("carE")
        template(true)
        color("38bdf8")
        Type { rectangle(70, 36) }
        var("waiting", 0)
        Movement { speed(2.2) AutoMove { direction(1, 0) } }
        Events {
            OnCollision("lineW") { Actions { whenBar("light", "!=", 0) { Actions {
                updateSpeed(0) changeVar("waiting", 1) updateBar("jam", 1) } } } }
            OnCollision("carE") { Actions { whenVar("waiting", "==", 0) { Actions {
                updateSpeed(0) changeVar("waiting", 1) updateBar("jam", 1) } } } }
            OnCollision("truckE") { Actions { whenVar("waiting", "==", 0) { Actions {
                updateSpeed(0) changeVar("waiting", 1) updateBar("jam", 1) } } } }
            OnSignal("ewGo") { Actions { whenVar("waiting", "==", 1) { Actions {
                updateSpeed(2.2) changeVar("waiting", -1) updateBar("jam", -1) } } } }
            OnCollision("carN") { Actions {
                sound("death") shake(9, 0.5) burst(24, "f97316") updateBar("lives", -1)
                deleteCollider() deleteEntity() } }
            OnCollision { Actions { sound("blip") updateBar("flow", 1) deleteEntity() } }
        }
    }

    Entity {
        id("truckE")
        template(true)
        color("eab308")
        Type { rectangle(110, 48) }
        var("waiting", 0)
        Movement { speed(2.2) AutoMove { direction(1, 0) } }
        Events {
            OnCollision("lineW") { Actions { whenBar("light", "!=", 0) { Actions {
                updateSpeed(0) changeVar("waiting", 1) updateBar("jam", 1) } } } }
            OnCollision("carE") { Actions { whenVar("waiting", "==", 0) { Actions {
                updateSpeed(0) changeVar("waiting", 1) updateBar("jam", 1) } } } }
            OnCollision("truckE") { Actions { whenVar("waiting", "==", 0) { Actions {
                updateSpeed(0) changeVar("waiting", 1) updateBar("jam", 1) } } } }
            OnSignal("ewGo") { Actions { whenVar("waiting", "==", 1) { Actions {
                updateSpeed(2.2) changeVar("waiting", -1) updateBar("jam", -1) } } } }
            OnCollision("carN") { Actions {
                sound("death") shake(9, 0.5) burst(30, "f97316") updateBar("lives", -1)
                deleteCollider() deleteEntity() } }
            OnCollision { Actions { sound("blip") updateBar("flow", 1) updateBar("flow", 1) deleteEntity() } }
        }
    }

    Entity {
        id("carN")
        template(true)
        color("f472b6")
        Type { rectangle(36, 70) }
        var("waiting", 0)
        Movement { speed(2.2) AutoMove { direction(0, 1) } }
        Events {
            OnCollision("lineS") { Actions { whenBar("light", "!=", 2) { Actions {
                updateSpeed(0) changeVar("waiting", 1) updateBar("jam", 1) } } } }
            OnCollision("carN") { Actions { whenVar("waiting", "==", 0) { Actions {
                updateSpeed(0) changeVar("waiting", 1) updateBar("jam", 1) } } } }
            OnSignal("nsGo") { Actions { whenVar("waiting", "==", 1) { Actions {
                updateSpeed(2.2) changeVar("waiting", -1) updateBar("jam", -1) } } } }
            OnCollision { Actions { sound("blip") updateBar("flow", 1) deleteEntity() } }
        }
    }
}
