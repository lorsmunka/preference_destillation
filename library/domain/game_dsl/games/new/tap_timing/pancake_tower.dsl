# Pancake Tower: the spatula slides back and forth above the plate. Tap anywhere to
# drop its pancake; it lands on the plate or on the top of the stack. Blue ones count double.
Window {
    size(7, 12)
    background("fff7ed")
    music("bg_5")

    lives(3, "-- PANCAKES ALL OVER THE FLOOR --", "lose")
    Bar("height") {
        initial(0)
        OnValue(5) { Actions { broadcast("faster") setText("chef", "Faster!") } }
        OnValue(10) { Actions { broadcast("faster") setText("chef", "Even faster!") } }
    }
    winAt("height", 16, "-- A TOWER OF PANCAKES --", "win")
    Bar("holding") { initial(1) }

    Entity { id("kitchen") startPosition(0, 0) color("00000000") Type { rectangle(700, 1200) }
        Events {
            OnTapped("kitchen") { Actions { whenBar("holding", "==", 1) { Actions {
                updateBar("holding", -1)
                broadcast("drop")
            } } } }
        } }

    Entity { id("table") startPosition(0, 0) color("78350f") Type { rectangle(700, 40) } }
    Entity { id("plate") startPosition(250, 40) color("e5e7eb") Type { rectangle(200, 16) } }
    Entity { id("chef") startPosition(30, 1140) color("9a3412") Type { text("Stack 16 pancakes", 36) } }

    Entity {
        id("spatula")
        startPosition(40, 1050)
        color("f59e0b")
        Type { rectangle(120, 24) }
        var("kind", 0)
        Movement { speed(2.5) Path { waypoint(40, 1050) waypoint(540, 1050) } }
        Events {
            OnSignal("drop") {
                Actions {
                    sound("blip")
                    whenVar("kind", "==", 0) { Actions { spawnHere("batter") } }
                    whenVar("kind", "==", 1) { Actions { spawnHere("berry") changeVar("kind", -1) } }
                    setVisible(false)
                    delay(0.9) {
                        Actions {
                            setVisible(true)
                            randomAction {
                                Actions { setColor("f59e0b") }
                                Actions { setColor("f59e0b") }
                                Actions { setColor("f59e0b") }
                                Actions { setColor("6366f1") changeVar("kind", 1) }
                            }
                            updateBar("holding", 1)
                        }
                    }
                }
            }
            OnSignal("faster") {
                Actions {
                    whenBar("height", "==", 5) { Actions { updateSpeed(3.5) } }
                    whenBar("height", "==", 10) { Actions { updateSpeed(4.5) } }
                }
            }
        }
    }

    Entity { id("batter") template(true) startPosition(0, 0) color("f59e0b") Type { rectangle(120, 24) }
        Movement { speed(6) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("plate") { Actions { sound("thud") burst(6, "fde68a") spawnHere("stack") updateBar("height", 1) deleteEntity() } }
            OnCollision("stack") { Actions { sound("thud") burst(6, "fde68a") spawnHere("stack") updateBar("height", 1) deleteEntity() } }
            OnCollision("table") { Actions { sound("buzz") shake(5, 0.3) burst(12, "f59e0b") updateBar("lives", -1) deleteEntity() } }
        } }

    Entity { id("berry") template(true) startPosition(0, 0) color("6366f1") Type { rectangle(120, 24) }
        Movement { speed(6) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("plate") { Actions { sound("coin") burst(10, "a5b4fc") spawnHere("stack") deleteEntity() loopCount(2) updateBar("height", 1) } }
            OnCollision("stack") { Actions { sound("coin") burst(10, "a5b4fc") spawnHere("stack") deleteEntity() loopCount(2) updateBar("height", 1) } }
            OnCollision("table") { Actions { sound("buzz") shake(5, 0.3) burst(12, "6366f1") updateBar("lives", -1) deleteEntity() } }
        } }

    Entity { id("stack") template(true) startPosition(0, 0) color("d97706") Type { rectangle(120, 24) } }
}
