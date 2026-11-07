Window {
    size(7, 12)
    background("0a0e27")
    music("bg_1")
    gravity(0)

    Bar("win") {
        initial(0)
        OnValue(1) { Actions {
            sound("win")
            shake(10, 0.8)
            burst("boardBg", 50, "fbbf24")
            gameOver("-- 128 REACHED --")
        } }
    }

    Entity { id("boardBg") startPosition(70, 190) color("1e293b")
        Type { rectangle(540, 540) } }

    Entity { id("wallL") startPosition(60, 190) color("0a0e27")
        Type { rectangle(20, 540) } solid(true) }
    Entity { id("wallR") startPosition(610, 190) color("0a0e27")
        Type { rectangle(20, 540) } solid(true) }
    Entity { id("wallT") startPosition(60, 720) color("0a0e27")
        Type { rectangle(570, 20) } solid(true) }
    Entity { id("wallB") startPosition(60, 180) color("0a0e27")
        Type { rectangle(570, 20) } solid(true) }

    Entity { id("tile") template(true) startPosition(0, 0) color("fbbf24")
        Type { rectangle(120, 120) }
        var("value", 2)
        Events {
            OnSignal("recolor") {
                Actions {
                    sound("bubble")
                    whenVar("value", "==", 2) { Actions {
                        setColor("fbbf24") } }
                    whenVar("value", "==", 4) { Actions {
                        setColor("f97316") } }
                    whenVar("value", "==", 8) { Actions {
                        setColor("ef4444") } }
                    whenVar("value", "==", 16) { Actions {
                        setColor("a855f7") } }
                    whenVar("value", "==", 32) { Actions {
                        setColor("6366f1") } }
                    whenVar("value", "==", 64) { Actions {
                        setColor("06b6d4") } }
                    whenVar("value", "==", 128) { Actions {
                        setColor("10b981")
                        updateBar("win", 1) } }
                }
            }
        }
    }

    Entity { id("starter") startPosition(20, 20) color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.05) { repeatCount(1) Actions {
                spawnAtEmpty("tile", 130, 80, 200, 4, 4)
                spawnAtEmpty("tile", 130, 80, 200, 4, 4)
            } }
        } }

    Entity { id("swipeHandler") startPosition(40, 40) color("00000000")
        Type { circle(1) }
        Events {
            OnSwipe("left") { Actions {
                sound("blip")
                slideAll(-1, 0, 130, "tile", "value", "recolor")
                spawnAtEmpty("tile", 130, 80, 200, 4, 4)
            } }
            OnSwipe("right") { Actions {
                sound("blip")
                slideAll(1, 0, 130, "tile", "value", "recolor")
                spawnAtEmpty("tile", 130, 80, 200, 4, 4)
            } }
            OnSwipe("up") { Actions {
                sound("blip")
                slideAll(0, 1, 130, "tile", "value", "recolor")
                spawnAtEmpty("tile", 130, 80, 200, 4, 4)
            } }
            OnSwipe("down") { Actions {
                sound("blip")
                slideAll(0, -1, 130, "tile", "value", "recolor")
                spawnAtEmpty("tile", 130, 80, 200, 4, 4)
            } }
        } }
}
