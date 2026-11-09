# Tap LOCK when the sliding marker is inside the green zone. Three safes, each one faster.
Window {
    size(8, 8)
    background("1c1917")
    music("bg_5")
    lives(3, "THE ALARM WENT OFF", "lose")
    Bar("inzone") { initial(0) }

    Level {
        Entity { id("title") startPosition(40, 700) color("a8a29e") Type { text("Safe 1: slow dial", 40) } }
        Entity { id("zone") startPosition(400, 300) color("166534") Type { rectangle(120, 180) } }
        Entity { id("gate") startPosition(540, 300) color("00000000") Type { rectangle(10, 180) } }
        Entity {
            id("marker")
            startPosition(20, 340)
            color("fde047")
            Type { rectangle(20, 100) }
            Movement { speed(3) AutoMove { direction(1, 0) } }
            Events {
                OnCollision("zone") { Actions { updateBar("inzone", 1) } }
                OnCollision("gate") { Actions { whenBar("inzone", "==", 1) { Actions { updateBar("inzone", -1) } } } }
                OnCollision { Actions { setPosition("marker", 20, 340, 0) } }
            }
        }
        Entity {
            id("lock")
            startPosition(280, 80)
            color("0ea5e9")
            Type { rectangle(240, 110) }
            Events {
                OnTapped("lock") { Actions {
                    whenBar("inzone", "==", 0) { Actions { updateBar("lives", -1) sound("buzz") shake(5, 0.2) } }
                    whenBar("inzone", "==", 1) { Actions { updateBar("inzone", -1) sound("clang") burst(12, "4ade80") nextLevel() } }
                } }
            }
        }
    }

    Level {
        Entity { id("title") startPosition(40, 700) color("a8a29e") Type { text("Safe 2: backwards", 40) } }
        Entity { id("zone") startPosition(200, 300) color("166534") Type { rectangle(100, 180) } }
        Entity { id("gate") startPosition(180, 300) color("00000000") Type { rectangle(10, 180) } }
        Entity {
            id("marker")
            startPosition(760, 340)
            color("fde047")
            Type { rectangle(20, 100) }
            Movement { speed(4.5) AutoMove { direction(-1, 0) } }
            Events {
                OnCollision("zone") { Actions { updateBar("inzone", 1) } }
                OnCollision("gate") { Actions { whenBar("inzone", "==", 1) { Actions { updateBar("inzone", -1) } } } }
                OnCollision { Actions { setPosition("marker", 760, 340, 0) } }
            }
        }
        Entity {
            id("lock")
            startPosition(280, 80)
            color("0ea5e9")
            Type { rectangle(240, 110) }
            Events {
                OnTapped("lock") { Actions {
                    whenBar("inzone", "==", 0) { Actions { updateBar("lives", -1) sound("buzz") shake(5, 0.2) } }
                    whenBar("inzone", "==", 1) { Actions { updateBar("inzone", -1) sound("clang") burst(12, "4ade80") nextLevel() } }
                } }
            }
        }
    }

    Level {
        Entity { id("title") startPosition(40, 700) color("a8a29e") Type { text("Safe 3: rising dial", 40) } }
        Entity { id("zone") startPosition(300, 450) color("166534") Type { rectangle(200, 70) } }
        Entity { id("gate") startPosition(300, 530) color("00000000") Type { rectangle(200, 10) } }
        Entity {
            id("marker")
            startPosition(360, 120)
            color("fde047")
            Type { rectangle(80, 20) }
            Movement { speed(6) AutoMove { direction(0, 1) } }
            Events {
                OnCollision("zone") { Actions { updateBar("inzone", 1) } }
                OnCollision("gate") { Actions { whenBar("inzone", "==", 1) { Actions { updateBar("inzone", -1) } } } }
                OnCollision { Actions { setPosition("marker", 360, 120, 0) } }
            }
        }
        Entity {
            id("lock")
            startPosition(60, 300)
            color("0ea5e9")
            Type { rectangle(160, 110) }
            Events {
                OnTapped("lock") { Actions {
                    whenBar("inzone", "==", 0) { Actions { updateBar("lives", -1) sound("buzz") shake(5, 0.2) } }
                    whenBar("inzone", "==", 1) { Actions { sound("win") burst(20, "fde047") gameOver("SAFE CRACKED") } }
                } }
            }
        }
    }
}
