Window {
    size(8, 12)
    background("fef3c7")
    music("bg_nature")
    winAt("honey", 8, "THE HIVE IS FULL", "win2")
    Bar("honey") { initial(0) }
    Bar("storm") {
        initial(0)
        OnValue(10) { Actions { sound("lose") gameOver("THE STORM SCATTERED THE BEES") } }
    }

    Entity {
        id("hive")
        startPosition(320, 150)
        color("d97706")
        Type { polygon(6, 80) }
    }
    Entity {
        id("flowerA")
        startPosition(100, 900)
        color("ec4899")
        Type { star(6, 45, 20) }
        Events {
            OnTapped("flowerA") { Actions { setScale("flowerA", 1.4) delay(0.4) { Actions { setScale("flowerA", 1) } } } }
        }
    }
    Entity {
        id("flowerB")
        startPosition(620, 800)
        color("8b5cf6")
        Type { star(5, 45, 20) }
        Events {
            OnTapped("flowerB") { Actions { setScale("flowerB", 1.4) delay(0.4) { Actions { setScale("flowerB", 1) } } } }
        }
    }
    Entity {
        id("beeOne")
        startPosition(300, 500)
        color("facc15")
        Type { circle(22) }
        Movement { speed(3) chase("flowerA") }
        Events {
            OnCollision("flowerA") { Actions { retarget("hive") sound("click") } }
            OnCollision("hive") {
                Actions {
                    updateBar("honey", 1)
                    retarget("flowerA")
                    sound("coin")
                    burst(6, "fbbf24")
                }
            }
            OnSignal("goB") { Actions { retarget("flowerB") } }
        }
    }
    Entity {
        id("beeTwo")
        startPosition(450, 500)
        color("fde047")
        Type { circle(22) }
        Movement { speed(2.5) chase("flowerB") }
        Events {
            OnCollision("flowerB") { Actions { retarget("hive") sound("click") } }
            OnCollision("hive") {
                Actions {
                    updateBar("honey", 1)
                    retarget("flowerB")
                    sound("coin")
                    burst(6, "fbbf24")
                }
            }
        }
    }
    Entity {
        id("cloud")
        startPosition(0, 1050)
        color("64748b")
        Type { rectangle(240, 90) }
        Movement { speed(1.5) AutoMove { direction(1, 0) } }
        Events {
            OnCollision { Actions { setPosition("cloud", 0, 1050, 0) } }
            OnElapsedTime(3) {
                repeatCount(-1)
                Actions { updateBar("storm", 1) }
            }
            OnTapped("cloud") {
                Actions {
                    updateBar("storm", -2)
                    setColor("94a3b8")
                    delay(0.5) { Actions { setColor("64748b") } }
                    broadcast("goB")
                }
            }
        }
    }
}
