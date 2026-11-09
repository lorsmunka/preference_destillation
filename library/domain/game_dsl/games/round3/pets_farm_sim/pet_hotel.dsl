Window {
    size(8, 10)
    background("e0f2fe")
    music("bg_ambient")
    lives(3, "THE GUESTS CHECKED OUT", "lose")
    Bar("fed") { initial(0) }

    Level {
        Entity {
            id("dog")
            startPosition(300, 450)
            color("a16207")
            Type { rectangle(160, 100) }
            Events {
                OnSignal("bowlFilled") {
                    Actions { setScale("dog", 1.2) delay(0.3) { Actions { setScale("dog", 1) } } }
                }
            }
        }
        Entity {
            id("bowl")
            startPosition(330, 200)
            color("64748b")
            Type { rectangle(100, 40) }
            Events {
                OnTapped("bowl") {
                    Actions {
                        updateBar("fed", 1)
                        broadcast("bowlFilled")
                        sound("pop")
                        whenBar("fed", ">=", 3) { Actions { sound("win") nextLevel() } }
                    }
                }
            }
        }
        Entity {
            id("grumpTimer")
            startPosition(0, 0)
            color("00000000")
            Type { circle(1) }
            Events {
                OnElapsedTime(7) {
                    repeatCount(-1)
                    Actions { updateBar("lives", -1) setColor("ef4444") shake(4, 0.3) }
                }
            }
        }
        Entity {
            id("level1Text")
            startPosition(80, 850)
            color("075985")
            Type { text("Room 1: feed the dog 3 times", 36) }
        }
    }
    Level {
        Entity {
            id("cat")
            startPosition(300, 450)
            color("f97316")
            Type { triangle(120, "up") }
            Events {
                OnSignal("bowlFilled") { Actions { setColor("fdba74") } }
                OnTapped("cat") { Actions { updateBar("fed", 1) burst(5, "fb923c") sound("blip") } }
            }
        }
        Entity {
            id("bowl")
            startPosition(330, 200)
            color("64748b")
            Type { rectangle(100, 40) }
            Events {
                OnTapped("bowl") {
                    Actions {
                        broadcast("bowlFilled")
                        updateBar("fed", 1)
                        sound("pop")
                        whenBar("fed", ">=", 8) { Actions { gameOver("A FIVE STAR HOTEL") } }
                    }
                }
            }
        }
        Entity {
            id("mouse")
            startPosition(40, 700)
            color("94a3b8")
            Type { circle(25) }
            Movement { speed(3) AutoMove { direction(1, 0) } }
            Events {
                OnCollision { Actions { bounceBack() } }
                OnCollision("cat") { Actions { updateBar("fed", 1) deleteEntity() sound("pickup") } }
            }
        }
        Entity {
            id("level2Text")
            startPosition(80, 850)
            color("075985")
            Type { text("Room 2: pet the cat and feed it", 36) }
        }
    }
}
