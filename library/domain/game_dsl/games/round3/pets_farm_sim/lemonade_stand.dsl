Window {
    size(8, 12)
    background("fef9c3")
    music("bg_4")
    lives(3, "NOBODY LIKES YOUR LEMONADE", "lose")
    winAt("money", 6, "YOU OPENED A SECOND STAND", "win")
    Bar("money") { initial(0) }
    Bar("sweet") { initial(4) }

    Entity {
        id("customer")
        template(true)
        startPosition(0, 0)
        color("93c5fd")
        Type { circle(45) }
        Movement { speed(0.8) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("stand") {
                Actions {
                    whenBar("sweet", ">=", 3) {
                        Actions {
                            whenBar("sweet", "<=", 6) {
                                Actions {
                                    updateBar("money", 1)
                                    sound("coin")
                                    burst(8, "fde047")
                                    deleteEntity()
                                }
                            }
                        }
                    }
                    whenBar("sweet", "<", 3) {
                        Actions { updateBar("lives", -1) sound("hurt") deleteEntity() }
                    }
                    whenBar("sweet", ">", 6) {
                        Actions { updateBar("lives", -1) sound("hurt") deleteEntity() }
                    }
                }
            }
        }
    }
    Entity {
        id("stand")
        startPosition(200, 380)
        color("f59e0b")
        Type { rectangle(400, 60) }
    }
    Entity {
        id("street")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(3.5) {
                repeatCount(-1)
                Actions { spawn("customer", random(200, 560), 1100) }
            }
        }
    }
    Entity {
        id("sugarButton")
        startPosition(60, 120)
        color("f472b6")
        Type { rectangle(320, 120) }
        Events {
            OnTapped("sugarButton") {
                Actions {
                    updateBar("sweet", 1)
                    setText("sweetText", "Sweeter")
                    sound("blip")
                }
            }
        }
    }
    Entity {
        id("lemonButton")
        startPosition(420, 120)
        color("eab308")
        Type { rectangle(320, 120) }
        Events {
            OnTapped("lemonButton") {
                Actions {
                    updateBar("sweet", -1)
                    setText("sweetText", "More sour")
                    sound("blip")
                }
            }
        }
    }
    Entity {
        id("sugarLabel")
        startPosition(120, 150)
        color("ffffff")
        Type { text("SUGAR", 44) }
    }
    Entity {
        id("lemonLabel")
        startPosition(480, 150)
        color("422006")
        Type { text("LEMON", 44) }
    }
    Entity {
        id("sweetText")
        startPosition(240, 280)
        color("78350f")
        Type { text("Keep it just right", 36) }
    }
}
