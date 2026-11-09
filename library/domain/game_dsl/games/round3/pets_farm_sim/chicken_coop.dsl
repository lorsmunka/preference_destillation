Window {
    size(8, 12)
    background("a3e635")
    music("bg_1")
    lives(3, "THE FOX GOT THE HENS", "lose")
    winAt("eggs", 8, "EGG BASKET FULL", "win")
    Bar("eggs") { initial(0) }

    Entity {
        id("egg")
        template(true)
        startPosition(0, 0)
        color("fef9c3")
        Type { circle(22) }
        Events {
            OnTapped {
                Actions {
                    updateBar("eggs", 1)
                    sound("pickup")
                    burst(6, "fef08a")
                    deleteEntity()
                }
            }
            OnElapsedTime(6) { Actions { deleteEntity() } }
        }
    }
    Entity {
        id("henRed")
        startPosition(120, 600)
        color("dc2626")
        Type { circle(45) }
        Events {
            OnElapsedTime(4) {
                repeatCount(-1)
                Actions { spawnHere("egg") sound("blip") }
            }
        }
    }
    Entity {
        id("henWhite")
        startPosition(560, 400)
        color("fafafa")
        Type { circle(45) }
        Events {
            OnElapsedTime(6) {
                repeatCount(-1)
                Actions {
                    spawnHere("egg")
                    setScale("henWhite", 1.2)
                    delay(0.3) { Actions { setScale("henWhite", 1) } }
                }
            }
        }
    }
    Entity {
        id("fox")
        startPosition(700, 1100)
        color("ea580c")
        Type { triangle(70, "left") }
        Movement { speed(1.2) chase("henRed", 0.3) }
        Events {
            OnCollision("henRed") {
                Actions {
                    updateBar("lives", -1)
                    sound("hurt")
                    shake(6, 0.3)
                    setPosition("fox", 700, 1100, 0)
                }
            }
            OnTapped("fox") {
                Actions {
                    setPosition("fox", 700, 1100, 0)
                    sound("thud")
                    flash("fox", "ffffff", 0.2)
                }
            }
            OnElapsedTime(15) { Actions { retarget("henWhite") } }
        }
    }
    Entity {
        id("hint")
        startPosition(80, 60)
        color("365314")
        Type { text("Tap eggs, tap the fox away", 36) }
    }
}
