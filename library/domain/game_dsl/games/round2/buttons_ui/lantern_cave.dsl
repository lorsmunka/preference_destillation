# A short choose-a-path story. The bar "scene" says where you are.
Window {
    size(8, 12)
    background("1c1917")
    music("bg_ambient")
    Bar("scene") { initial(0) }

    Entity {
        id("story")
        startPosition(60, 1000)
        color("fde68a")
        Type { text("You wake in a cave. A lantern glows.", 38) }
    }
    Entity {
        id("story2")
        startPosition(60, 930)
        color("fde68a")
        Type { text("Two tunnels lead away.", 38) }
    }
    Entity {
        id("choiceLeft")
        startPosition(100, 560)
        color("fb923c")
        Type { text("> Take the lantern, go left", 42) }
        Events {
            OnTapped("choiceLeft") {
                Actions {
                    whenBar("scene", "==", 2) { Actions {
                        sound("win")
                        setText("story", "The door opens onto daylight!")
                        setText("story2", "You are free.")
                        setVisible("choiceRight", false)
                        gameOver("YOU ESCAPED")
                    } }
                    whenBar("scene", "==", 1) { Actions {
                        sound("bubble")
                        updateBar("scene", 2)
                        setText("story", "The boat glides to a stone door.")
                        setText("story2", "It has a lock shaped like a lantern.")
                        setText("choiceLeft", "> Hold up the lantern")
                        setText("choiceRight", "> Knock loudly")
                    } }
                    whenBar("scene", "==", 0) { Actions {
                        sound("pickup")
                        updateBar("scene", 1)
                        setText("story", "The left tunnel is dry and warm.")
                        setText("story2", "A river blocks the way. A boat waits.")
                        setText("choiceLeft", "> Row across")
                        setText("choiceRight", "> Swim instead")
                    } }
                }
            }
        }
    }
    Entity {
        id("choiceRight")
        startPosition(100, 360)
        color("60a5fa")
        Type { text("> Leave it, go right", 42) }
        Events {
            OnTapped("choiceRight") {
                Actions {
                    whenBar("scene", "==", 0) { Actions {
                        sound("thud")
                        shake(6, 0.4)
                        setText("story", "Darkness. You step into a pit.")
                        gameOver("LOST IN THE DARK")
                    } }
                    whenBar("scene", "==", 1) { Actions {
                        sound("hurt")
                        setText("story", "The current drags you under.")
                        gameOver("SWEPT AWAY")
                    } }
                    whenBar("scene", "==", 2) { Actions {
                        sound("buzz")
                        shake(6, 0.4)
                        setText("story", "Your knocking wakes a stone guardian.")
                        gameOver("CRUSHED")
                    } }
                }
            }
        }
    }
}
