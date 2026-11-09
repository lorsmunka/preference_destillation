# The screen shows an arrow word. Swipe that way before the next beat.
Window {
    size(7, 11)
    background("2e1065")
    music("bg_6")
    lives(3, "OUT OF STEP", "lose")
    winAt("score", 9, "DANCE FLOOR CHAMPION", "win")
    Bar("up") { initial(0) }
    Bar("down") { initial(0) }
    Bar("left") { initial(0) }
    Bar("right") { initial(0) }

    Entity {
        id("prompt")
        startPosition(180, 620)
        color("fde047")
        Type { text("Get ready", 70) }
    }
    Entity {
        id("hint")
        startPosition(150, 960)
        color("c4b5fd")
        Type { text("Swipe anywhere", 36) }
    }

    Entity {
        id("stage")
        startPosition(0, 0)
        color("00000000")
        Type { rectangle(700, 1100) }
        Events {
            OnSwipe("up") { Actions {
                whenBar("up", "==", 0) { Actions { updateBar("lives", -1) sound("buzz") } }
                whenBar("up", "==", 1) { Actions { updateBar("up", -1) updateBar("score", 1) sound("jump") } }
            } }
            OnSwipe("down") { Actions {
                whenBar("down", "==", 0) { Actions { updateBar("lives", -1) sound("buzz") } }
                whenBar("down", "==", 1) { Actions { updateBar("down", -1) updateBar("score", 1) sound("pop") } }
            } }
            OnSwipe("left") { Actions {
                whenBar("left", "==", 0) { Actions { updateBar("lives", -1) sound("buzz") } }
                whenBar("left", "==", 1) { Actions { updateBar("left", -1) updateBar("score", 1) sound("blip") } }
            } }
            OnSwipe("right") { Actions {
                whenBar("right", "==", 0) { Actions { updateBar("lives", -1) sound("buzz") } }
                whenBar("right", "==", 1) { Actions { updateBar("right", -1) updateBar("score", 1) sound("bubble") } }
            } }
        }
    }

    Entity {
        id("caller")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(1, 2.0) { Actions {
                    setText("prompt", "UP")
                    updateBar("up", 1)
                    delay(1.3) { Actions { whenBar("up", "==", 1) { Actions { updateBar("up", -1) updateBar("lives", -1) sound("hurt") } } } }
                } }
                Phase(1, 1.6) { Actions {
                    setText("prompt", "LEFT")
                    updateBar("left", 1)
                    delay(1.2) { Actions { whenBar("left", "==", 1) { Actions { updateBar("left", -1) updateBar("lives", -1) sound("hurt") } } } }
                } }
                Phase(1, 1.6) { Actions {
                    setText("prompt", "RIGHT")
                    updateBar("right", 1)
                    delay(1.2) { Actions { whenBar("right", "==", 1) { Actions { updateBar("right", -1) updateBar("lives", -1) sound("hurt") } } } }
                } }
                Phase(1, 1.6) { Actions {
                    setText("prompt", "DOWN")
                    updateBar("down", 1)
                    delay(1.2) { Actions { whenBar("down", "==", 1) { Actions { updateBar("down", -1) updateBar("lives", -1) sound("hurt") } } } }
                } }
                Phase(1, 1.4) { Actions {
                    setText("prompt", "UP")
                    updateBar("up", 1)
                    delay(1.0) { Actions { whenBar("up", "==", 1) { Actions { updateBar("up", -1) updateBar("lives", -1) sound("hurt") } } } }
                } }
                Phase(1, 1.4) { Actions {
                    setText("prompt", "RIGHT")
                    updateBar("right", 1)
                    delay(1.0) { Actions { whenBar("right", "==", 1) { Actions { updateBar("right", -1) updateBar("lives", -1) sound("hurt") } } } }
                } }
                Phase(1, 1.4) { Actions {
                    setText("prompt", "LEFT")
                    updateBar("left", 1)
                    delay(1.0) { Actions { whenBar("left", "==", 1) { Actions { updateBar("left", -1) updateBar("lives", -1) sound("hurt") } } } }
                } }
                Phase(1, 1.2) { Actions {
                    setText("prompt", "DOWN")
                    updateBar("down", 1)
                    delay(0.9) { Actions { whenBar("down", "==", 1) { Actions { updateBar("down", -1) updateBar("lives", -1) sound("hurt") } } } }
                } }
                Phase(1, 1.2) { Actions {
                    setText("prompt", "LEFT")
                    updateBar("left", 1)
                    delay(0.9) { Actions { whenBar("left", "==", 1) { Actions { updateBar("left", -1) updateBar("lives", -1) sound("hurt") } } } }
                } }
                Phase(1, 1.2) { Actions {
                    setText("prompt", "UP")
                    updateBar("up", 1)
                    delay(0.9) { Actions { whenBar("up", "==", 1) { Actions { updateBar("up", -1) updateBar("lives", -1) sound("hurt") } } } }
                } }
                Phase(1, 1.2) { Actions {
                    setText("prompt", "RIGHT")
                    updateBar("right", 1)
                    delay(0.9) { Actions { whenBar("right", "==", 1) { Actions { updateBar("right", -1) updateBar("lives", -1) sound("hurt") } } } }
                } }
                Phase(1, 1.2) { Actions {
                    setText("prompt", "DOWN")
                    updateBar("down", 1)
                    delay(0.9) { Actions { whenBar("down", "==", 1) { Actions { updateBar("down", -1) updateBar("lives", -1) sound("hurt") } } } }
                } }
            }
        }
    }
}
