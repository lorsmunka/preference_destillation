Window {
    size(8, 8)
    background("1c1917")
    music("bg_3")
    Entity {
        id("count")
        startPosition(340, 600)
        color("ef4444")
        Type { text("5", 120) }
        Events {
            OnElapsedTime(1) { Actions { setText("count", "4") sound("beep") } }
            OnElapsedTime(2) { Actions { setText("count", "3") sound("beep") } }
            OnElapsedTime(3) { Actions { setText("count", "2") sound("beep") } }
            OnElapsedTime(4) { Actions { setText("count", "1") sound("beep") } }
            OnElapsedTime(5) { Actions { sound("death") shake(10, 0.6) gameOver("BOOM") } }
        }
    }
    Entity {
        id("hint")
        startPosition(120, 480)
        color("e7e5e4")
        Type { text("Cut the blue wire", 44) }
    }
    Entity {
        id("redWire")
        startPosition(100, 200)
        color("dc2626")
        Type { rectangle(120, 200) }
        Events {
            OnTapped("redWire") { Actions { sound("death") gameOver("WRONG WIRE") } }
        }
    }
    Entity {
        id("blueWire")
        startPosition(340, 200)
        color("2563eb")
        Type { rectangle(120, 200) }
        Events {
            OnTapped("blueWire") { Actions { sound("win") burst(12, "2563eb") gameOver("DEFUSED") } }
        }
    }
    Entity {
        id("greenWire")
        startPosition(580, 200)
        color("16a34a")
        Type { rectangle(120, 200) }
        Events {
            OnTapped("greenWire") { Actions { sound("death") gameOver("WRONG WIRE") } }
        }
    }
}
