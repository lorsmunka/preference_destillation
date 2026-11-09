Window {
    size(8, 8)
    background("111827")
    winAt(5, "PERFECT GOLD", "win")
    Entity {
        id("square") startPosition(300, 480) color("6b7280") Type { rectangle(200, 200) }
        Events {
            OnSignal("goRed") { Actions { setColor("ef4444") } }
            OnSignal("goBlue") { Actions { setColor("3b82f6") } }
            OnSignal("goGold") { Actions { setColor("facc15") } }
        }
    }
    Entity { id("hint") startPosition(180, 720) color("e5e7eb") Type { text("Tap gold 5 times", 40) } }
    Entity {
        id("redBtn") startPosition(80, 200) color("ef4444") Type { rectangle(160, 120) }
        Events { OnTapped("redBtn") { Actions { sound("buzz") broadcast("goRed") updateBar(-1) } } }
    }
    Entity {
        id("blueBtn") startPosition(320, 200) color("3b82f6") Type { rectangle(160, 120) }
        Events { OnTapped("blueBtn") { Actions { sound("buzz") broadcast("goBlue") updateBar(-1) } } }
    }
    Entity {
        id("goldBtn") startPosition(560, 200) color("facc15") Type { rectangle(160, 120) }
        Events { OnTapped("goldBtn") { Actions { sound("coin") broadcast("goGold") updateBar(1) } } }
    }
    Entity {
        id("timer") startPosition(0, 0) color("00000000") Type { circle(1) }
        Events { OnElapsedTime(30) { Actions { gameOver("OUT OF TIME") } } }
    }
}
