Window {
    size(7, 11)
    background("e0f2fe")
    Bar { initial(0) OnValue(15) { Actions { sound("win") gameOver("-- ALL POPPED --") } } }

    Entity {
        id("balloon")
        template(true)
        startPosition(0, 0)
        color("ef4444")
        Type { circle(45) }
        Movement { speed(1.8) AutoMove { direction(0, 1) } }
        Events {
            OnTapped { Actions { sound("pop") updateBar(1) burst(10, "ef4444") deleteEntity() } }
            OnCollision("spikes") { Actions { sound("lose") gameOver("-- A BALLOON HIT THE SPIKES --") } }
        }
    }

    Entity {
        id("spikes")
        startPosition(0, 1070)
        color("64748b")
        Type { rectangle(700, 30) }
        Events { OnCollision("balloon") { Actions { flash("ef4444", 0.3) } } }
    }

    Entity {
        id("pump")
        startPosition(350, 20)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1.3) { repeatCount(-1) Actions { spawn("balloon", random(20, 600), 20) } }
        }
    }
}
