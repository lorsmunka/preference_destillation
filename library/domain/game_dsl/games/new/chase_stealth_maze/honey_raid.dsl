Window {
    size(8, 10)
    background("3f6212")
    music("bg_2")
    gravity(0)

    lives(3, "-- STUNG TOO OFTEN --", "lose")
    Bar("honey") {
        initial(0)
        OnValue(4) { Actions { sound("win") burst("bear", 30, "fbbf24") gameOver("-- A WINTER'S WORTH OF HONEY --") } }
    }

    Entity {
        id("bear")
        startPosition(80, 60)
        color("92400e")
        Type { circle(28) }
        Movement { speed(3.6) Control { movementControl(0) } }
        Events {
            OnCollision("pond") { Actions { sound("bubble") burst(14, "7dd3fc") broadcast("splash") } }
        }
    }

    Entity { id("hive") startPosition(360, 860) color("a16207") Type { polygon(6, 40) } }
    Entity { id("pond") startPosition(300, 110) color("0ea5e9") Type { rectangle(220, 120) } }

    # The bees idle on the hive. A stolen comb sends them after the bear;
    # a splash in the pond (or a sting) sends them home.
    Entity { id("bee1") startPosition(370, 880) color("facc15") Type { circle(10) }
        Movement { speed(2.6) chase("hive") }
        Events {
            OnSignal("stolen") { Actions { retarget("bear") } }
            OnSignal("splash") { Actions { retarget("hive") } }
            OnCollision("bear") { Actions { sound("hurt") updateBar("lives", -1) flash("bear", "ff0000", 0.3) shake(5, 0.3) retarget("hive") } } } }
    Entity { id("bee2") startPosition(400, 900) color("fde047") Type { circle(10) }
        Movement { speed(2.9) chase("hive") }
        Events {
            OnSignal("stolen") { Actions { retarget("bear") } }
            OnSignal("splash") { Actions { retarget("hive") } }
            OnCollision("bear") { Actions { sound("hurt") updateBar("lives", -1) flash("bear", "ff0000", 0.3) shake(5, 0.3) retarget("hive") } } } }
    Entity { id("bee3") startPosition(420, 870) color("eab308") Type { circle(10) }
        Movement { speed(3.2) chase("hive") }
        Events {
            OnSignal("stolen") { Actions { retarget("bear") } }
            OnSignal("splash") { Actions { retarget("hive") } }
            OnCollision("bear") { Actions { sound("hurt") updateBar("lives", -1) flash("bear", "ff0000", 0.3) shake(5, 0.3) retarget("hive") } } } }

    Entity { id("tree1") startPosition(220, 620) color("14532d") solid(true) Type { rectangle(80, 80) } }
    Entity { id("tree2") startPosition(520, 620) color("14532d") solid(true) Type { rectangle(80, 80) } }
    Entity { id("tree3") startPosition(600, 180) color("14532d") solid(true) Type { rectangle(80, 80) } }

    Entity { id("comb1") startPosition(60, 760) color("f59e0b") Type { polygon(6, 18) }
        Events { OnCollision("bear") { Actions { sound("pickup") updateBar("honey", 1) burst(10, "f59e0b") broadcast("stolen") deleteEntity() } } } }
    Entity { id("comb2") startPosition(700, 780) color("f59e0b") Type { polygon(6, 18) }
        Events { OnCollision("bear") { Actions { sound("pickup") updateBar("honey", 1) burst(10, "f59e0b") broadcast("stolen") deleteEntity() } } } }
    Entity { id("comb3") startPosition(700, 320) color("f59e0b") Type { polygon(6, 18) }
        Events { OnCollision("bear") { Actions { sound("pickup") updateBar("honey", 1) burst(10, "f59e0b") broadcast("stolen") deleteEntity() } } } }
    Entity { id("comb4") startPosition(100, 400) color("f59e0b") Type { polygon(6, 18) }
        Events { OnCollision("bear") { Actions { sound("pickup") updateBar("honey", 1) burst(10, "f59e0b") broadcast("stolen") deleteEntity() } } } }
}
