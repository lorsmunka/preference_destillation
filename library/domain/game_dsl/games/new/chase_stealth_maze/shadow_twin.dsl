Window {
    size(7, 10)
    background("0f0f1a")
    music("bg_4")
    gravity(0)

    Bar("shards") {
        initial(0)
        OnValue(3) { Actions { broadcast("echo") } }
        OnValue(5) { Actions { sound("win") shake(6, 0.5) gameOver("-- THE MIRROR SHATTERS --") } }
    }

    Entity { id("p1") startPosition(330, 60) color("f8fafc") Type { circle(20) }
        Movement { speed(4.2) Control { movementControl(0) } }
        Events { OnSignal("shard") { Actions { flash("c4b5fd", 0.2) } } } }

    # Your shadow: a little faster with every shard you take.
    Entity { id("shadow") startPosition(330, 900) color("4c1d95") Type { circle(20) }
        hostile("p1", "-- YOUR SHADOW CAUGHT UP --", "death")
        Movement { speed(1.5) chase("p1") }
        Events { OnSignal("shard") { Actions {
            burst(10, "7c3aed")
            whenBar("shards", "==", 1) { Actions { updateSpeed(1.8) } }
            whenBar("shards", "==", 2) { Actions { updateSpeed(2.1) } }
            whenBar("shards", "==", 3) { Actions { updateSpeed(2.4) } }
            whenBar("shards", "==", 4) { Actions { updateSpeed(2.7) } } } } } }

    # A second reflection steps out of the corner after the third shard.
    Entity { id("echo") startPosition(40, 900) color("831843") Type { circle(18) }
        hostile("p1", "-- YOUR ECHO CAUGHT UP --", "death")
        Movement { speed(1.7) chase("p1") }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("echo") { Actions { setEnable(true) sound("buzz") burst(20, "db2777") shake(4, 0.3) } } } }

    Entity { id("mirror1") startPosition(140, 280) color("334155") solid(true) Type { rectangle(90, 90) } }
    Entity { id("mirror2") startPosition(470, 280) color("334155") solid(true) Type { rectangle(90, 90) } }
    Entity { id("mirror3") startPosition(305, 460) color("334155") solid(true) Type { rectangle(90, 90) } }
    Entity { id("mirror4") startPosition(140, 640) color("334155") solid(true) Type { rectangle(90, 90) } }
    Entity { id("mirror5") startPosition(470, 640) color("334155") solid(true) Type { rectangle(90, 90) } }

    Entity { id("shard1") startPosition(60, 120) color("e0f2fe") Type { star(4, 16) }
        Events { OnCollision("p1") { Actions { sound("glass") updateBar("shards", 1) broadcast("shard") deleteEntity() } } } }
    Entity { id("shard2") startPosition(600, 120) color("e0f2fe") Type { star(4, 16) }
        Events { OnCollision("p1") { Actions { sound("glass") updateBar("shards", 1) broadcast("shard") deleteEntity() } } } }
    Entity { id("shard3") startPosition(334, 600) color("e0f2fe") Type { star(4, 16) }
        Events { OnCollision("p1") { Actions { sound("glass") updateBar("shards", 1) broadcast("shard") deleteEntity() } } } }
    Entity { id("shard4") startPosition(40, 480) color("e0f2fe") Type { star(4, 16) }
        Events { OnCollision("p1") { Actions { sound("glass") updateBar("shards", 1) broadcast("shard") deleteEntity() } } } }
    Entity { id("shard5") startPosition(620, 860) color("e0f2fe") Type { star(4, 16) }
        Events { OnCollision("p1") { Actions { sound("glass") updateBar("shards", 1) broadcast("shard") deleteEntity() } } } }
}
