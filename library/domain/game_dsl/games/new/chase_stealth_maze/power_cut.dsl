Window {
    size(8, 12)
    background("111827")
    music("bg_2")
    gravity(0)

    lives(3, "-- SECURITY LOCKDOWN --", "lose")

    # The breaker cuts the power for 2 s out of every 5. Camera cones only
    # see you while the lights are on; the guards see you either way.

    # ---------- floor 1 ----------
    Level {
        wallBox(0, 0, 800, 1200, "374151", 20)

        Entity { id("breaker") startPosition(740, 1130) color("facc15") Type { rectangle(30, 40) }
            Events { OnElapsedTime(5) { repeatCount(-1) Actions {
                broadcast("dark") flash("1f2937", 2)
                delay(2) { Actions { broadcast("lit") } } } } } }
        Entity { id("lamp") startPosition(40, 1150) color("fde047") Type { text("LIGHTS ON", 22) }
            Events {
                OnSignal("dark") { Actions { setText("lamp", "LIGHTS OUT - MOVE!") setColor("64748b") } }
                OnSignal("lit") { Actions { setText("lamp", "LIGHTS ON") setColor("fde047") } } } }

        Entity {
            id("hero")
            startPosition(60, 60)
            color("34d399")
            Type { rectangle(40, 40) }
            Movement { speed(4) Control { movementControl(0) } }
            Events {
                OnSignal("serverOpen") { Actions { flash("67e8f9", 0.3) burst(10, "67e8f9") } }
                OnCollision("exitHatch") { Actions { sound("win2") flash("4ade80", 0.3) burst(16, "4ade80") nextLevel() } }
            }
        }

        Entity { id("mid1") startPosition(20, 400) color("374151") solid(true) Type { rectangle(560, 30) } }
        Entity { id("mid2") startPosition(220, 800) color("374151") solid(true) Type { rectangle(560, 30) } }
        Entity { id("serverWallL") startPosition(20, 1000) color("374151") solid(true) Type { rectangle(300, 30) } }
        Entity { id("serverDoor") startPosition(320, 1000) color("0891b2") Type { rectangle(160, 30) } door("serverOpen", "clang") }
        Entity { id("serverWallR") startPosition(480, 1000) color("374151") solid(true) Type { rectangle(300, 30) } }

        Entity { id("keycard") startPosition(700, 620) color("22d3ee") Type { rectangle(40, 28) } key("serverOpen", "hero", "pickup") }
        Entity { id("exitHatch") startPosition(360, 1080) color("22c55e") Type { rectangle(80, 60) } }

        Entity { id("cone1") startPosition(40, 180) color("55fde68a") Type { rectangle(200, 120) }
            Movement { speed(1.8) Path { waypoint(40, 180) waypoint(560, 180) dwell(0.8) } }
            Events {
                OnSignal("dark") { Actions { setEnable(false) } }
                OnSignal("lit") { Actions { setEnable(true) } }
                OnCollision("hero") { Actions { sound("buzz") updateBar("lives", -1) flash("hero", "ff0000", 0.3)
                    shake(6, 0.3) setPosition("hero", 60, 60, 0) } } } }
        Entity { id("cone2") startPosition(40, 870) color("55fde68a") Type { rectangle(160, 100) }
            Movement { speed(2) Path { waypoint(40, 870) waypoint(600, 870) dwell(0.5) } }
            Events {
                OnSignal("dark") { Actions { setEnable(false) } }
                OnSignal("lit") { Actions { setEnable(true) } }
                OnCollision("hero") { Actions { sound("buzz") updateBar("lives", -1) flash("hero", "ff0000", 0.3)
                    shake(6, 0.3) setPosition("hero", 60, 60, 0) } } } }

        HostilePatrol("guard1", "hero", "-- THE NIGHT GUARD CAUGHT YOU --", 2.2, "ef4444") {
            waypoint(380, 440)
            waypoint(380, 740)
            dwell(0.8)
        }
    }

    # ---------- floor 2 ----------
    Level {
        wallBox(0, 0, 800, 1200, "374151", 20)

        Entity { id("breaker") startPosition(740, 1130) color("facc15") Type { rectangle(30, 40) }
            Events { OnElapsedTime(5) { repeatCount(-1) Actions {
                broadcast("dark") flash("1f2937", 2)
                delay(2) { Actions { broadcast("lit") } } } } } }
        Entity { id("lamp") startPosition(40, 1150) color("fde047") Type { text("LIGHTS ON", 22) }
            Events {
                OnSignal("dark") { Actions { setText("lamp", "LIGHTS OUT - MOVE!") setColor("64748b") } }
                OnSignal("lit") { Actions { setText("lamp", "LIGHTS ON") setColor("fde047") } } } }

        Entity {
            id("hero")
            startPosition(60, 60)
            color("34d399")
            Type { rectangle(40, 40) }
            Movement { speed(4) Control { movementControl(0) } }
            Events {
                OnSignal("vaultOpen") { Actions { flash("67e8f9", 0.3) burst(10, "67e8f9") } }
            }
        }

        # Three columns: up the first, down the second, up the third.
        Entity { id("wallV1") startPosition(260, 20) color("374151") solid(true) Type { rectangle(30, 880) } }
        Entity { id("wallV2") startPosition(520, 300) color("374151") solid(true) Type { rectangle(30, 880) } }

        Entity { id("keycard") startPosition(380, 700) color("22d3ee") Type { rectangle(40, 28) } key("vaultOpen", "hero", "pickup") }
        Entity { id("vaultDoor") startPosition(550, 900) color("0891b2") Type { rectangle(230, 30) } door("vaultOpen", "clang") }
        goal("exit", 600, 1000, 120, 100, "-- OUT BEFORE THE LIGHTS CAME BACK --")

        # This cone fills the first column: only the dark gets you past it.
        Entity { id("cone1") startPosition(40, 300) color("55fde68a") Type { rectangle(200, 150) }
            Movement { speed(1.5) Path { waypoint(40, 300) waypoint(40, 700) dwell(0.6) } }
            Events {
                OnSignal("dark") { Actions { setEnable(false) } }
                OnSignal("lit") { Actions { setEnable(true) } }
                OnCollision("hero") { Actions { sound("buzz") updateBar("lives", -1) flash("hero", "ff0000", 0.3)
                    shake(6, 0.3) setPosition("hero", 60, 60, 0) } } } }
        Entity { id("cone2") startPosition(555, 450) color("55fde68a") Type { rectangle(110, 120) }
            Movement { speed(1.2) Path { waypoint(555, 450) waypoint(665, 450) dwell(0.5) } }
            Events {
                OnSignal("dark") { Actions { setEnable(false) } }
                OnSignal("lit") { Actions { setEnable(true) } }
                OnCollision("hero") { Actions { sound("buzz") updateBar("lives", -1) flash("hero", "ff0000", 0.3)
                    shake(6, 0.3) setPosition("hero", 60, 60, 0) } } } }

        HostilePatrol("guard2", "hero", "-- THE NIGHT GUARD CAUGHT YOU --", 1.6, "ef4444") {
            waypoint(300, 500)
            waypoint(460, 500)
            dwell(0.8)
        }
        HostilePatrol("guard3", "hero", "-- THE NIGHT GUARD CAUGHT YOU --", 1.8, "f97316") {
            waypoint(560, 680)
            waypoint(720, 680)
            dwell(0.6)
        }
    }
}
