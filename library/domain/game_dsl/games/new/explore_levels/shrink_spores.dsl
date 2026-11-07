# Shrink Spores: blue caps make you tiny for a while (slip through the narrow gaps in the hedges),
# red caps make you huge (smash cracked logs, squash beetles). The "size" bar says which you are.
Window {
    size(8, 12)
    worldSize(18, 12)
    cameraFollow("hero")
    background("14532d")
    music("bg_1")
    gravity(0)

    lives(3, "-- THE BEETLES WON --", "lose")
    Bar("size") { initial(0) }
    Bar("beetles") {
        initial(0)
        OnValue(3) { Actions { broadcast("bugFree") } }
    }

    Level {
        Entity { id("sign") startPosition(60, 1120) color("bbf7d0") Type { text("Blue caps shrink you. Red caps make you huge.", 26) }
            Events { OnSignal("acorn") { Actions { setText("sign", "The burrow door is open, far east") } } } }

        Entity { id("hero") startPosition(100, 550) color("fde68a") Type { circle(30) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnCollision("capBlue") { Actions { whenBar("size", "==", 0) { Actions {
                    setScale(0.45) updateBar("size", -1) sound("pop") burst(12, "60a5fa")
                    delay(7) { Actions { setScale(1) updateBar("size", 1) sound("blip") } } } } } }
                OnCollision("capRed") { Actions { whenBar("size", "==", 0) { Actions {
                    setScale(1.6) updateBar("size", 1) sound("pop") burst(12, "f87171")
                    delay(6) { Actions { setScale(1) updateBar("size", -1) sound("blip") } } } } } }
                OnSignal("acorn") { Actions { burst(12, "a16207") } }
                OnCollision("burrow") { Actions { sound("win2") burst(20, "fde68a") nextLevel() } } } }
        joystick("hero", 80, 3)

        Entity { id("capBlue") template(true) color("60a5fa") Type { circle(22) } }
        Entity { id("capRed") template(true) color("f87171") Type { circle(22) } }
        Entity { id("garden") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnElapsedTime { Actions { spawn("capBlue", 300, 300) spawn("capRed", 900, 1000) } } } }

        Entity { id("hedgeA1") startPosition(600, 0) color("166534") solid(true) Type { rectangle(40, 580) } }
        Entity { id("hedgeA2") startPosition(600, 620) color("166534") solid(true) Type { rectangle(40, 580) } }
        Entity { id("hedgeB1") startPosition(1200, 0) color("166534") solid(true) Type { rectangle(40, 500) } }
        Entity { id("hedgeB2") startPosition(1200, 700) color("166534") solid(true) Type { rectangle(40, 500) } }
        Entity { id("log1") startPosition(1200, 500) color("92400e") solid(true) Type { rectangle(40, 200) } }
        Entity { id("crack1") startPosition(1170, 480) color("00000000") Type { rectangle(100, 240) }
            Events { OnCollision("hero") { Actions { whenBar("size", "==", 1) { Actions {
                sound("thud") shake(8, 0.4) burst(20, "92400e") setEnable("log1", false) deleteEntity() } } } } } }

        Entity { id("nookSide") startPosition(1450, 0) color("166534") solid(true) Type { rectangle(40, 330) } }
        Entity { id("nookTop") startPosition(1450, 330) color("166534") solid(true) Type { rectangle(200, 40) } }
        Entity { id("nookDoor") startPosition(1650, 330) color("a16207") Type { rectangle(150, 40) } door("acorn", "clang") }
        Entity { id("acornKey") startPosition(1600, 1000) color("ca8a04") Type { polygon(5, 22) } key("acorn", "hero", "pickup") }
        Entity { id("burrow") startPosition(1600, 100) color("292524") Type { ring(50, 30) } }

        Entity { id("beetle1") color("1e1b4b") Type { polygon(6, 26) }
            Movement { speed(2) Path { waypoint(700, 200) waypoint(1100, 200) waypoint(1100, 900) waypoint(700, 900) } }
            Events { OnCollision("hero") { Actions {
                whenBar("size", "==", 1) { Actions { sound("pop") burst(16, "6366f1") updateBar("beetles", 1) deleteEntity() } }
                whenBar("size", "<", 1) { Actions { sound("hurt") shake(6, 0.3) flash("hero", "ef4444", 0.3) updateBar("lives", -1) } } } } } }
        Entity { id("beetle2") color("1e1b4b") Type { polygon(6, 26) }
            Movement { speed(2.4) Path { waypoint(1300, 800) waypoint(1750, 800) dwell(0.5) } }
            Events { OnCollision("hero") { Actions {
                whenBar("size", "==", 1) { Actions { sound("pop") burst(16, "6366f1") updateBar("beetles", 1) deleteEntity() } }
                whenBar("size", "<", 1) { Actions { sound("hurt") shake(6, 0.3) flash("hero", "ef4444", 0.3) updateBar("lives", -1) } } } } } }
    }

    Level {
        Entity { id("sign") startPosition(60, 1120) color("bbf7d0") Type { text("Grow to break the log, shrink for the high gap", 26) }
            Events {
                OnSignal("acorn") { Actions { setText("sign", "Door open - into the burrow!") } }
                OnSignal("bugFree") { Actions { setText("sign", "Three beetles squashed - the garden thanks you") } } } }

        Entity { id("hero") startPosition(100, 100) color("fde68a") Type { circle(30) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnElapsedTime { Actions {
                    whenBar("size", "==", -1) { Actions { updateBar("size", 1) } }
                    whenBar("size", "==", 1) { Actions { updateBar("size", -1) } } } }
                OnCollision("capBlue") { Actions { whenBar("size", "==", 0) { Actions {
                    setScale(0.45) updateBar("size", -1) sound("pop") burst(12, "60a5fa")
                    delay(7) { Actions { setScale(1) updateBar("size", 1) sound("blip") } } } } } }
                OnCollision("capRed") { Actions { whenBar("size", "==", 0) { Actions {
                    setScale(1.6) updateBar("size", 1) sound("pop") burst(12, "f87171")
                    delay(6) { Actions { setScale(1) updateBar("size", -1) sound("blip") } } } } } }
                OnSignal("acorn") { Actions { burst(12, "a16207") } }
                OnSignal("bugFree") { Actions { sound("sparkle") flash("fde047", 0.4) } } } }
        joystick("hero", 80, 3)

        Entity { id("capBlue") template(true) color("60a5fa") Type { circle(22) } }
        Entity { id("capRed") template(true) color("f87171") Type { circle(22) } }
        Entity { id("garden") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnElapsedTime { Actions { spawn("capRed", 250, 900) spawn("capBlue", 800, 200) spawn("capRed", 1400, 300) } } } }

        Entity { id("hedgeA") startPosition(500, 200) color("166534") solid(true) Type { rectangle(40, 1000) } }
        Entity { id("log1") startPosition(500, 0) color("92400e") solid(true) Type { rectangle(40, 200) } }
        Entity { id("crack1") startPosition(470, 0) color("00000000") Type { rectangle(100, 220) }
            Events { OnCollision("hero") { Actions { whenBar("size", "==", 1) { Actions {
                sound("thud") shake(8, 0.4) burst(20, "92400e") setEnable("log1", false) deleteEntity() } } } } } }
        Entity { id("hedgeB1") startPosition(1100, 0) color("166534") solid(true) Type { rectangle(40, 980) } }
        Entity { id("hedgeB2") startPosition(1100, 1020) color("166534") solid(true) Type { rectangle(40, 180) } }

        Entity { id("nookSide") startPosition(1450, 0) color("166534") solid(true) Type { rectangle(40, 330) } }
        Entity { id("nookTop") startPosition(1450, 330) color("166534") solid(true) Type { rectangle(200, 40) } }
        Entity { id("nookDoor") startPosition(1650, 330) color("a16207") Type { rectangle(150, 40) } door("acorn", "clang") }
        Entity { id("acornKey") startPosition(1700, 1100) color("ca8a04") Type { polygon(5, 22) } key("acorn", "hero", "pickup") }
        goal("burrow", 1580, 80, 120, 120, "-- HOME IN THE BURROW --")

        Entity { id("beetle1") color("1e1b4b") Type { polygon(6, 26) }
            Movement { speed(2.2) Path { waypoint(600, 600) waypoint(1000, 600) waypoint(1000, 1100) waypoint(600, 1100) } }
            Events { OnCollision("hero") { Actions {
                whenBar("size", "==", 1) { Actions { sound("pop") burst(16, "6366f1") updateBar("beetles", 1) deleteEntity() } }
                whenBar("size", "<", 1) { Actions { sound("hurt") shake(6, 0.3) flash("hero", "ef4444", 0.3) updateBar("lives", -1) } } } } } }
        Entity { id("beetle2") color("1e1b4b") Type { polygon(6, 26) }
            Movement { speed(2.6) Path { waypoint(1200, 700) waypoint(1750, 700) } }
            Events { OnCollision("hero") { Actions {
                whenBar("size", "==", 1) { Actions { sound("pop") burst(16, "6366f1") updateBar("beetles", 1) deleteEntity() } }
                whenBar("size", "<", 1) { Actions { sound("hurt") shake(6, 0.3) flash("hero", "ef4444", 0.3) updateBar("lives", -1) } } } } } }
        Entity { id("beetle3") color("1e1b4b") Type { polygon(6, 26) }
            Movement { speed(1.8) Path { waypoint(1250, 450) waypoint(1250, 1100) dwell(0.4) } }
            Events { OnCollision("hero") { Actions {
                whenBar("size", "==", 1) { Actions { sound("pop") burst(16, "6366f1") updateBar("beetles", 1) deleteEntity() } }
                whenBar("size", "<", 1) { Actions { sound("hurt") shake(6, 0.3) flash("hero", "ef4444", 0.3) updateBar("lives", -1) } } } } } }
    }
}
