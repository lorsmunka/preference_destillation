Window {
    size(8, 10) worldSize(8, 24) cameraFollow("mole")
    background("422006") music("bg_nature") gravity(0)

    lives(3, "-- TOO MUCH MARSH GAS --", "lose")
    Bar("worms") { initial(0)
        OnValue(3) { Actions { broadcast("halfway") } } }
    winAt("worms", 6, "-- A FULL LARDER FOR WINTER --", "win")

    Entity { id("sign") startPosition(40, 2330) color("fef3c7") Type { text("Swipe to dig. Worms give themselves away with a wriggle.", 22) }
        Events { OnSignal("halfway") { Actions { setText("sign", "Half the larder is full - dig deeper") } } } }

    # Worms and gas pockets are placed first, so the soil spawned below hides them.
    Entity { id("w1") startPosition(125, 1842) color("f472b6") Type { rectangle(50, 16) }
        Events { OnElapsedTime(random(1.5, 3)) { repeatCount(-1) Actions { burst(3, "f472b6") } }
            OnCollision("mole") { Actions { sound("pickup") updateBar("worms", 1) deleteEntity() } } } }
    Entity { id("w2") startPosition(625, 1542) color("f472b6") Type { rectangle(50, 16) }
        Events { OnElapsedTime(random(1.5, 3)) { repeatCount(-1) Actions { burst(3, "f472b6") } }
            OnCollision("mole") { Actions { sound("pickup") updateBar("worms", 1) deleteEntity() } } } }
    Entity { id("w3") startPosition(225, 1142) color("f472b6") Type { rectangle(50, 16) }
        Events { OnElapsedTime(random(1.5, 3)) { repeatCount(-1) Actions { burst(3, "f472b6") } }
            OnCollision("mole") { Actions { sound("pickup") updateBar("worms", 1) deleteEntity() } } } }
    Entity { id("w4") startPosition(525, 742) color("f472b6") Type { rectangle(50, 16) }
        Events { OnElapsedTime(random(1.5, 3)) { repeatCount(-1) Actions { burst(3, "f472b6") } }
            OnCollision("mole") { Actions { sound("pickup") updateBar("worms", 1) deleteEntity() } } } }
    Entity { id("w5") startPosition(25, 342) color("f472b6") Type { rectangle(50, 16) }
        Events { OnElapsedTime(random(1.5, 3)) { repeatCount(-1) Actions { burst(3, "f472b6") } }
            OnCollision("mole") { Actions { sound("pickup") updateBar("worms", 1) deleteEntity() } } } }
    Entity { id("w6") startPosition(725, 142) color("f472b6") Type { rectangle(50, 16) }
        Events { OnElapsedTime(random(1.5, 3)) { repeatCount(-1) Actions { burst(3, "f472b6") } }
            OnCollision("mole") { Actions { sound("pickup") updateBar("worms", 1) deleteEntity() } } } }

    Entity { id("gas") template(true) color("84cc16") Type { circle(30) }
        Events { OnCollision("mole") { Actions { sound("hurt") shake(7, 0.4) burst(20, "84cc16") flash("mole", "84cc16", 0.5) updateBar("lives", -1) deleteEntity() } } } }
    Entity { id("dirt") template(true) color("78350f") Type { rectangle(100, 100) } }
    Entity { id("rock") template(true) color("57534e") solid(true) Type { polygon(6, 50) } }

    Entity { id("digger") startPosition(0, 2100) color("00000000") Type { rectangle(0, 0) }
        Movement { GridMove(100) }
        Events {
            OnElapsedTime { Actions { spawn("gas", 420, 1720) spawn("gas", 120, 1320) spawn("gas", 620, 920) spawn("gas", 320, 420) } }
            OnElapsedTime(0.02) { repeatCount(22) Actions {
                spawnHere("dirt") loopCount(7) gridStep(1, 0) spawnHere("dirt") loopCount(1) gridStep(-7, -1) } }
            OnElapsedTime(0.6) { Actions {
                spawn("rock", 200, 2000) spawn("rock", 500, 1900) spawn("rock", 300, 1600) spawn("rock", 0, 1400)
                spawn("rock", 600, 1200) spawn("rock", 400, 1000) spawn("rock", 100, 800) spawn("rock", 700, 600)
                spawn("rock", 300, 600) spawn("rock", 500, 300) spawn("rock", 200, 100) } } } }

    Entity { id("mole") startPosition(310, 2210) color("a8a29e") Type { circle(40) }
        Movement { GridMove(100, 0.22) { direction(0, 0) } }
        Events {
            OnSwipe("left") { Actions { setDirection("mole", -1, 0) } }
            OnSwipe("right") { Actions { setDirection("mole", 1, 0) } }
            OnSwipe("up") { Actions { setDirection("mole", 0, 1) } }
            OnSwipe("down") { Actions { setDirection("mole", 0, -1) } }
            OnGridBlocked { Actions { setDirection("mole", 0, 0) sound("bump") } }
            OnCollision("dirt") { Actions { deleteCollider() burst(4, "92400e") } }
            OnSignal("halfway") { Actions { sound("sparkle") flash("fde047", 0.4) } } } }
}
