# Flower Guard: bugs crawl in from every edge toward the flower. Tap them to squash them.
Window {
    size(8, 8)
    background("166534")
    music("bg_nature")
    gravity(0)

    lives(4, "-- THE FLOWER WILTED --", "lose")
    Bar("squashed") { initial(0) }

    Wave(12, 1.4) { Actions { randomAction {
        Actions { spawn("ant", random(0, 770), 770) }
        Actions { spawn("ant", random(0, 770), 0) }
        Actions { spawn("ant", 0, random(0, 770)) }
        Actions { spawn("ant", 770, random(0, 770)) } } } }
    Wave(15, 1.0) { Actions { randomAction {
        Actions { spawn("ant", random(0, 770), 770) }
        Actions { spawn("beetle", 0, random(0, 750)) }
        Actions { spawn("ant", 770, random(0, 770)) }
        Actions { spawn("beetle", random(0, 750), 0) } } } }
    Wave(18, 0.7) { Actions { randomAction {
        Actions { spawn("ant", random(0, 770), 0) spawn("ant", 0, random(0, 770)) }
        Actions { spawn("beetle", 750, random(0, 750)) }
        Actions { spawn("beetle", random(0, 750), 750) }
        Actions { spawn("ant", 770, random(0, 770)) spawn("ant", random(0, 770), 770) } } } }
    Wave(5) { Actions { sound("win") shake(5, 0.5) burst("flower", 40, "fbcfe8")
        gameOver("-- THE FLOWER BLOOMED --") } }

    Entity { id("flower") startPosition(360, 360) color("f472b6") Type { star(8, 40, 22) }
        var("bloom", 4)
        Events {
            OnCollision("ant") { Actions { deleteCollider() broadcast("bitten") } }
            OnCollision("beetle") { Actions { deleteCollider() broadcast("bitten") } }
            OnSignal("bitten") { Actions {
                sound("hurt") shake(5, 0.3) flash("ef4444", 0.3) burst(12, "f9a8d4")
                updateBar("lives", -1) changeVar("bloom", -1)
                whenVar("bloom", "==", 3) { Actions { setScale(0.85) setPosition("flower", 366, 366, 0) } }
                whenVar("bloom", "==", 2) { Actions { setScale(0.7) setPosition("flower", 372, 372, 0) } }
                whenVar("bloom", "==", 1) { Actions { setScale(0.55) setPosition("flower", 378, 378, 0) } } } }
        } }

    Entity { id("ant") template(true) color("1c1917") Type { circle(15) }
        Movement { speed(1.1) chase("flower") }
        Events { OnTapped { Actions {
            sound("pop") burst(8, "a8a29e") updateBar("squashed", 1) deleteEntity() } } } }

    Entity { id("beetle") template(true) color("7c3aed") Type { polygon(6, 25) }
        var("shell", 3)
        Movement { speed(0.7) chase("flower") }
        Events { OnTapped { Actions {
            changeVar("shell", -1) sound("clang") flash("ffffff", 0.12)
            whenVar("shell", "==", 2) { Actions { setScale(0.85) } }
            whenVar("shell", "==", 1) { Actions { setScale(0.7) } }
            whenVar("shell", "<=", 0) { Actions {
                sound("pop") burst(14, "c4b5fd") updateBar("squashed", 1) deleteEntity() } } } } } }
}
