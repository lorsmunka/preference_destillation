# Roastery chute: one bean at a time drops down the middle lane. Swipe it into the left jar
# (light roast) or the right jar (dark roast) before it lands; a gold bean is good in either jar.
Window {
    size(7, 12)
    background("1c1410")
    music("bg_4")
    gravity(0)

    Bar("sorted") { initial(0) }
    winAt("sorted", 20, "-- THE ROASTERY IS STOCKED --", "win")
    lives(3, "-- TOO MANY BEANS ON THE FLOOR --", "lose")

    Entity { id("wallW") startPosition(0, 40) color("292018") Type { rectangle(170, 1160) } solid(true) }
    Entity { id("wallE") startPosition(530, 40) color("292018") Type { rectangle(170, 1160) } solid(true) }
    Entity { id("floor") startPosition(170, 40) color("44403c") Type { rectangle(360, 110) } solid(true) }
    Entity { id("jarLight") startPosition(170, 150) color("d6a26b") Type { rectangle(120, 110) } }
    Entity { id("spill") startPosition(290, 150) color("7f1d1d") Type { rectangle(120, 110) } }
    Entity { id("jarDark") startPosition(410, 150) color("4a2511") Type { rectangle(120, 110) } }

    Entity { id("light") template(true) startPosition(0, 0) color("e7b98a") Type { circle(40) }
        Movement { GridMove(120, 0.35) { direction(0, -1) } }
        Events {
            OnCollision("jarLight") { Actions { sound("coin") burst(8, "e7b98a") updateBar("sorted", 1) } }
            OnCollision("jarDark") { Actions { sound("hurt") flash("ef4444", 0.3) updateBar("lives", -1) } }
            OnCollision("spill") { Actions { sound("hurt") shake(4, 0.3) updateBar("lives", -1) } }
            OnGridBlocked { Actions { deleteEntity() broadcast("nextBean") } } } }
    Entity { id("dark") template(true) startPosition(0, 0) color("3b1a08") Type { circle(40) }
        Movement { GridMove(120, 0.35) { direction(0, -1) } }
        Events {
            OnCollision("jarDark") { Actions { sound("coin") burst(8, "a16207") updateBar("sorted", 1) } }
            OnCollision("jarLight") { Actions { sound("hurt") flash("ef4444", 0.3) updateBar("lives", -1) } }
            OnCollision("spill") { Actions { sound("hurt") shake(4, 0.3) updateBar("lives", -1) } }
            OnGridBlocked { Actions { deleteEntity() broadcast("nextBean") } } } }
    Entity { id("gold") template(true) startPosition(0, 0) color("facc15") Type { star(5, 40, 18) }
        Movement { GridMove(120, 0.3) { direction(0, -1) } }
        Events {
            OnCollision("jarLight") { Actions { sound("sparkle") burst(14, "facc15") updateBar("sorted", 1) updateBar("lives", 1) } }
            OnCollision("jarDark") { Actions { sound("sparkle") burst(14, "facc15") updateBar("sorted", 1) updateBar("lives", 1) } }
            OnCollision("spill") { Actions { sound("hurt") shake(4, 0.3) updateBar("lives", -1) } }
            OnGridBlocked { Actions { deleteEntity() broadcast("nextBean") } } } }

    Entity { id("hopper") startPosition(300, 1110) color("78350f") Type { rectangle(100, 60) }
        Events {
            OnElapsedTime(1) { Actions { spawn("light", 310, 1020) } }
            OnSignal("nextBean") { Actions { randomAction {
                Actions { spawn("light", 310, 1020) }
                Actions { spawn("dark", 310, 1020) }
                Actions { spawn("light", 310, 1020) }
                Actions { spawn("dark", 310, 1020) }
                Actions { spawn("gold", 310, 1020) } } } } } }

    # Swipes shove whichever bean is falling (only one is ever in the chute).
    Entity { id("roaster") startPosition(20, 1150) color("fcd34d") Type { text("swipe to sort", 30) }
        Events {
            OnSwipe("left") { Actions { sound("blip") gridStep("light", -1, 0) gridStep("dark", -1, 0) gridStep("gold", -1, 0) } }
            OnSwipe("right") { Actions { sound("blip") gridStep("light", 1, 0) gridStep("dark", 1, 0) gridStep("gold", 1, 0) } }
            OnSwipe("down") { Actions { sound("click") gridStep("light", 0, -1) gridStep("dark", 0, -1) gridStep("gold", 0, -1) } } } }
}
