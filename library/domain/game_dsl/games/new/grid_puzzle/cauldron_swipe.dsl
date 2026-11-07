# Witch's kitchen: swipe and every ingredient slides that way; two equal ones steep into the next
# kind (sprig 1, root 2, cap 4, crystal 8, stardust 16). Only a crystal or better that comes to
# rest on the cauldron (bottom-left square) is brewed. A new sprig drops in after every swipe.
Window {
    size(7, 11)
    background("1e1b2e")
    music("bg_5")
    gravity(0)

    Bar("potions") { initial(0) }
    winAt("potions", 5, "-- THE COVEN IS IMPRESSED --", "win")
    Bar("moves") { initial(80)
        OnValue(0) { Actions { sound("lose") gameOver("-- THE FIRE WENT OUT --") } } }

    Entity { id("wallL") startPosition(30, 230) color("3b3363") Type { rectangle(20, 640) } solid(true) }
    Entity { id("wallR") startPosition(650, 230) color("3b3363") Type { rectangle(20, 640) } solid(true) }
    Entity { id("wallT") startPosition(30, 850) color("3b3363") Type { rectangle(640, 20) } solid(true) }
    Entity { id("wallB") startPosition(30, 230) color("3b3363") Type { rectangle(640, 20) } solid(true) }

    Entity { id("cauldron") startPosition(55, 255) color("4d7c0f") Type { ring(55, 38) }
        Events { OnElapsedTime(1.5) { repeatCount(-1) Actions { burst(4, "a3e635") } } } }

    # Legend: the colour of each kind of ingredient.
    Entity { id("key1") startPosition(80, 950) color("86efac") Type { text("1", 48) } }
    Entity { id("key2") startPosition(200, 950) color("fca5a5") Type { text("2", 48) } }
    Entity { id("key4") startPosition(320, 950) color("c4b5fd") Type { text("4", 48) } }
    Entity { id("key8") startPosition(440, 950) color("fde047") Type { text("8", 48) } }
    Entity { id("key16") startPosition(550, 950) color("f0abfc") Type { text("16", 48) } }

    Entity { id("herb") template(true) startPosition(0, 0) color("86efac") Type { rectangle(110, 110) } var("potency", 1)
        Events {
            OnSignal("steeped") { Actions {
                whenVar("potency", "==", 2) { Actions { setColor("fca5a5") } }
                whenVar("potency", "==", 4) { Actions { setColor("c4b5fd") } }
                whenVar("potency", "==", 8) { Actions { setColor("fde047") } }
                whenVar("potency", "==", 16) { Actions { setColor("f0abfc") } } } }
            OnCollision("cauldron") { Actions { whenVar("potency", ">=", 8) { Actions {
                sound("sparkle") flash("cauldron", "fde047", 0.4) burst(20, "f0abfc") shake(3, 0.2)
                updateBar("potions", 1) deleteEntity() } } } } } }

    Entity { id("pantry") startPosition(0, 0) color("00000000") Type { circle(1) }
        Events { OnElapsedTime(0.1) { Actions {
            spawnAtEmpty("herb", 120, 55, 255, 5, 5) spawnAtEmpty("herb", 120, 55, 255, 5, 5) } } } }

    Entity { id("spoon") startPosition(40, 120) color("e9d5ff") Type { text("swipe to stir", 40) }
        Events {
            OnSwipe("left") { Actions { sound("bubble") slideAll(-1, 0, 120, "herb", "potency", "steeped")
                spawnAtEmpty("herb", 120, 55, 255, 5, 5) updateBar("moves", -1) } }
            OnSwipe("right") { Actions { sound("bubble") slideAll(1, 0, 120, "herb", "potency", "steeped")
                spawnAtEmpty("herb", 120, 55, 255, 5, 5) updateBar("moves", -1) } }
            OnSwipe("up") { Actions { sound("bubble") slideAll(0, 1, 120, "herb", "potency", "steeped")
                spawnAtEmpty("herb", 120, 55, 255, 5, 5) updateBar("moves", -1) } }
            OnSwipe("down") { Actions { sound("bubble") slideAll(0, -1, 120, "herb", "potency", "steeped")
                spawnAtEmpty("herb", 120, 55, 255, 5, 5) updateBar("moves", -1) } } } }
}
