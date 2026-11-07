# Reef Tank: keep a small aquarium alive for a week.
# Tap the bubbler for oxygen, drop food from the three hatches as a fish swims below (missed food rots into algae),
# and swipe to scrub the glass. Well-fed fish breed, and every new fish breathes too.
Window {
    size(10, 8)
    background("082f49")
    music("bg_ambient")
    gravity(0)

    Bar("oxygen") { initial(8)
        OnValue(3) { Actions { broadcast("gasp") } }
        OnValue(4) { Actions { broadcast("breathe") } }
        OnValue(0) { Actions { sound("death") shake(6, 0.5) gameOver("-- THE TANK RAN OUT OF AIR --") } } }
    Bar("hunger") { initial(2)
        OnValue(7) { Actions { sound("beep") flash("fish1", "ef4444", 0.4) flash("fish2", "ef4444", 0.4) flash("fish3", "ef4444", 0.4) } }
        OnValue(10) { Actions { sound("lose") gameOver("-- THE FISH STARVED --") } } }
    Bar("algae") { initial(0)
        OnValue(0) { Actions { setOpacity("film", 0) } }
        OnValue(1) { Actions { setOpacity("film", 0.08) } }
        OnValue(2) { Actions { setOpacity("film", 0.16) } }
        OnValue(3) { Actions { setOpacity("film", 0.24) } }
        OnValue(4) { Actions { setOpacity("film", 0.32) } }
        OnValue(5) { Actions { setOpacity("film", 0.4) } }
        OnValue(6) { Actions { setOpacity("film", 0.5) sound("buzz") } }
        OnValue(7) { Actions { setOpacity("film", 0.6) sound("buzz") } }
        OnValue(8) { Actions { sound("lose") gameOver("-- THE TANK TURNED GREEN --") } } }
    Bar("fed") { initial(0)
        OnValue(8) { Actions { setEnable("fish4", true) sound("sparkle") burst("fish4", 14, "f472b6") } }
        OnValue(16) { Actions { setEnable("fish5", true) sound("sparkle") burst("fish5", 14, "a3e635") } } }
    Bar("days") { initial(0)
        OnValue(1) { Actions { broadcast("dawn") } }
        OnValue(2) { Actions { broadcast("dawn") } }
        OnValue(3) { Actions { broadcast("dawn") } }
        OnValue(4) { Actions { broadcast("dawn") } }
        OnValue(5) { Actions { broadcast("dawn") } }
        OnValue(6) { Actions { sound("win") burst("lamp", 30, "fde047") gameOver("-- A HEALTHY WEEK IN THE TANK --") } } }

    Entity { id("gravel") startPosition(0, 0) color("a8a29e") Type { rectangle(1000, 60) }
        Events {
            OnElapsedTime { Actions { ramp("days", 0, 6, 96) } }
            OnElapsedTime(2.5) { repeatCount(-1) Actions { updateBar("oxygen", -1) } }
            OnElapsedTime(4) { repeatCount(-1) Actions { updateBar("hunger", 1) } } } }
    Entity { id("lamp") startPosition(0, 780) color("fef3c7") Type { rectangle(1000, 20) }
        Events { OnSignal("dawn") { Actions { sound("bell") flash("ffffff", 0.6) updateBar("algae", 1) } } } }

    Entity { id("fish1") startPosition(60, 560) color("fb923c") Type { diamond(70, 40) }
        Movement { speed(2) Path { waypoint(60, 560) waypoint(870, 560) } }
        Events { OnSignal("gasp") { Actions { setOpacity(0.5) } } OnSignal("breathe") { Actions { setOpacity(1) } } } }
    Entity { id("fish2") startPosition(870, 380) color("38bdf8") Type { diamond(70, 40) }
        Movement { speed(2.6) Path { waypoint(870, 380) waypoint(60, 380) } }
        Events { OnSignal("gasp") { Actions { setOpacity(0.5) } } OnSignal("breathe") { Actions { setOpacity(1) } } } }
    Entity { id("fish3") startPosition(60, 240) color("facc15") Type { diamond(70, 40) }
        Movement { speed(1.5) Path { waypoint(60, 240) waypoint(870, 240) dwell(0.8) } }
        Events { OnSignal("gasp") { Actions { setOpacity(0.5) } } OnSignal("breathe") { Actions { setOpacity(1) } } } }
    Entity { id("fish4") startPosition(870, 470) color("f472b6") Type { diamond(60, 34) }
        Movement { speed(3) Path { waypoint(870, 470) waypoint(60, 470) } }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnElapsedTime(5) { repeatCount(-1) Actions { updateBar("oxygen", -1) } } } }
    Entity { id("fish5") startPosition(60, 140) color("a3e635") Type { diamond(60, 34) }
        Movement { speed(2.2) Path { waypoint(60, 140) waypoint(870, 140) } }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnElapsedTime(5) { repeatCount(-1) Actions { updateBar("oxygen", -1) } } } }

    Entity { id("bubbler") startPosition(920, 60) color("64748b") Type { rectangle(60, 90) } var("cool", 0)
        Events { OnTapped("bubbler") { Actions { whenVar("cool", "==", 0) { Actions { whenBar("oxygen", "<", 10) { Actions {
            changeVar("cool", 1) updateBar("oxygen", 1) sound("bubble") spawnHere("bubble")
            delay(0.5) { Actions { changeVar("cool", -1) } } } } } } } } } }
    Entity { id("hatchA") startPosition(170, 730) color("78350f") Type { rectangle(80, 40) }
        Events { OnTapped("hatchA") { Actions { spawnHere("pellet") sound("click") } } } }
    Entity { id("hatchB") startPosition(460, 730) color("78350f") Type { rectangle(80, 40) }
        Events { OnTapped("hatchB") { Actions { spawnHere("pellet") sound("click") } } } }
    Entity { id("hatchC") startPosition(750, 730) color("78350f") Type { rectangle(80, 40) }
        Events { OnTapped("hatchC") { Actions { spawnHere("pellet") sound("click") } } } }
    Entity { id("scraper") startPosition(20, 400) color("e2e8f0") Type { rectangle(26, 60) } var("busy", 0)
        Events { OnSwipe { Actions { whenVar("busy", "==", 0) { Actions { whenBar("algae", ">", 0) { Actions {
            changeVar("busy", 1) updateBar("algae", -1) sound("glass") burst(8, "86efac")
            setPosition("scraper", random(20, 950), random(100, 700), 0.3)
            delay(0.8) { Actions { changeVar("busy", -1) } } } } } } } } } }

    Entity { id("film") startPosition(0, 0) color("16a34a") Type { rectangle(1000, 800) }
        Events { OnElapsedTime { Actions { setOpacity(0) } } } }

    Entity { id("pellet") template(true) color("d97706") Type { circle(7) }
        Movement { speed(0.9) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("fish1") { Actions { broadcast("munch") deleteEntity() } }
            OnCollision("fish2") { Actions { broadcast("munch") deleteEntity() } }
            OnCollision("fish3") { Actions { broadcast("munch") deleteEntity() } }
            OnCollision("fish4") { Actions { broadcast("munch") deleteEntity() } }
            OnCollision("fish5") { Actions { broadcast("munch") deleteEntity() } }
            OnCollision("gravel") { Actions { updateBar("algae", 1) burst(5, "65a30d") deleteEntity() } } } }
    Entity { id("bubble") template(true) color("a5f3fc") Type { ring(10, 6) }
        Movement { speed(2.4) AutoMove { direction(0, 1) } }
        Events { OnCollision { Actions { deleteEntity() } } } }

    Entity { id("feedLog") startPosition(20, 70) color("00000000") Type { circle(1) }
        Events { OnSignal("munch") { Actions { sound("pop") updateBar("fed", 1) whenBar("hunger", ">", 0) { Actions { updateBar("hunger", -1) } } } } } }
}
