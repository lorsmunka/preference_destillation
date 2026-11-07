# Searchlight Yard: a raccoon raids a scrapyard at night. Stay out of the sweeping lights;
# every time one catches you, the kennel lets a dog out for a few seconds.
Window {
    size(10, 10)
    background("0b1120")
    music("bg_6")
    gravity(0)

    lives(4, "-- CAUGHT IN THE LIGHT --", "lose")
    Bar("snacks") {
        initial(0)
        OnValue(6) { Actions { broadcast("brighter") } }
    }
    winAt("snacks", 12, "-- BELLY FULL, BACK TO THE WOODS --", "win")

    joystick("raccoon", 85, 4.5)

    Entity {
        id("raccoon")
        startPosition(180, 420)
        color("a8a29e")
        Type { circle(20) }
        Movement { speed(0) Control { movementControl(0) } }
        Events {
            OnElapsedTime { Actions { spawn("snack", 500, 500) } }
        }
    }

    Entity { id("crate1") startPosition(250, 600) color("57534e") solid(true) Type { rectangle(120, 80) } }
    Entity { id("crate2") startPosition(620, 300) color("57534e") solid(true) Type { rectangle(80, 140) } }
    Entity { id("crate3") startPosition(450, 820) color("57534e") solid(true) Type { rectangle(140, 60) } }

    Entity {
        id("snack")
        template(true)
        startPosition(0, 0)
        color("fb923c")
        Type { polygon(5, 16) }
        Events {
            OnCollision("raccoon") {
                Actions {
                    sound("coin")
                    burst(10, "fb923c")
                    updateBar("snacks", 1)
                    randomAction {
                        Actions { spawn("snack", 300, 300) }
                        Actions { spawn("snack", 800, 150) }
                        Actions { spawn("snack", 150, 850) }
                        Actions { spawn("snack", 850, 850) }
                        Actions { spawn("snack", 500, 500) }
                        Actions { spawn("snack", 760, 600) }
                        Actions { spawn("snack", 300, 450) }
                        Actions { spawn("snack", 600, 700) }
                    }
                    deleteEntity()
                }
            }
        }
    }

    Entity { id("light1") startPosition(0, 0) color("44fef08a") Type { circle(70) }
        Movement { speed(2) Path { waypoint(0, 650) waypoint(860, 650) dwell(0.6) } }
        Events {
            OnCollision("raccoon") { Actions { sound("buzz") updateBar("lives", -1) flash("raccoon", "fef08a", 0.4)
                shake(5, 0.3) broadcast("alarm") } }
            OnSignal("brighter") { Actions { updateSpeed(3) setColor("66fde047") } }
        } }
    Entity { id("light2") startPosition(0, 0) color("44fef08a") Type { circle(70) }
        Movement { speed(2.4) Path { waypoint(420, 0) waypoint(420, 860) } }
        Events {
            OnCollision("raccoon") { Actions { sound("buzz") updateBar("lives", -1) flash("raccoon", "fef08a", 0.4)
                shake(5, 0.3) broadcast("alarm") } }
            OnSignal("brighter") { Actions { updateSpeed(3.4) setColor("66fde047") } }
        } }
    Entity { id("light3") startPosition(0, 0) color("44fef08a") Type { circle(55) }
        Movement { speed(1.8) Path { waypoint(100, 100) waypoint(800, 100) waypoint(800, 800) waypoint(100, 800) } }
        Events {
            OnCollision("raccoon") { Actions { sound("buzz") updateBar("lives", -1) flash("raccoon", "fef08a", 0.4)
                shake(5, 0.3) broadcast("alarm") } }
            OnSignal("brighter") { Actions { updateSpeed(2.6) setColor("66fde047") } }
        } }

    Entity {
        id("kennel")
        startPosition(900, 900)
        color("7f1d1d")
        Type { rectangle(100, 100) }
        Events {
            OnSignal("alarm") {
                Actions {
                    setColor("ef4444")
                    delay(0.6) { Actions { sound("clang") spawn("dog", 920, 920) setColor("7f1d1d") } }
                }
            }
        }
    }

    Entity {
        id("dog")
        template(true)
        startPosition(0, 0)
        color("b45309")
        Type { rectangle(36, 28) }
        Movement { speed(3.2) chase("raccoon") }
        Events {
            OnCollision("raccoon") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("raccoon", "ff0000", 0.4)
                    shake(7, 0.35)
                    burst(14, "b45309")
                    deleteEntity()
                }
            }
            OnElapsedTime(6) { Actions { burst(8, "d6d3d1") deleteEntity() } }
        }
    }
}
