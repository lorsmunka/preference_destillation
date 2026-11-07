# Quilting. Each step of the needle leaves a stitch in the square it left, and the needle cannot
# cross its own stitches. A fully stitched 3x3 patch is finished and lifts off the frame; so does
# a fully stitched seam (one column). Unpick takes back the three newest stitches.
Window {
    size(8, 12)
    background("1f1630")
    music("bg_6")
    gravity(0)

    Bar("patches") {
        initial(0)
        OnValue(6) { Actions { sound("win") shake(5, 0.5) burst("title", 30, "f9a8d4") gameOver("-- THE QUILT IS FINISHED --") } }
    }
    Bar("thread") {
        initial(50)
        OnValue(0) { Actions { sound("lose") gameOver("-- OUT OF THREAD --") } }
    }

    Entity { id("title") startPosition(110, 1120) color("f9a8d4") Type { text("STITCH SIX PATCHES", 40) } }

    Entity { id("frameL") startPosition(0, 400) color("4c1d95") Type { rectangle(100, 700) } solid(true) }
    Entity { id("frameR") startPosition(700, 400) color("4c1d95") Type { rectangle(100, 700) } solid(true) }
    Entity { id("frameT") startPosition(100, 1000) color("4c1d95") Type { rectangle(600, 100) } solid(true) }

    dpad("needle", "step")

    Entity {
        id("stitch")
        template(true)
        startPosition(0, 0)
        color("f472b6")
        Type { cross(80, 80, 18) }
        solid(true)
        var("age", 0)
        Events {
            OnElapsedTime {
                Actions {
                    clearFullBoxes(100, 3, 100, 400, 6, 6, "stitch", "patchDone")
                    clearFullColumns(100, 6, 100, 400, 6, "stitch", "seamDone")
                }
            }
            OnSignal("aged") { Actions { changeVar("age", 1) } }
            OnSignal("unravel") { Actions { whenVar("age", "<=", 2) { Actions { burst(6, "f472b6") deleteEntity() } } } }
        }
    }

    Entity {
        id("needle")
        startPosition(110, 410)
        color("e2e8f0")
        Type { diamond(80, 80) }
        Movement { GridMove(100, 1) { direction(0, 0) } }
        Events {
            OnGridStep {
                Actions {
                    sound("click")
                    broadcast("aged")
                    spawnHere("stitch")
                    updateBar("thread", -1)
                }
            }
        }
    }

    Entity {
        id("unpick")
        startPosition(625, 250)
        color("fbcfe8")
        Type { text("UNPICK", 34) }
        Events {
            OnTapped("unpick") {
                Actions {
                    whenBar("thread", "<=", 3) {
                        Actions {
                            sound("lose")
                            gameOver("-- THE THREAD SNAPPED --")
                        }
                    }
                    whenBar("thread", ">", 3) {
                        Actions {
                            sound("pop")
                            flash("ffffff", 0.15)
                            broadcast("unravel")
                            updateBar("thread", -3)
                        }
                    }
                }
            }
        }
    }

    Entity {
        id("basket")
        startPosition(20, 120)
        color("fda4af")
        Type { ring(60, 38) }
        Events {
            OnSignal("patchDone") {
                Actions {
                    sound("sparkle")
                    shake(3, 0.25)
                    burst(18, "f9a8d4")
                    updateBar("patches", 1)
                    updateBar("thread", 6)
                }
            }
            OnSignal("seamDone") {
                Actions {
                    sound("bell")
                    burst(10, "c4b5fd")
                    updateBar("thread", 4)
                }
            }
        }
    }
}
