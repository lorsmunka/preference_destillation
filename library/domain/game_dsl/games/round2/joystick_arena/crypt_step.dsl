# One tap, one step. A thief takes five gems from the crypt; every gem wakes another bat.
Window {
    size(8, 12)
    background("111827")
    music("bg_3")
    gravity(0)

    lives(3, "-- THE CRYPT KEEPS YOU --", "lose")
    winAt("gems", 5, "-- ESCAPED WITH THE GEMS --", "win")
    dpad("thief", "step")
    wallBox(0, 360, 800, 840, "374151", 20)

    Entity {
        id("thief")
        startPosition(80, 440)
        color("4ade80")
        Type { circle(28) }
        Movement { GridMove(80, 0.15) { direction(0, 0) } }
    }

    Entity { id("tomb1") startPosition(240, 600) color("1f2937") solid(true) Type { rectangle(80, 160) } }
    Entity { id("tomb2") startPosition(480, 760) color("1f2937") solid(true) Type { rectangle(160, 80) } }
    Entity { id("tomb3") startPosition(320, 920) color("1f2937") solid(true) Type { rectangle(80, 160) } }
    Entity { id("tomb4") startPosition(560, 520) color("1f2937") solid(true) Type { rectangle(80, 80) } }

    Entity {
        id("gemA")
        startPosition(640, 1000)
        color("38bdf8")
        Type { diamond(36, 44) }
        Events {
            OnCollision("thief") {
                Actions {
                    sound("sparkle")
                    updateBar("gems", 1)
                    burst(10, "38bdf8")
                    spawn("bat", 700, 1100)
                    deleteEntity()
                }
            }
        }
    }
    Entity {
        id("gemB")
        startPosition(80, 1080)
        color("f472b6")
        Type { diamond(36, 44) }
        Events {
            OnCollision("thief") {
                Actions {
                    sound("sparkle")
                    updateBar("gems", 1)
                    burst(10, "f472b6")
                    spawn("bat", 700, 440)
                    deleteEntity()
                }
            }
        }
    }
    Entity {
        id("gemC")
        startPosition(640, 600)
        color("facc15")
        Type { diamond(36, 44) }
        Events {
            OnCollision("thief") {
                Actions {
                    sound("sparkle")
                    updateBar("gems", 1)
                    burst(10, "facc15")
                    spawn("bat", 80, 1100)
                    deleteEntity()
                }
            }
        }
    }
    Entity {
        id("gemD")
        startPosition(160, 760)
        color("a78bfa")
        Type { diamond(36, 44) }
        Events {
            OnCollision("thief") {
                Actions {
                    sound("sparkle")
                    updateBar("gems", 1)
                    burst(10, "a78bfa")
                    spawn("bat", 400, 1100)
                    deleteEntity()
                }
            }
        }
    }
    Entity {
        id("gemE")
        startPosition(400, 1080)
        color("34d399")
        Type { diamond(36, 44) }
        Events {
            OnCollision("thief") {
                Actions {
                    sound("sparkle")
                    updateBar("gems", 1)
                    burst(10, "34d399")
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("bat")
        template(true)
        startPosition(0, 0)
        color("9333ea")
        Type { triangle(34, "down") }
        Movement { speed(0.9) chase("thief") }
        Events {
            OnCollision("thief") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("thief", "ef4444", 0.25)
                    shake(5, 0.3)
                    burst(8, "9333ea")
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("mummy")
        startPosition(80, 880)
        color("d6d3d1")
        Type { rectangle(50, 70) }
        Movement { speed(1.4) Path { waypoint(80, 880) waypoint(700, 880) dwell(0.8) } }
        Events {
            OnCollision("thief") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    shake(5, 0.3)
                    setPosition("thief", 80, 440, 0)
                }
            }
        }
    }
}
