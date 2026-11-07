Window {
    size(10, 14)
    background("1c1917")
    music("bg_1")
    gravity(0)

    Bar("intel") {
        initial(0)
        OnValue(3) {
            Actions {
                broadcast("extract")
            }
        }
    }

    wallBox(0, 0, 1000, 1400, "292524", 20)

    # ---------- the spy ----------
    Entity {
        id("hero")
        startPosition(60, 60)
        color("d4d4d8")
        Type { rectangle(36, 36) }
        Movement {
            speed(4.2)
            Control { movementControl(0) }
        }
        Events {
            OnSignal("drop1") {
                Actions {
                    flash("fde68a", 0.3)
                    burst(10, "fde68a")
                }
            }
            OnSignal("burned") {
                Actions {
                    flash("ef4444", 0.5)
                    shake(5, 0.4)
                }
            }
            OnSignal("extract") {
                Actions {
                    flash("4ade80", 0.4)
                    burst(16, "4ade80")
                }
            }
        }
    }

    Entity {
        id("status")
        startPosition(40, 1350)
        color("fde68a")
        Type { text("Drop 1: the bench by the east path", 20) }
        Events {
            OnSignal("drop1") { Actions { setText("status", "Drop 2: the phone booth on the west side") } }
            OnSignal("burned") { Actions { setText("status", "Cover blown! A drone is up. Drop 3: the rose garden") } }
            OnSignal("extract") { Actions { setText("status", "Helicopter inbound: get to the pad") } }
        }
    }

    # ---------- the three drops, each revealed by the one before ----------
    Entity {
        id("drop1")
        startPosition(860, 200)
        color("a16207")
        Type { rectangle(80, 30) }
        Events {
            OnCollision("hero") {
                Actions {
                    sound("pickup")
                    updateBar("intel", 1)
                    burst(12, "fbbf24")
                    setEnable("drop2", true)
                    broadcast("drop1")
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("drop2")
        startPosition(60, 720)
        color("dc2626")
        Type { rectangle(40, 70) }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnCollision("hero") {
                Actions {
                    sound("pickup")
                    updateBar("intel", 1)
                    burst(12, "f87171")
                    setEnable("drop3", true)
                    broadcast("burned")
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("drop3")
        startPosition(880, 1250)
        color("a8a29e")
        Type { star(5, 26, 11) }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnCollision("hero") {
                Actions {
                    sound("sparkle")
                    updateBar("intel", 1)
                    burst(16, "e7e5e4")
                    deleteEntity()
                }
            }
        }
    }

    # The handler keeps the helicopter away until all three drops are in.
    goal("heli", 440, 1230, 120, 100, "-- EXTRACTED --")
    Entity {
        id("handler")
        startPosition(980, 1380)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime { Actions { setEnable("heli", false) } }
            OnSignal("extract") { Actions { setEnable("heli", true) sound("bell") } }
        }
    }

    # ---------- the park ----------
    Entity { id("kiosk1") startPosition(180, 320) color("44403c") solid(true) Type { rectangle(120, 80) } }
    Entity { id("kiosk2") startPosition(600, 360) color("44403c") solid(true) Type { rectangle(100, 80) } }
    Entity { id("fountain") startPosition(420, 620) color("0e7490") solid(true) Type { circle(80) } }
    Entity { id("hedgeA") startPosition(100, 950) color("14532d") solid(true) Type { rectangle(300, 40) } }
    Entity { id("hedgeB") startPosition(600, 950) color("14532d") solid(true) Type { rectangle(300, 40) } }

    # The rose garden in the north-east corner: its gate key is in the shed.
    Entity { id("gardenW") startPosition(720, 1100) color("57534e") solid(true) Type { rectangle(24, 280) } }
    Entity { id("gardenS") startPosition(744, 1100) color("57534e") solid(true) Type { rectangle(120, 24) } }
    Entity { id("gardenGate") startPosition(864, 1100) color("9f1239") Type { rectangle(116, 24) } door("gardenOpen", "clang") }
    Entity { id("shedKey") startPosition(60, 1290) color("facc15") Type { rectangle(40, 40) } key("gardenOpen", "hero", "pickup") }

    # Gravel crunches: every guard hurries for a few seconds.
    Entity { id("gravel1") startPosition(300, 80) color("57534e") Type { rectangle(160, 180) }
        Events { OnCollision("hero") { Actions { sound("thud") flash("a8a29e", 0.3) broadcast("noise") } } } }
    Entity { id("gravel2") startPosition(560, 1110) color("57534e") Type { rectangle(140, 110) }
        Events { OnCollision("hero") { Actions { sound("thud") flash("a8a29e", 0.3) broadcast("noise") } } } }

    # ---------- the watchers ----------
    Entity {
        id("guardFountain")
        startPosition(360, 560)
        color("ef4444")
        Type { circle(26) }
        hostile("hero", "-- STOPPED BY A PARK GUARD --", "lose")
        Movement {
            speed(2)
            Path {
                waypoint(360, 560)
                waypoint(620, 560)
                waypoint(620, 820)
                waypoint(360, 820)
                dwell(0.6)
            }
        }
        Events {
            OnSignal("noise") { Actions { updateSpeed(3.2) delay(3) { Actions { updateSpeed(2) } } } }
        }
    }

    Entity {
        id("guardEast")
        startPosition(760, 30)
        color("ef4444")
        Type { circle(26) }
        hostile("hero", "-- STOPPED BY A PARK GUARD --", "lose")
        Movement {
            speed(2.2)
            Path {
                waypoint(760, 30)
                waypoint(760, 560)
                dwell(1)
            }
        }
        Events {
            OnSignal("noise") { Actions { updateSpeed(3.4) delay(3) { Actions { updateSpeed(2.2) } } } }
        }
    }

    Entity {
        id("guardLawn")
        startPosition(40, 1020)
        color("f97316")
        Type { circle(26) }
        hostile("hero", "-- STOPPED BY A PARK GUARD --", "lose")
        Movement {
            speed(2.6)
            Path {
                waypoint(40, 1020)
                waypoint(900, 1020)
                dwell(1.2)
            }
        }
        Events {
            OnSignal("noise") { Actions { updateSpeed(3.8) delay(3) { Actions { updateSpeed(2.6) } } } }
        }
    }

    # Launched once your cover is blown. Hedges, kiosks and walls stop it.
    Entity {
        id("drone")
        startPosition(300, 1250)
        color("e11d48")
        Type { diamond(40, 40) }
        hostile("hero", "-- THE DRONE TAGGED YOU --", "lose")
        Movement {
            speed(1.3)
            chase("hero")
        }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("burned") {
                Actions {
                    setEnable(true)
                    sound("buzz")
                    burst(14, "e11d48")
                }
            }
            OnElapsedTime(0.6) { repeatCount(-1) Actions { flash("fb7185", 0.2) } }
        }
    }
}
