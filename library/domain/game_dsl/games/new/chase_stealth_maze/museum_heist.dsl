Window {
    size(9, 16)
    background("1a1423")
    music("bg_3")
    gravity(0)

    lives(3, "-- LOCKDOWN: THREE ALARMS --", "lose")

    wallBox(0, 0, 900, 1600, "3b2f4a", 20)

    # ---------- the thief ----------
    Entity {
        id("hero")
        startPosition(60, 60)
        color("a3e635")
        Type { rectangle(44, 44) }
        Movement {
            speed(5)
            Control { movementControl(0) }
        }
        Events {
            OnSignal("cardSwiped") {
                Actions {
                    flash("67e8f9", 0.3)
                    burst(12, "67e8f9")
                }
            }
            OnSignal("gemTaken") {
                Actions {
                    flash("f0abfc", 0.4)
                    shake(4, 0.4)
                    burst(20, "f0abfc")
                }
            }
            OnSignal("alarm") {
                Actions {
                    flash("ff0000", 0.4)
                    shake(6, 0.4)
                }
            }
        }
    }

    Entity {
        id("status")
        startPosition(40, 1540)
        color("e9d5ff")
        Type { text("Find the keycard in the lobby", 24) }
        Events {
            OnSignal("cardSwiped") { Actions { setText("status", "Gallery open. The lasers blink: cross in the gaps") } }
            OnSignal("alarm") { Actions { setText("status", "ALARM! The guard dog is loose") } }
            OnSignal("gemTaken") { Actions { setText("status", "Got the gem! Back down to the van") } }
        }
    }

    # ---------- lobby ----------
    Entity { id("desk") startPosition(300, 100) color("57534e") solid(true) Type { rectangle(200, 60) } }

    Entity {
        id("keycard")
        startPosition(800, 320)
        color("22d3ee")
        Type { rectangle(40, 28) }
        key("cardSwiped", "hero", "pickup")
    }

    HostilePatrol("lobbyGuard", "hero", "-- THE LOBBY GUARD SAW YOU --", 2.6, "f97316") {
        waypoint(30, 240)
        waypoint(820, 240)
        dwell(1.2)
    }

    Entity { id("lobbyWallL") startPosition(20, 420) color("3b2f4a") solid(true) Type { rectangle(360, 30) } }
    Entity { id("lobbyWallR") startPosition(520, 420) color("3b2f4a") solid(true) Type { rectangle(360, 30) } }
    Entity {
        id("securityDoor")
        startPosition(380, 420)
        color("0891b2")
        Type { rectangle(140, 30) }
        door("cardSwiped", "clang")
    }

    # The getaway van only pulls up once the gem is in the bag.
    goal("van", 720, 40, 140, 90, "-- CLEAN GETAWAY --")

    # ---------- gallery ----------
    Entity { id("case1") startPosition(150, 540) color("312e81") solid(true) Type { rectangle(120, 80) } }
    Entity { id("case2") startPosition(630, 540) color("312e81") solid(true) Type { rectangle(120, 80) } }
    Entity { id("case3") startPosition(380, 770) color("312e81") solid(true) Type { rectangle(140, 80) } }

    Entity {
        id("laserA")
        startPosition(20, 680)
        color("ef4444")
        Type { rectangle(860, 8) }
        Events {
            OnSignal("beamsA") { Actions { setEnable(true) } }
            OnSignal("beamsB") { Actions { setEnable(false) } }
            OnCollision("hero") {
                Actions {
                    sound("buzz")
                    updateBar("lives", -1)
                    burst("hero", 10, "ef4444")
                    broadcast("alarm")
                }
            }
        }
    }

    Entity {
        id("laserB")
        startPosition(20, 930)
        color("f43f5e")
        Type { rectangle(860, 8) }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("beamsB") { Actions { setEnable(true) } }
            OnSignal("beamsA") { Actions { setEnable(false) } }
            OnCollision("hero") {
                Actions {
                    sound("buzz")
                    updateBar("lives", -1)
                    burst("hero", 10, "f43f5e")
                    broadcast("alarm")
                }
            }
        }
    }

    # The two beams take turns, 1.6 s each.
    Entity {
        id("laserPanel")
        startPosition(850, 460)
        color("7f1d1d")
        Type { rectangle(20, 30) }
        Events {
            OnElapsedTime(1.6) { Actions { broadcast("beamsB") flash("f43f5e", 0.2) } }
            OnElapsedTime(3.2) {
                repeatCount(-1)
                Actions {
                    broadcast("beamsA")
                    flash("ef4444", 0.2)
                    delay(1.6) { Actions { broadcast("beamsB") flash("f43f5e", 0.2) } }
                }
            }
        }
    }

    HostilePatrol("galleryGuard", "hero", "-- THE NIGHT GUARD GRABBED YOU --", 2.2, "ef4444") {
        waypoint(300, 710)
        waypoint(560, 710)
        waypoint(560, 870)
        waypoint(300, 870)
        dwell(0.5)
    }

    # Asleep in the corner until an alarm or the stolen gem wakes it.
    Entity {
        id("dog")
        startPosition(820, 950)
        color("a8a29e")
        Type { circle(22) }
        hostile("hero", "-- THE GUARD DOG GOT YOU --", "lose")
        Movement {
            speed(2.3)
            chase("hero")
        }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("alarm") {
                Actions {
                    setEnable(true)
                    sound("bump")
                    burst(8, "a8a29e")
                }
            }
            OnSignal("gemTaken") {
                Actions {
                    setEnable(true)
                    updateSpeed(2.8)
                }
            }
        }
    }

    Entity { id("galleryWall") startPosition(200, 1000) color("3b2f4a") solid(true) Type { rectangle(680, 30) } }

    # ---------- vault ----------
    Entity { id("pillarL") startPosition(120, 1200) color("4c1d95") solid(true) Type { rectangle(60, 200) } }
    Entity { id("pillarR") startPosition(720, 1200) color("4c1d95") solid(true) Type { rectangle(60, 200) } }
    Entity { id("plinth") startPosition(400, 1290) color("a78bfa") solid(true) Type { rectangle(100, 40) } }

    Entity {
        id("gem")
        startPosition(424, 1340)
        color("f0abfc")
        Type { star(6, 26) }
        key("gemTaken", "hero", "sparkle")
        Events {
            OnElapsedTime(1.2) { repeatCount(-1) Actions { burst(3, "f0abfc") } }
        }
    }

    HostilePatrol("vaultGuard", "hero", "-- CAUGHT IN THE VAULT --", 2.4, "dc2626") {
        waypoint(300, 1180)
        waypoint(560, 1180)
        waypoint(560, 1440)
        waypoint(300, 1440)
        dwell(0.8)
    }

    # Security system: parks the van at the start, calls it in when the gem moves.
    Entity {
        id("vaultSystem")
        startPosition(860, 1560)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime { Actions { setEnable("van", false) } }
            OnSignal("gemTaken") {
                Actions {
                    setEnable("van", true)
                    sound("bell")
                }
            }
        }
    }
}
