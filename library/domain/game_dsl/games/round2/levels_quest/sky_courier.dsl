# Sky Courier: fly a tall scrolling sky with the joystick, pick up the parcel, deliver it to the tower.
Window {
    size(8, 12)
    worldSize(8, 24)
    cameraFollow("courier")
    background("38bdf8")
    music("bg_1")
    lives(3, "-- THE PARCEL IS LOST --", "lose")
    Bar("parcels") { initial(0) }

    Level {
        joystick("courier", 80, 5)
        wallBox(0, 0, 800, 2400, "0369a1", 12)
        Entity {
            id("courier")
            startPosition(380, 80)
            color("fef08a")
            Type { triangle(44, "up") }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnCollision("parcel") { Actions { sound("pickup") updateBar("parcels", 1) deleteCollider() } }
                OnCollision("tower") { Actions { sound("win2") burst(14, "fbbf24") nextLevel() } }
            }
        }
        Entity { id("parcel") startPosition(600, 1200) color("a16207") Type { rectangle(36, 36) } }
        Entity { id("tower") startPosition(340, 2260) color("f8fafc") Type { rectangle(120, 100) } }
        Entity { id("cloud1") startPosition(100, 600) color("e2e8f0") solid(true) Type { rectangle(300, 40) } }
        Entity { id("cloud2") startPosition(420, 1500) color("e2e8f0") solid(true) Type { rectangle(300, 40) } }
        HostilePatrol("gull1", "courier", "-- A GULL GOT YOU --", 3, "f8fafc") { waypoint(80, 900) waypoint(700, 900) }
        HostilePatrol("gull2", "courier", "-- A GULL GOT YOU --", 3, "f8fafc") { waypoint(700, 1800) waypoint(80, 1800) }
    }

    Level {
        joystick("courier", 80, 5)
        wallBox(0, 0, 800, 2400, "0f172a", 12)
        Entity {
            id("courier")
            startPosition(380, 80)
            color("fef08a")
            Type { triangle(44, "up") }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnCollision("parcel") { Actions { sound("pickup") updateBar("parcels", 1) deleteCollider() } }
                OnCollision("storm") { Actions { sound("hurt") updateBar("lives", -1) setPosition("courier", 380, 80, 0) } }
                OnCollision("tower") { Actions { sound("win") shake(6, 0.4) gameOver("-- DELIVERED IN THE STORM --") } }
            }
        }
        Entity { id("parcel") startPosition(120, 1100) color("a16207") Type { rectangle(36, 36) } }
        Entity { id("tower") startPosition(100, 2260) color("f8fafc") Type { rectangle(120, 100) } }
        Entity { id("ledge1") startPosition(0, 500) color("1e293b") solid(true) Type { rectangle(500, 40) } }
        Entity { id("ledge2") startPosition(300, 1000) color("1e293b") solid(true) Type { rectangle(500, 40) } }
        Entity { id("ledge3") startPosition(0, 1500) color("1e293b") solid(true) Type { rectangle(500, 40) } }
        Entity { id("storm") startPosition(600, 700) color("facc15") Type { star(6, 40, 18) }
            Movement { speed(2.5) Path { waypoint(600, 700) waypoint(600, 1300) } } }
        Entity { id("stormChaser") startPosition(700, 2000) color("a855f7") Type { circle(26) }
            Movement { speed(1.5) chase("courier") }
            Events { OnCollision("courier") { Actions { sound("hurt") updateBar("lives", -1) setPosition("courier", 380, 80, 0) setPosition("stormChaser", 700, 2000, 0) } } } }
        HostilePatrol("hawk", "courier", "-- A HAWK GOT YOU --", 4, "92400e") { waypoint(80, 1250) waypoint(700, 1250) dwell(0.5) }
    }
}
