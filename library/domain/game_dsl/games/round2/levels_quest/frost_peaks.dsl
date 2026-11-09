# Frost Peaks: a side-view climb in three levels; jump up the ledges to the flag, mind the icicles.
Window {
    size(8, 12)
    background("bae6fd")
    gravity(700)
    music("bg_nature")
    lives(3, "-- FROZEN SOLID --", "lose")

    Level {
        Entity { id("hiker") startPosition(60, 120) color("1d4ed8") Type { rectangle(36, 56) }
            Movement { speed(5) Control { movementControl(2) } }
            Events {
                OnCollision("flag") { Actions { sound("win2") burst(14, "fbbf24") nextLevel() } }
                OnCollision("icicle") { Actions { sound("hurt") updateBar("lives", -1) flash("ff0000", 0.3) setPosition("hiker", 60, 120, 0) } }
            } }
        Entity { id("ground") startPosition(0, 0) color("e0f2fe") solid(true) Type { rectangle(800, 100) } }
        Entity { id("ledge1") startPosition(180, 360) color("94a3b8") oneWay(true) Type { rectangle(200, 20) } }
        Entity { id("ledge2") startPosition(480, 620) color("94a3b8") oneWay(true) Type { rectangle(200, 20) } }
        Entity { id("ledge3") startPosition(160, 880) color("94a3b8") oneWay(true) Type { rectangle(200, 20) } }
        Entity { id("flag") startPosition(200, 920) color("ef4444") Type { triangle(50, "right") } }
        Entity { id("jumpL") startPosition(40, 20) color("1e3a8a") Type { rectangle(200, 60) }
            Events { OnTapped("jumpL") { Actions { setVelocity("hiker", -180, 620) } } } }
        Entity { id("jumpR") startPosition(560, 20) color("1e3a8a") Type { rectangle(200, 60) }
            Events { OnTapped("jumpR") { Actions { setVelocity("hiker", 180, 620) } } } }
        Entity { id("icicle") startPosition(520, 1100) color("67e8f9") gravity(0) Type { triangle(36, "down") }
            Movement { speed(4) AutoMove { direction(0, -1) } }
            Events { OnCollision { Actions { setPosition("icicle", 520, 1140, 0) } } } }
    }

    Level {
        Entity { id("hiker") startPosition(60, 120) color("1d4ed8") Type { rectangle(36, 56) }
            Movement { speed(5) Control { movementControl(2) } }
            Events {
                OnCollision("flag") { Actions { sound("win2") burst(14, "fbbf24") nextLevel() } }
                OnCollision("icicle") { Actions { sound("hurt") updateBar("lives", -1) flash("ff0000", 0.3) setPosition("hiker", 60, 120, 0) } }
                OnCollision("snowball") { Actions { sound("thud") updateBar("lives", -1) setPosition("hiker", 60, 120, 0) } }
            } }
        Entity { id("ground") startPosition(0, 0) color("e0f2fe") solid(true) Type { rectangle(800, 100) } }
        Entity { id("ledge1") startPosition(300, 330) color("94a3b8") oneWay(true) Type { rectangle(180, 20) } }
        Entity { id("ledge2") startPosition(80, 560) color("94a3b8") oneWay(true) Type { rectangle(180, 20) } }
        Entity { id("ledge3") startPosition(420, 790) color("94a3b8") oneWay(true) Type { rectangle(180, 20) } }
        Entity { id("ledge4") startPosition(600, 1000) color("94a3b8") oneWay(true) Type { rectangle(160, 20) } }
        Entity { id("flag") startPosition(660, 1040) color("ef4444") Type { triangle(50, "right") } }
        Entity { id("jumpL") startPosition(40, 20) color("1e3a8a") Type { rectangle(200, 60) }
            Events { OnTapped("jumpL") { Actions { setVelocity("hiker", -200, 640) } } } }
        Entity { id("jumpR") startPosition(560, 20) color("1e3a8a") Type { rectangle(200, 60) }
            Events { OnTapped("jumpR") { Actions { setVelocity("hiker", 200, 640) } } } }
        Entity { id("icicle") startPosition(300, 1100) color("67e8f9") gravity(0) Type { triangle(36, "down") }
            Movement { speed(5) AutoMove { direction(0, -1) } }
            Events { OnCollision { Actions { setPosition("icicle", 300, 1140, 0) } } } }
        Entity { id("snowball") startPosition(700, 130) color("f8fafc") gravity(0) Type { circle(24) }
            Movement { speed(3) AutoMove { direction(-1, 0) } }
            Events { OnCollision { Actions { setPosition("snowball", 700, 130, 0) } } } }
    }

    Level {
        Entity { id("hiker") startPosition(60, 120) color("1d4ed8") Type { rectangle(36, 56) }
            Movement { speed(5) Control { movementControl(2) } }
            Events {
                OnCollision("summit") { Actions { sound("win") shake(6, 0.4) gameOver("-- YOU REACHED THE SUMMIT --") } }
                OnCollision("icicle") { Actions { sound("hurt") updateBar("lives", -1) flash("ff0000", 0.3) setPosition("hiker", 60, 120, 0) } }
                OnCollision("snowball") { Actions { sound("thud") updateBar("lives", -1) setPosition("hiker", 60, 120, 0) } }
            } }
        Entity { id("ground") startPosition(0, 0) color("e0f2fe") solid(true) Type { rectangle(800, 100) } }
        Entity { id("ledge1") startPosition(520, 320) color("94a3b8") oneWay(true) Type { rectangle(180, 20) } }
        Entity { id("ledge2") startPosition(220, 520) color("94a3b8") oneWay(true) Type { rectangle(160, 20) } }
        Entity { id("ledge3") startPosition(20, 720) color("94a3b8") oneWay(true) Type { rectangle(160, 20) } }
        Entity { id("ledge4") startPosition(300, 920) color("94a3b8") oneWay(true) Type { rectangle(160, 20) } }
        Entity { id("ledge5") startPosition(580, 1080) color("94a3b8") oneWay(true) Type { rectangle(160, 20) } }
        Entity { id("summit") startPosition(630, 1120) color("fbbf24") Type { star(5, 36, 16) } }
        Entity { id("jumpL") startPosition(40, 20) color("1e3a8a") Type { rectangle(200, 60) }
            Events { OnTapped("jumpL") { Actions { setVelocity("hiker", -220, 660) } } } }
        Entity { id("jumpR") startPosition(560, 20) color("1e3a8a") Type { rectangle(200, 60) }
            Events { OnTapped("jumpR") { Actions { setVelocity("hiker", 220, 660) } } } }
        Entity { id("icicle") startPosition(200, 1100) color("67e8f9") gravity(0) Type { triangle(36, "down") }
            Movement { speed(6) AutoMove { direction(0, -1) } }
            Events { OnCollision { Actions { setPosition("icicle", 200, 1140, 0) } } } }
        Entity { id("icicle2") startPosition(640, 1100) color("a5f3fc") gravity(0) Type { triangle(36, "down") }
            Movement { speed(4) AutoMove { direction(0, -1) } }
            Events { OnCollision { Actions { setPosition("icicle2", 640, 1140, 0) } } } }
        Entity { id("snowball") startPosition(700, 130) color("f8fafc") gravity(0) Type { circle(24) }
            Movement { speed(4) AutoMove { direction(-1, 0) } }
            Events { OnCollision { Actions { setPosition("snowball", 700, 130, 0) } } } }
    }
}
