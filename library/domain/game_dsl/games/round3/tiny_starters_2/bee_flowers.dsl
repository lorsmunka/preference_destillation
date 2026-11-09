Window {
    size(9, 9)
    background("bef264")
    music("bg_nature")
    winAt(6, "THE HIVE IS FULL", "win")
    Entity {
        id("bee")
        startPosition(100, 100)
        color("facc15")
        Type { circle(24) }
        Movement { speed(5) Control { movementControl(0) } }
    }
    Entity {
        id("flowerA")
        startPosition(200, 650)
        color("ec4899")
        Type { star(5, 36, 16) }
        Events {
            OnCollision("bee") { Actions { sound("sparkle") updateBar(1) setPosition("flowerA", random(60, 800), random(100, 800), 0) } }
        }
    }
    Entity {
        id("flowerB")
        startPosition(650, 450)
        color("a855f7")
        Type { star(6, 34, 14) }
        Events {
            OnCollision("bee") { Actions { sound("sparkle") updateBar(1) setPosition("flowerB", random(60, 800), random(100, 800), 0) } }
        }
    }
    Entity {
        id("spider")
        startPosition(450, 100)
        color("1f2937")
        Type { circle(30) }
        Movement { speed(3) Path { waypoint(450, 100) waypoint(450, 800) } }
        Events {
            OnCollision("bee") { Actions { sound("hurt") gameOver("THE SPIDER GOT THE BEE") } }
        }
    }
    Entity {
        id("hive")
        startPosition(20, 820)
        color("92400e")
        Type { text("HIVE", 40) }
    }
}
