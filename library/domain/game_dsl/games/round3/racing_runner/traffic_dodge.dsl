Window {
    size(6, 12)
    background("1f2937")
    music("bg_4")
    gravity(0)

    lives(3, "-- WRECKED --", "lose")
    Bar("distance") { initial(0) }
    winAt("distance", 25, "-- MADE IT HOME --", "win")

    Entity { id("laneLeft") startPosition(195, 0) color("4b5563")
        Type { rectangle(6, 1200) } }
    Entity { id("laneRight") startPosition(395, 0) color("4b5563")
        Type { rectangle(6, 1200) } }

    Entity {
        id("car")
        startPosition(270, 80)
        color("22d3ee")
        Type { rectangle(60, 100) }
        Movement { speed(6) Control { movementControl(2) } }
        Events {
            OnCollision("truck") {
                Actions {
                    sound("hurt")
                    shake(8, 0.4)
                    flash("ff0000", 0.3)
                    updateBar("lives", -1)
                }
            }
        }
    }

    Entity {
        id("truck")
        template(true)
        startPosition(0, 0)
        color("f97316")
        Type { rectangle(70, 130) }
        Movement { speed(3) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("car") { Actions { deleteEntity() } }
            OnCollision {
                Actions {
                    updateBar("distance", 1)
                    deleteEntity()
                }
            }
        }
    }

    Wave(10, 1.4) { Actions { spawn("truck", random(30, 500), 1060) } }
    Wave(10, 1.1) { Actions { spawn("truck", random(30, 500), 1060) } }
    Wave(20, 0.8) { Actions { spawn("truck", random(30, 500), 1060) } }
}
