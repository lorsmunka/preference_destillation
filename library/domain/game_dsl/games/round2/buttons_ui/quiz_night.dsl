# Three questions, three answers each. Wrong answers cost a life.
Window {
    size(8, 10)
    background("1e1b4b")
    music("bg_ambient")
    lives(3, "OUT OF GUESSES", "lose")
    Bar("score") { initial(0) }

    Level {
        Entity {
            id("question")
            startPosition(60, 820)
            color("e0e7ff")
            Type { text("1/3  How many legs has a spider?", 40) }
        }
        Entity {
            id("feedback")
            startPosition(60, 90)
            color("a5b4fc")
            Type { text("Tap an answer", 36) }
        }
        Entity {
            id("answerSix")
            startPosition(160, 600)
            color("fbbf24")
            Type { text("A) Six", 56) }
            Events {
                OnTapped("answerSix") {
                    Actions {
                        sound("buzz")
                        updateBar("lives", -1)
                        setText("feedback", "Count again")
                        setEnable(false)
                    }
                }
            }
        }
        Entity {
            id("answerEight")
            startPosition(160, 440)
            color("34d399")
            Type { text("B) Eight", 56) }
            Events {
                OnTapped("answerEight") {
                    Actions {
                        sound("coin")
                        updateBar("score", 1)
                        nextLevel()
                    }
                }
            }
        }
        Entity {
            id("answerTen")
            startPosition(160, 280)
            color("f87171")
            Type { text("C) Ten", 56) }
            Events {
                OnTapped("answerTen") {
                    Actions {
                        sound("buzz")
                        updateBar("lives", -1)
                        setText("feedback", "Too many")
                        setEnable(false)
                    }
                }
            }
        }
    }

    Level {
        Entity {
            id("question")
            startPosition(60, 820)
            color("e0e7ff")
            Type { text("2/3  Which planet is red?", 40) }
        }
        Entity {
            id("feedback")
            startPosition(60, 90)
            color("a5b4fc")
            Type { text("Wrong guesses cost a life", 32) }
        }
        Entity {
            id("answerVenus")
            startPosition(160, 600)
            color("fbbf24")
            Type { text("A) Venus", 56) }
            Events {
                OnTapped("answerVenus") {
                    Actions {
                        sound("buzz")
                        updateBar("lives", -1)
                        setText("feedback", "Venus is pale yellow")
                        setVisible(false)
                    }
                }
            }
        }
        Entity {
            id("answerMars")
            startPosition(160, 440)
            color("34d399")
            Type { text("B) Mars", 56) }
            Events {
                OnTapped("answerMars") {
                    Actions {
                        sound("coin")
                        updateBar("score", 1)
                        nextLevel()
                    }
                }
            }
        }
        Entity {
            id("answerNeptune")
            startPosition(160, 280)
            color("f87171")
            Type { text("C) Neptune", 56) }
            Events {
                OnTapped("answerNeptune") {
                    Actions {
                        sound("buzz")
                        updateBar("lives", -1)
                        setText("feedback", "Neptune is blue")
                        setEnable(false)
                    }
                }
            }
        }
    }

    Level {
        Entity {
            id("question")
            startPosition(60, 820)
            color("e0e7ff")
            Type { text("3/3  What is 7 x 6?", 40) }
        }
        Entity {
            id("feedback")
            startPosition(60, 90)
            color("a5b4fc")
            Type { text("Last one!", 36) }
        }
        Entity {
            id("answer36")
            startPosition(160, 600)
            color("fbbf24")
            Type { text("A) 36", 56) }
            Events {
                OnTapped("answer36") {
                    Actions {
                        sound("buzz")
                        updateBar("lives", -1)
                        setText("feedback", "That is 6 x 6")
                        setEnable(false)
                    }
                }
            }
        }
        Entity {
            id("answer42")
            startPosition(160, 440)
            color("34d399")
            Type { text("B) 42", 56) }
            Events {
                OnTapped("answer42") {
                    Actions {
                        sound("win")
                        updateBar("score", 1)
                        shake(4, 0.3)
                        gameOver("QUIZ MASTER")
                    }
                }
            }
        }
        Entity {
            id("answer48")
            startPosition(160, 280)
            color("f87171")
            Type { text("C) 48", 56) }
            Events {
                OnTapped("answer48") {
                    Actions {
                        sound("buzz")
                        updateBar("lives", -1)
                        setText("feedback", "That is 6 x 8")
                        setEnable(false)
                    }
                }
            }
        }
    }
}
