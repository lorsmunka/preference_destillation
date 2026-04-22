- 3 core contributions: input projectionr education, ouput proecjtion reudaction and loss / temp annealing combo
- output reduction seems trivial, it's not, you need to take into considartion disitllation type. Is the distirbution sharp? Structured data? Do you need to learn next token or the whole distirbution? Both?
- inptu reduccation is actually mroe tirival, you need to gather enough data to have a representative text corpus of whaty ou need to tokneize, you tokenize it, add the ouput tokens in union, and oyu have the input projection layer
- I ran many exeirments, including scalings, it seems that even with great reduction of param count on speicfic domains distilaltion is possible, furthermore very plausable. On different domsins you can combine startegies, in some strucutred sharp output data mybe pure CE and no auxilery tokens could be enogh, but if you use the CE + KL annealing combo for free form egenration, plus the well established temp annealing you can get the benifit of poth, as per some exmaple runs prove it (this needs further standardised teesting)
- also i rana  shitton of expeirments, there is even tooling to comapre them. we need to point out what we found

512 mb / batch vagy exmaple?
Kép minden domain generálásról