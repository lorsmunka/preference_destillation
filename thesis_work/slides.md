# Prezentáció — 15 perc, ~40 slide

Időzítés: átlag ~22 mp/slide. A fókusz a 17–32. slide-okon legyen (újdonság + post generation eredmény). A bevezető slide-okon (3–9) gyorsan haladj át.

Slide-okon csak támogatás van — bullet, kép, rövid szöveg. Mondatok élőszóban.

---

## Slide 1 — Címlap

**Domain-specifikus LLM desztillációs pipeline**

Laczkó Örs
Témavezető: [név]
TDK 2026

---

## Slide 2 — Cím

> *Csak a cím, nagyban, középen.*
> *„Most végigmegyek rajta szavanként."*

**Domain-specifikus LLM desztillációs pipeline**

---

## Slide 3 — LLM

**LLM = Large Language Model**

- ChatGPT, Claude, Gemini mögött
- Szöveget generál, klasszifikál, strukturált választ ad

> *Vizuál: ismert chatbot logók vagy egy egyszerű "input → LLM → output" ábra*

---

## Slide 4 — Open-weight LLM

- Lokálisan futtatható modell, nem csak API
- **Logit eloszlás** kinyerhető minden lépésnél
- Ez a desztilláció alapfeltétele

Tanár modell: **Gemma 3 4B** (Google)

---

## Slide 5 — Desztilláció

> *Kép: kőolaj-desztillációs torony*
> *„Pontosan ugyanaz az elv."*

Nagyobból kisebb, tisztább, célzottabb.

---

## Slide 6 — Domain-specifikus

- A modellnek nem kell mindent tudnia
- Csak a feladathoz szükséges tudás → **kis modell, fókuszált**
- Cserébe: olcsóbb, gyorsabb, on-device futtatható

---

## Slide 7 — Domain 1: Reddit sentiment

Bemenet: `"F**k you and your family"`

```json
{
  "tone": "aggressive",
  "sentiment": "negative",
  "safety": "harmful",
  "toxicity": "toxic"
}
```

---

## Slide 8 — Domain 2: Math word problem

```
Problem: "Emma has 24 stickers and buys 18 more..."
A=24
B=18
C=A+B=42
Solution: 42;
```

---

## Slide 9 — Domain 3: Post generation

Bemenet: `"Same thing happened to me last week, support was useless."`

Kimenet: *„I'm starting to think Reddit is actively targeting me…"* `<end>`

---

## Slide 10 — Pipeline

**End-to-end rendszer** a desztillációhoz

- Adat előkészítés
- Tanár-adatgenerálás
- Modell konfiguráció
- Tanítás
- Kiértékelés

---

## Slide 11 — Nem framework

- Nem `new Domain()` típusú API
- **Kódbázis-kiterjesztés**: meglévő kódot módosítasz
- Cserébe: maximális rugalmasság a kutatáshoz

---

## Slide 12 — Domain definíció

- Bemeneti és kimeneti tokenek listája
- Prompt sablon
- Stop token
- Adatgenerálás konfigurálása

---

## Slide 13 — Adatgenerálás

- Tanár modell (Gemma 3 4B) végigfut a példákon
- **Minden lépésnél**: nyers logit-ok mentése a watched tokenekre
- JSONL batchekben tárolva (32 példa / batch)

---

## Slide 14 — Modell + tanítás

- Saját transformer (RMSNorm, RoPE, GeGLU)
- Konfigurálható: hidden dim, rétegek, head-ek
- KL + CE veszteség, AdamW, cosine schedule

---

## Slide 15 — Konfigurálható tanítás

> *Vizuál: egy valós config objektum kódrészlet*

```python
EPOCH_COUNT = 3
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
KL_RATIO_START = 0.99
KL_RATIO_END = 0.50
DISTILLATION_TEMPERATURE_START = 5.0
DISTILLATION_TEMPERATURE_END = 3.0
```

---

## Slide 16 — Experiment tagek

- Minden run külön mappa
- Konfiguráció + log + checkpoint együtt
- Skálázható kísérletezés (>100 run)

---

## Slide 17 — Újdonságok

> *Szekciócím slide, nagyban.*

**A projekt újdonságai**

---

## Slide 18 — Annealing kombináció

- **Loss annealing**: kutatott (nem LLM-eken)
- **Temperature annealing**: kutatott (LLM-eken)
- **Együtt**, ugyanabban a pipeline-ban: nem találtam publikációt

---

## Slide 19 — Input + output redukció

- Vocabulary redukció létezik (orosz nyelvi modellek [10])
- De ennyire **explicit** és **task-specifikus**: nem találtam
- Crude technika — épp ezért gyakorlatias

---

## Slide 20 — Output redukció elv

- Logikus első lépés: minek 262 144 token, ha 30 elég?
- **Csapda**: a desztilláció a teljes eloszlásból tanul
- Túl szűk output → elveszik a "dark knowledge"

---

## Slide 21 — Szükséges tokenek (JSON)

- Kapcsos zárójelek, idézőjelek, kettőspont, vessző
- Mezőnevek: `tone`, `sentiment`, `safety`, `toxicity`
- Címke értékek: `aggressive`, `neutral`, `toxic`, …

**~27 token**

---

## Slide 22 — Szükséges tokenek (math)

- Változók: `A`, `B`, `C`, …
- Szimbólumok: `=`, `+`, `-`, `*`, `/`, `;`
- Számjegyek: `0–9`

**~30 token**

---

## Slide 23 — Auxiliary tokenek

| | Szükséges | + Auxiliary | Teljes Gemma |
|---|---:|---:|---:|
| Sentiment | 27 | **525** | 262 144 |
| Math | 30 | **528** | 262 144 |

**99,8% redukció** — és gazdagabb soft target

---

## Slide 24 — Strukturált ≠ KL info

- A legtöbb auto-generált domain strukturált
- Strukturált kimenetnél a tanár eloszlása **éles**
- A különbség több metrikán a futások közötti szórással összemérhető

> *Itt válik érdekessé a free-form domain.*

---

## Slide 25 — Free-form domain

**Post generation**: kommentből generálj posztot

- Több plauzibilis folytatás
- Lágyabb tanár eloszlás
- Itt számít, hogy KL-t vagy CE-t használsz

---

## Slide 26 — Méretek

| | Bemeneti vocab | Kimeneti vocab |
|---|---:|---:|
| Sentiment | 68 328 | 525 |
| Math | 1 282 | 528 |
| **Post generation** | **72 133** | **26 659** |

Output redukció: 89,8% (még mindig jelentős)

---

## Slide 27 — Pure KL

- **Jól követi a tanár top-k eloszlását**
- Variációhoz, sampling-hoz fontos
- Temperature annealing: lágy → éles átmenet

---

## Slide 28 — Pure CE

- **Jól eltalálja a következő tokent**
- De gyengébben őrzi a tanár eloszlás-szerkezetét

---

## Slide 29 — Best of both worlds

**KL/CE + temperature annealing együtt**

- Magabiztos next-token predikció
- És megőrzött tanár top-k eloszlás
- Lépésenként hangolható kompromisszum

---

## Slide 30 — Célhoz szabható

A schedule nemcsak domainfüggő — **célfüggő** is.

- Determinisztikus next-token? → CE-felé
- Kreatív, sokszínű generálás? → KL-felé
- Vegyes? → annealing

---

## Slide 31 — Top-k metrikák

- **Top-20 cél-token találat**: a tanár cél tokenje benne van-e a student top-20-ban
- **Teacher-student top-20 overlap**: mennyire egyezik a két halmaz
- **Átlagos cél-token rang**: hol helyezkedik el a cél a student eloszlásában (kisebb = jobb)

---

## Slide 32 — Eredmény

| Stratégia | Top-20 hit | Top-20 overlap | Átl. rang |
|---|---:|---:|---:|
| Pure CE | 67,55% | 26,69% | 1068,92 |
| Pure KL | 64,42% | 37,31% | 464,70 |
| **KL/CE + temp anneal** | **69,08%** | 34,61% | 486,31 |

---

## Slide 33 — Felhasználás 1

**Offline matek-megoldó telefonon**

- Természetes nyelvű magyarázattal
- API nélkül, offline

---

## Slide 34 — Felhasználás 2

**Auto-moderáció kliens oldalon**

- Komment ellenőrzése küldés előtt
- Magas konfidenciánál visszadobás már a telefonon

---

## Slide 35 — Felhasználás 3

**Személyre szabott email-író asszisztens**

- A te stílusodra finomhangolva
- Töredék költséggel egy nagy modellhez képest

---

## Slide 36 — Méret + sebesség (1)

> *Vizuál: 4. ábra (skálázási görbe) vagy 2. ábra (vocab redukció)*

- 5M–129M paraméter vs. 4,3 milliárd
- **34×–860× kisebb**

---

## Slide 37 — Méret + sebesség (2)

> *Vizuál: 3. ábra (paraméterallokáció no reduction / input only / input+output)*

- 80h/5L body, sentiment: 42,72M → **6,02M**
- Math: 42,72M → **0,66M**
- A felszabadult paraméterek a transformer body-ra mehetnek

---

## Slide 38 — Összegzés

**Saját LLM nagyobb modellből, sok domainen**

1. Definiáld a domaint
2. Generálj tanító adatot
3. Konfiguráld a transformer modellt
4. Konfiguráld a tanítási regimet

---

## Slide 39 — Tooling

- Run-onkénti log + checkpoint
- Kísérlet-összehasonlítás, skálázási ábrák
- Top-k elemzés, confusion matrix
- Graceful shutdown, queue futtatás

---

## Slide 40 — Köszönöm

**Köszönöm a figyelmet!**

Kérdések?
