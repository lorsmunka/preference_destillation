import pptxgen from "pptxgenjs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const IMG = (name) => path.join(__dirname, "grafikonok", name);

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Domain-specifikus LLM desztillációs pipeline";
pres.author = "Laczkó Örs";

const W = 13.333;
const H = 7.5;

const NAVY = "1E2761";
const ICE = "CADCFC";
const WHITE = "FFFFFF";
const BODY = "333333";
const MUTED = "707070";
const ACCENT = "F96167";

const FH = "Cambria";
const FB = "Calibri";
const FM = "Consolas";

function title(slide, text) {
  slide.addText(text, {
    x: 0.6, y: 0.45, w: W - 1.2, h: 0.9,
    fontSize: 32, bold: true, fontFace: FH, color: NAVY,
    align: "left", valign: "middle"
  });
}

function subtitle(slide, text) {
  slide.addText(text, {
    x: 0.6, y: 1.2, w: W - 1.2, h: 0.5,
    fontSize: 18, italic: true, fontFace: FB, color: MUTED,
    align: "left", valign: "top"
  });
}

function bullets(slide, items, opts = {}) {
  const { x = 0.7, y = 1.9, w = W - 1.4, h = 4.8, fontSize = 20 } = opts;
  slide.addText(
    items.map(t => {
      if (typeof t === "string") {
        return { text: t, options: { bullet: { code: "25CF" }, paraSpaceAfter: 12 } };
      }
      return { text: t.text, options: { bullet: { code: "25CF" }, paraSpaceAfter: 10, indentLevel: t.level || 0 } };
    }),
    { x, y, w, h, fontSize, fontFace: FB, color: BODY, valign: "top" }
  );
}

function darkBg(slide) {
  slide.background = { color: NAVY };
}

const slides = [];
function add(fn) { slides.push(fn); }

// 1 — Title
add(() => {
  const s = pres.addSlide();
  darkBg(s);
  s.addText("Domain-specifikus", {
    x: 0.8, y: 2.3, w: W - 1.6, h: 0.9,
    fontSize: 44, bold: true, fontFace: FH, color: WHITE, align: "left"
  });
  s.addText("LLM desztillációs pipeline", {
    x: 0.8, y: 3.1, w: W - 1.6, h: 0.9,
    fontSize: 44, bold: true, fontFace: FH, color: ICE, align: "left"
  });
  s.addShape(pres.ShapeType.line, {
    x: 0.8, y: 4.3, w: 1.2, h: 0,
    line: { color: ACCENT, width: 2 }
  });
  s.addText("Laczkó Örs", {
    x: 0.8, y: 4.5, w: W - 1.6, h: 0.5,
    fontSize: 22, fontFace: FB, color: WHITE, align: "left"
  });
  s.addText("Témavezető: Szénási Sándor, egyetemi tanár", {
    x: 0.8, y: 5.05, w: W - 1.6, h: 0.5,
    fontSize: 18, fontFace: FB, color: ICE, align: "left"
  });
  s.addText("TDK · 2026", {
    x: 0.8, y: 6.5, w: W - 1.6, h: 0.5,
    fontSize: 18, italic: true, fontFace: FB, color: ICE, align: "left"
  });
});

// 2 — Cím (large title)
add(() => {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  s.addText([
    { text: "Domain-specifikus\n", options: { color: NAVY, bold: true } },
    { text: "LLM desztillációs pipeline", options: { color: NAVY, bold: true } }
  ], {
    x: 0.6, y: 2.4, w: W - 1.2, h: 2.5,
    fontSize: 54, fontFace: FH, align: "center", valign: "middle"
  });
  s.addText("Rövid szavankénti magyarázat", {
    x: 0.6, y: 5.5, w: W - 1.2, h: 0.5,
    fontSize: 18, italic: true, fontFace: FB, color: MUTED, align: "center"
  });
});

// 3 — LLM
add(() => {
  const s = pres.addSlide();
  title(s, "LLM");
  subtitle(s, "Large Language Model");
  bullets(s, [
    "A ChatGPT, Claude, Gemini és társai mögött álló modellek",
    "Szöveget generálnak, klasszifikálnak, strukturált választ adnak",
    "Általános célú, nagy erőforrásigényű modellek"
  ]);
});

// 4 — Open-weight LLM
add(() => {
  const s = pres.addSlide();
  title(s, "Open-weight LLM");
  bullets(s, [
    "Lokálisan futtatható modell, nem csak API",
    "Logit eloszlás kinyerhető minden generálási lépésnél",
    "Ez a desztilláció alapfeltétele",
    "Tanár modell ebben a projektben: Gemma 3 4B (Google)",
    "Tokenizer csere nélkül jobb hardveren nagyobb Gemma 3 modell is használható"
  ]);
});

// 5 — Desztilláció: két oldal egy slide-on
add(() => {
  const s = pres.addSlide();
  title(s, "Desztilláció");

  const colY = 1.9;
  const colH = 5.0;
  const leftX = 0.6;
  const rightX = 6.93;
  const colW = 5.8;

  // Left column: Kőolaj
  s.addText("Kőolaj", {
    x: leftX, y: colY, w: colW, h: 0.7,
    fontSize: 26, bold: true, fontFace: FH, color: NAVY, align: "left"
  });
  s.addText([
    { text: "Cél: ", options: { bold: true, color: MUTED } },
    { text: "hasznos összetevők kinyerése.", options: { color: BODY } }
  ], {
    x: leftX, y: colY + 0.9, w: colW, h: 1.2,
    fontSize: 20, fontFace: FB, valign: "top"
  });
  s.addText("Pl. benzin, dízel, kerozin külön frakciókba.", {
    x: leftX, y: colY + 2.3, w: colW, h: 1.2,
    fontSize: 18, italic: true, fontFace: FB, color: MUTED, valign: "top"
  });

  // Right column: LLM
  s.addText("LLM", {
    x: rightX, y: colY, w: colW, h: 0.7,
    fontSize: 26, bold: true, fontFace: FH, color: NAVY, align: "left"
  });
  s.addText([
    { text: "Cél: ", options: { bold: true, color: MUTED } },
    { text: "egy hasznos szegmens kinyerése a tanár modellből.", options: { color: BODY } }
  ], {
    x: rightX, y: colY + 0.9, w: colW, h: 1.2,
    fontSize: 20, fontFace: FB, valign: "top"
  });
  s.addText("Megjegyzés: LLM kontextusban a desztilláció tömörítést is jelenthet.", {
    x: rightX, y: colY + 2.3, w: colW, h: 1.4,
    fontSize: 18, italic: true, fontFace: FB, color: ACCENT, valign: "top"
  });

  // Vertical divider
  s.addShape(pres.ShapeType.line, {
    x: W / 2, y: colY, w: 0, h: colH,
    line: { color: "DDDDDD", width: 1 }
  });
});

// 7 — Domain-specifikus
add(() => {
  const s = pres.addSlide();
  title(s, "Domain-specifikus");
  bullets(s, [
    "A modellnek nem kell mindent tudnia",
    "Csak a feladathoz szükséges tudás, kis modell, fókuszált viselkedés",
    "Cserébe: olcsóbb, gyorsabb, on-device futtatható",
    "API-független, lokális, független a szolgáltatótól"
  ]);
});

// 8 — Domain 1: Reddit sentiment
add(() => {
  const s = pres.addSlide();
  title(s, "1. domain: Reddit sentiment");
  s.addText("Bemenet:", {
    x: 0.7, y: 1.85, w: 3, h: 0.4,
    fontSize: 18, bold: true, fontFace: FB, color: MUTED
  });
  s.addText('"F**k you and your family"', {
    x: 0.7, y: 2.25, w: W - 1.4, h: 0.6,
    fontSize: 20, italic: true, fontFace: FB, color: BODY
  });
  s.addText("Kimenet (JSON):", {
    x: 0.7, y: 3.1, w: 3, h: 0.4,
    fontSize: 18, bold: true, fontFace: FB, color: MUTED
  });
  s.addText(
`{
  "tone": "aggressive",
  "sentiment": "negative",
  "safety": "harmful",
  "toxicity": "toxic"
}`,
    {
      x: 0.7, y: 3.5, w: W - 1.4, h: 3.2,
      fontSize: 18, fontFace: FM, color: NAVY, valign: "top"
    }
  );
});

// 9 — Domain 2: math
add(() => {
  const s = pres.addSlide();
  title(s, "2. domain: matek szöveges feladat");
  s.addText("Bemenet:", {
    x: 0.7, y: 1.85, w: 3, h: 0.4,
    fontSize: 18, bold: true, fontFace: FB, color: MUTED
  });
  s.addText(
`Problem: "Emma has 24 stickers and buys 18 more.
How many stickers does Emma have in total?"
A=?
B=?
C=A+B=?
Solution: ?;`,
    {
      x: 0.7, y: 2.25, w: W - 1.4, h: 2.6,
      fontSize: 18, fontFace: FM, color: BODY, valign: "top"
    }
  );
  s.addText("Kimenet:", {
    x: 0.7, y: 4.95, w: 3, h: 0.4,
    fontSize: 18, bold: true, fontFace: FB, color: MUTED
  });
  s.addText(
`A=24
B=18
C=A+B=42
Solution: 42;`,
    {
      x: 0.7, y: 5.4, w: W - 1.4, h: 1.8,
      fontSize: 18, fontFace: FM, color: NAVY, valign: "top"
    }
  );
});

// 10 — Domain 3: post generation
add(() => {
  const s = pres.addSlide();
  title(s, "3. domain: Reddit poszt generálás");
  s.addText("Bemenet (komment):", {
    x: 0.7, y: 1.85, w: 6, h: 0.4,
    fontSize: 18, bold: true, fontFace: FB, color: MUTED
  });
  s.addText('"Same thing happened to me last week, support was useless."', {
    x: 0.7, y: 2.25, w: W - 1.4, h: 0.7,
    fontSize: 20, italic: true, fontFace: FB, color: BODY
  });
  s.addText("Kimenet (poszt):", {
    x: 0.7, y: 3.2, w: 6, h: 0.4,
    fontSize: 18, bold: true, fontFace: FB, color: MUTED
  });
  s.addText(
    '"I\'m starting to think Reddit is actively targeting me. I just got shadowbanned after a completely neutral comment on a RAW photography thread, and I have no idea why it happened…"  <end>',
    {
      x: 0.7, y: 3.6, w: W - 1.4, h: 3.2,
      fontSize: 18, italic: true, fontFace: FB, color: BODY, valign: "top"
    }
  );
});

// 11 — Pipeline
add(() => {
  const s = pres.addSlide();
  title(s, "Pipeline");
  subtitle(s, "End-to-end rendszer a desztillációhoz, sok komponens");
  bullets(s, [
    "Adat előkészítés és domain definíció",
    "Tanár modell adatgenerálás (logit-okkal)",
    "Modell konfiguráció",
    "Tanítás konfigurálható regimennel",
    "Kiértékelés, log, checkpoint, kísérlet összehasonlítás"
  ]);
});

// 12 — Nem framework
add(() => {
  const s = pres.addSlide();
  title(s, "Nem framework");
  bullets(s, [
    "Nem `new Domain()` típusú API",
    "Új dolgok bevezetésére kódbővítés szükséges",
    "Új domain bevezetése jellemzően:",
    { text: "pár fájl", level: 1 },
    { text: "új bemeneti adatok feldolgozása (pl. nyers kommentek tisztítása)", level: 1 },
    { text: "tokenlista (input + output)", level: 1 },
    { text: "tooling (logging, modellfuttatási frissítések, pl. inference)", level: 1 }
  ], { fontSize: 18 });
});

// 13 — Domain definíció
add(() => {
  const s = pres.addSlide();
  title(s, "Pipeline: domain definíció");
  bullets(s, [
    "Bemeneti és kimeneti tokenek listája",
    "Prompt sablon a tanár modellhez",
    "Stop token a generálás lezárására",
    "Adatgenerálás konfigurálása (példaszám, batch, futtatás)"
  ]);
});

// 14 — Adatgenerálás
add(() => {
  const s = pres.addSlide();
  title(s, "Pipeline: adatgenerálás");
  bullets(s, [
    "Tanár modell (Gemma 3 4B) végigfut a példákon",
    "Minden lépésnél: nyers logit-ok mentése a figyelt tokenekre",
    "JSONL batchekben tárolva, 32 példa / batch",
    "Egyszer fut le, több modell tanításához újrahasznosítható"
  ]);
});

// 15 — Modell + tanítás
add(() => {
  const s = pres.addSlide();
  title(s, "Pipeline: modell és tanítás");
  bullets(s, [
    "Saját transformer student (RMSNorm, RoPE, GeGLU, Gemma-szerű)",
    "Konfigurálható: hidden dim, rétegszám, head-ek",
    "KL + CE veszteség, AdamW optimizer, LR cosine schedule",
    "Általában 5M és 129M paraméter között skálázva"
  ]);
});

// 16 — Config object
add(() => {
  const s = pres.addSlide();
  title(s, "Pipeline: konfigurálható tanítás");
  subtitle(s, "Modell + tanítási regimen, training_queue.json alapján");
  s.addText(
`// training/training_queue.json
{
  "run_name": "exp-klce-anneal",
  "domain": "reddit_comment_sentiment",
  "epoch_count": 3,
  "batch_size": 32,
  "learning_rate": 3e-4,
  "kl_ratio_start": 0.99,
  "kl_ratio_end":   0.50,
  "distillation_temperature_start": 5.0,
  "distillation_temperature_end":   3.0,
  "hidden_dim": 80,
  "num_layers": 5,
  "num_heads":  4
}`,
    {
      x: 0.8, y: 2.0, w: W - 1.6, h: 5.2,
      fontSize: 18, fontFace: FM, color: BODY, valign: "top"
    }
  );
});

// 17 — Experiment tags + stress test
add(() => {
  const s = pres.addSlide();
  title(s, "Pipeline: kísérletkezelés");
  bullets(s, [
    "Minden run külön mappa: konfig + log + checkpoint együtt",
    "Experiment tagek tetszőlegesen csoportosíthatók",
    "Több mint 100 run a registryben",
    "Stress-tesztelve: 512 MB-os batch fájlok a data generation oldalon",
    "Tanítás bizonyítottan fut consumer hardveren is",
    "Graceful shutdown, queue futtatás, felügyelet nélkül is"
  ], { fontSize: 18 });
});

// 18 — SECTION: Újdonságok
add(() => {
  const s = pres.addSlide();
  darkBg(s);
  s.addText("A projekt újdonságai", {
    x: 0.6, y: 3.0, w: W - 1.2, h: 1.5,
    fontSize: 54, bold: true, fontFace: FH, color: WHITE, align: "center", valign: "middle"
  });
  s.addShape(pres.ShapeType.line, {
    x: W / 2 - 0.6, y: 4.5, w: 1.2, h: 0,
    line: { color: ACCENT, width: 3 }
  });
});

// 19 — Annealing kombináció
add(() => {
  const s = pres.addSlide();
  title(s, "Annealing kombináció");
  bullets(s, [
    "Loss annealing: kutatott, LLM-ek területén kevésbé",
    "Temperature annealing: kutatott, LLM-eken is",
    "Együtt, ugyanabban a pipeline-ban: alulkutatott terület",
    "A pipeline mindkettőt függetlenül konfigurálhatóvá teszi"
  ]);
});

// 20 — Input + output projekció redukció
add(() => {
  const s = pres.addSlide();
  title(s, "Input és output projekció redukció");
  bullets(s, [
    "Vocabulary redukció létezik (pl. orosz LM-ek, Kolesnikova et al. 2022): általános vocab csökkentés alignment technikákkal",
    "Itt: azonos tokenizerre épülő, task-specifikus subset",
    "Ennyire explicit, task-specifikus formában alulkutatott terület",
    "Durva, gyakorlatias technika",
    "Egyszerre csökkenti az embedding és az output projekció költségét",
    "Kis modelleknél ezek aránytalanul nagy paraméterblokkok",
    "Math modell: offline kérdésmegválaszolás + a scaffold mutatja a levezetés képességét"
  ], { fontSize: 18 });
});

// 21 — Output redukció elv
add(() => {
  const s = pres.addSlide();
  title(s, "Output projekció redukció");
  subtitle(s, "Logikus első lépés: minek 262 144 token, ha 30 elég?");
  bullets(s, [
    "A kimeneti vocabulary közvetlenül skálázza az output projekciót",
    "A desztilláció a teljes eloszlásból tanul (KL)",
    'Ha túl szűk az output, elveszik a "dark knowledge"',
    "Auxiliary tokenekkel kompromisszum köthető"
  ]);
});

// 22 — JSON necessary tokens
add(() => {
  const s = pres.addSlide();
  title(s, "Szükséges tokenek: JSON");
  bullets(s, [
    "Strukturális karakterek: { } \" : ,",
    "Mezőnevek: tone, sentiment, safety, toxicity",
    "Kategória értékek: aggressive, neutral, toxic, …"
  ]);
  s.addText("≈ 27 token", {
    x: 0.7, y: 5.5, w: W - 1.4, h: 1.0,
    fontSize: 44, bold: true, fontFace: FH, color: ACCENT, align: "left"
  });
});

// 23 — Math necessary tokens
add(() => {
  const s = pres.addSlide();
  title(s, "Szükséges tokenek: math");
  bullets(s, [
    "Változók: A, B, C, …",
    "Szimbólumok: =, +, -, *, /, ;",
    "Számjegyek: 0–9"
  ]);
  s.addText("≈ 30 token", {
    x: 0.7, y: 5.5, w: W - 1.4, h: 1.0,
    fontSize: 44, bold: true, fontFace: FH, color: ACCENT, align: "left"
  });
});

// 24 — Auxiliary tokens table
add(() => {
  const s = pres.addSlide();
  title(s, "Auxiliary tokenek, gazdagabb soft target");
  subtitle(s, "Még mindig 99,8% redukció a teljes Gemma vocab-hoz képest");

  const head = { bold: true, color: WHITE, fill: { color: NAVY }, align: "center", valign: "middle", fontSize: 18, fontFace: FB };
  const cell = { color: BODY, align: "center", valign: "middle", fontSize: 18, fontFace: FB };
  const rows = [
    [
      { text: "", options: { ...head, fill: { color: WHITE } } },
      { text: "Szükséges", options: head },
      { text: "+ Auxiliary", options: head },
      { text: "Teljes Gemma", options: head },
      { text: "Redukció (csak szük.)", options: head },
      { text: "Redukció (+ aux.)", options: head }
    ],
    [
      { text: "Sentiment", options: { ...cell, bold: true, align: "left" } },
      { text: "27", options: cell },
      { text: "525", options: { ...cell, bold: true, color: ACCENT } },
      { text: "262 144", options: { ...cell, color: MUTED } },
      { text: "99,99%", options: cell },
      { text: "99,80%", options: { ...cell, bold: true, color: ACCENT } }
    ],
    [
      { text: "Math", options: { ...cell, bold: true, align: "left" } },
      { text: "30", options: cell },
      { text: "528", options: { ...cell, bold: true, color: ACCENT } },
      { text: "262 144", options: { ...cell, color: MUTED } },
      { text: "99,99%", options: cell },
      { text: "99,80%", options: { ...cell, bold: true, color: ACCENT } }
    ]
  ];
  s.addTable(rows, {
    x: 0.4, y: 2.4, w: W - 0.8,
    colW: [1.7, 1.6, 1.7, 2.2, 2.3, 2.03],
    rowH: 0.85,
    border: { type: "solid", color: "DDDDDD", pt: 1 }
  });
});

// 25 — Strukturált ≠ KL info
add(() => {
  const s = pres.addSlide();
  title(s, "Strukturált kimenet ≠ KL információ");
  bullets(s, [
    "A legtöbb auto-generált domain strukturált kimenetű",
    "Strukturált kimenetnél a tanár eloszlása éles",
    "A különbség több metrikán a futások közötti szórással összemérhető",
    "Ezért volt fontos egy free-form domain létrehozása, ahol a top-k token is érdekes"
  ]);
});

// 25b — Loss spread (figure)
add(() => {
  const s = pres.addSlide();
  title(s, "Loss-stratégiák szórása strukturált domaineken");
  subtitle(s, "Sentiment és math, 10-10 futtatás stratégiánként");
  s.addImage({
    path: IMG("05_structured_loss_spread.png"),
    x: 1.0, y: 1.9, w: 11.3, h: 5.2, sizing: { type: "contain", w: 11.3, h: 5.2 }
  });
});

// 25c — Confusion matrix
add(() => {
  const s = pres.addSlide();
  title(s, "Sentiment kategóriahibák");
  subtitle(s, "JSON struktúra stabil, a bizonytalanság a kategóriákban van");
  s.addImage({
    path: IMG("08_reddit_confusion_matrices.png"),
    x: 1.0, y: 1.9, w: 11.3, h: 5.2, sizing: { type: "contain", w: 11.3, h: 5.2 }
  });
});

// 26 — Free-form domain
add(() => {
  const s = pres.addSlide();
  title(s, "Free-form domain: post generation");
  subtitle(s, "Kommentből generálj rövid Reddit posztot");
  bullets(s, [
    "Több plauzibilis folytatás versenyez egymással",
    "Lágyabb tanár eloszlás, több KL információ",
    "Itt kezd számítani, hogy KL-t vagy CE-t használsz"
  ]);
});

// 27 — Sizes table
add(() => {
  const s = pres.addSlide();
  title(s, "Méretek a három domainen");
  const head = { bold: true, color: WHITE, fill: { color: NAVY }, align: "center", valign: "middle", fontSize: 18, fontFace: FB };
  const cell = { color: BODY, align: "center", valign: "middle", fontSize: 18, fontFace: FB };
  const rows = [
    [
      { text: "", options: { ...head, fill: { color: WHITE } } },
      { text: "Bemeneti vocab", options: head },
      { text: "Kimeneti vocab", options: head },
      { text: "Output redukció", options: head }
    ],
    [
      { text: "Sentiment", options: { ...cell, bold: true, align: "left" } },
      { text: "68 328", options: cell },
      { text: "525", options: cell },
      { text: "99,8%", options: { ...cell, color: ACCENT, bold: true } }
    ],
    [
      { text: "Math", options: { ...cell, bold: true, align: "left" } },
      { text: "1 282", options: cell },
      { text: "528", options: cell },
      { text: "99,8%", options: { ...cell, color: ACCENT, bold: true } }
    ],
    [
      { text: "Post generation", options: { ...cell, bold: true, align: "left" } },
      { text: "72 133", options: cell },
      { text: "26 659", options: cell },
      { text: "89,8%", options: { ...cell, color: ACCENT, bold: true } }
    ]
  ];
  s.addTable(rows, {
    x: 0.6, y: 2.1, w: W - 1.2,
    colW: [3.0, 3.0, 3.0, 3.13],
    rowH: 0.8,
    border: { type: "solid", color: "DDDDDD", pt: 1 }
  });
  s.addText("Vs. teljes Gemma kimeneti vocab: 262 144 token", {
    x: 0.6, y: 5.7, w: W - 1.2, h: 0.5,
    fontSize: 18, italic: true, fontFace: FB, color: MUTED, align: "center"
  });
});

// 28 — Pure KL
add(() => {
  const s = pres.addSlide();
  title(s, "Pure KL");
  subtitle(s, "Az eloszlás követése");
  bullets(s, [
    "Jól követi a tanár top-k eloszlását",
    "Variáció és sampling szempontjából fontos",
    "Temperature annealing: lágy, majd élesebb átmenet a tanítás során"
  ]);
});

// 29 — Pure CE
add(() => {
  const s = pres.addSlide();
  title(s, "Pure CE");
  subtitle(s, "A helyes következő token");
  bullets(s, [
    "Magabiztosan eltalálja a következő tokent",
    "De elveszti a tanár eloszlás-szerkezetét",
    "Top-k overlap ebben a mérésben látványosan alacsonyabb"
  ]);
});

// 30 — Best of both worlds
add(() => {
  const s = pres.addSlide();
  title(s, "Best of both worlds");
  subtitle(s, "KL/CE annealing + temperature annealing");
  bullets(s, [
    "Magabiztos next-token predikció",
    "És megőrzött tanár top-k eloszlás",
    "Lépésenként hangolható kompromisszum a kettő között"
  ]);
});

// 30a — Training curve: Pure CE
add(() => {
  const s = pres.addSlide();
  title(s, "Tanulási dinamika: Pure CE");
  subtitle(s, "Nagyobb sentiment futtatás, batchenkénti loss és accuracy");
  s.addImage({
    path: IMG("07d_large_pure_ce_training_curves.png"),
    x: 1.0, y: 1.9, w: 11.3, h: 5.2, sizing: { type: "contain", w: 11.3, h: 5.2 }
  });
});

// 30b — Training curve: Pure KL
add(() => {
  const s = pres.addSlide();
  title(s, "Tanulási dinamika: Pure KL");
  subtitle(s, "Ugyanaz a futtatás, KL veszteséggel");
  s.addImage({
    path: IMG("07e_large_pure_kl_training_curves.png"),
    x: 1.0, y: 1.9, w: 11.3, h: 5.2, sizing: { type: "contain", w: 11.3, h: 5.2 }
  });
});

// 30c — Training curve: KL/CE annealing
add(() => {
  const s = pres.addSlide();
  title(s, "Tanulási dinamika: KL/CE annealing");
  subtitle(s, "0,99 → 0,50 KL arány a tanítás során");
  s.addImage({
    path: IMG("07f_large_klce_annealing_training_curves.png"),
    x: 1.0, y: 1.9, w: 11.3, h: 5.2, sizing: { type: "contain", w: 11.3, h: 5.2 }
  });
});

// 30d — Training curve: tsc-scale 53M U-shape
add(() => {
  const s = pres.addSlide();
  title(s, "Tanulási dinamika: 53M paraméteres modell");
  subtitle(s, "Érdekes minta: a loss csökken, megugrik, majd újra csökken");
  s.addImage({
    path: IMG("07g_tsc_scale_53M_training_curves.png"),
    x: 1.0, y: 1.9, w: 11.3, h: 5.2, sizing: { type: "contain", w: 11.3, h: 5.2 }
  });
});

// 31 — Goal-tunable
add(() => {
  const s = pres.addSlide();
  title(s, "Célhoz szabható");
  subtitle(s, "A schedule nem csak domainfüggő, célfüggő is");
  bullets(s, [
    "Determinisztikus next-token? CE-felé toló schedule",
    "Kreatív, sokszínű generálás? KL-felé toló schedule",
    "Kompromisszum? Annealing",
    "Kis 8000 példás futtatásokon is jól tesztelhető, sok kísérlet futtatható"
  ], { fontSize: 18 });
});

// 32 — Top-k metrics
add(() => {
  const s = pres.addSlide();
  title(s, "Top-k metrikák");
  bullets(s, [
    "Top-20 cél-token találat: a tanár cél tokenje benne van-e a student top-20-ban",
    "Teacher-student top-20 overlap: mennyire egyezik a két halmaz",
    "Átlagos cél-token rang: hol helyezkedik el a cél a student eloszlásában (kisebb a jobb)"
  ], { fontSize: 18 });
});

// 33 — Result table (best per column highlighted)
add(() => {
  const s = pres.addSlide();
  title(s, "Eredmény: post generation top-k");
  const head = { bold: true, color: WHITE, fill: { color: NAVY }, align: "center", valign: "middle", fontSize: 18, fontFace: FB };
  const cell = { color: BODY, align: "center", valign: "middle", fontSize: 18, fontFace: FB };
  const best = { ...cell, bold: true, color: ACCENT };
  const rows = [
    [
      { text: "Stratégia", options: { ...head, align: "left" } },
      { text: "Top-20 hit", options: head },
      { text: "Top-20 overlap", options: head },
      { text: "Átlagos rang", options: head }
    ],
    [
      { text: "Pure CE", options: { ...cell, align: "left" } },
      { text: "67,55%", options: cell },
      { text: "26,69%", options: cell },
      { text: "1068,92", options: cell }
    ],
    [
      { text: "Pure KL", options: { ...cell, align: "left" } },
      { text: "64,42%", options: cell },
      { text: "37,31%", options: best },
      { text: "464,70", options: best }
    ],
    [
      { text: "KL/CE + temp anneal", options: { ...cell, align: "left", bold: true } },
      { text: "69,08%", options: best },
      { text: "34,61%", options: cell },
      { text: "486,31", options: cell }
    ]
  ];
  s.addTable(rows, {
    x: 0.6, y: 2.1, w: W - 1.2,
    colW: [4.0, 2.7, 2.7, 2.73],
    rowH: 0.85,
    border: { type: "solid", color: "DDDDDD", pt: 1 }
  });
  s.addText("Magasabb hit + KL-közeli overlap, a két cél kombinálható.", {
    x: 0.6, y: 6.0, w: W - 1.2, h: 0.5,
    fontSize: 18, italic: true, fontFace: FB, color: MUTED, align: "center"
  });
});

// 33a — Post-gen top-k visualization
add(() => {
  const s = pres.addSlide();
  title(s, "Post generation top-k, vizuálisan");
  subtitle(s, "Pure CE / Pure KL / KL/CE + temperature annealing összehasonlítása");
  s.addImage({
    path: IMG("09_postgen_topk_metrics.png"),
    x: 1.0, y: 1.9, w: 11.3, h: 5.2, sizing: { type: "contain", w: 11.3, h: 5.2 }
  });
});

// 33b — Domain data cost
add(() => {
  const s = pres.addSlide();
  title(s, "Desztillációs adat költsége domainenként");
  subtitle(s, "Miért nehezebb a post generation kísérletsorozat skálázása");
  s.addImage({
    path: IMG("10_domain_data_cost.png"),
    x: 1.0, y: 1.9, w: 11.3, h: 5.2, sizing: { type: "contain", w: 11.3, h: 5.2 }
  });
});

// 34 — Use case 1: math
add(() => {
  const s = pres.addSlide();
  title(s, "Felhasználás: offline matek megoldó");
  bullets(s, [
    "Természetes nyelvű magyarázattal",
    "Telefonon, offline futtatható",
    "API-mentes, díjmentes inference"
  ]);
});

// 35 — Use case 2: auto-moderation as guide
add(() => {
  const s = pres.addSlide();
  title(s, "Felhasználás: kommentelési útmutató");
  bullets(s, [
    "Komment ellenőrzése küldés előtt, kliens oldalon",
    'Magas konfidenciánál: "biztos el akarod küldeni? A kommented rosszindulatúnak tűnik."',
    "Nem tilt, hanem visszajelzést ad a felhasználónak",
    "Server-side toxicity terhelés is csökken"
  ], { fontSize: 18 });
});

// 36 — Use case 3: customer support email approval
add(() => {
  const s = pres.addSlide();
  title(s, "Felhasználás: customer support email");
  bullets(s, [
    "Filléres ügyfélszolgálati email-fogalmazás",
    "Kimenet emberi vagy nagy LLM jóváhagyásra vár",
    "Nagy kontextus + tipikus minták, kis modellben is megtanulhatók",
    "Volumenban tört költséggel egy nagy modellhez képest"
  ]);
});

// 37 — Use case 4: edge deploy general
add(() => {
  const s = pres.addSlide();
  title(s, "Felhasználás: bármilyen edge deploy");
  subtitle(s, "Bármilyen nyelvi igénnyel rendelkező edge deployra alkalmazható");
  bullets(s, [
    "Szűk, gyakran ismételt feladat → desztillált student",
    "Gyors válaszidő, lokális adatkezelés",
    "Skálázható: új domain = új tokenlista + új tanító adat"
  ]);
});

// 38 — Scaling curve image
add(() => {
  const s = pres.addSlide();
  title(s, "Méret: 34× ... 860× kisebb");
  subtitle(s, "Sentiment skálázási görbe, a feladat korán telítődik");
  s.addImage({
    path: IMG("03_sentiment_scaling_curve.png"),
    x: 1.5, y: 1.9, w: 10.3, h: 5.2, sizing: { type: "contain", w: 10.3, h: 5.2 }
  });
});

// 38a — Reduced vs full vocab
add(() => {
  const s = pres.addSlide();
  title(s, "Teljes és redukált input vocabulary");
  subtitle(s, "Hasonló paraméterkeret, redukált vocab versenyképes vagy jobb");
  s.addImage({
    path: IMG("04_reduced_vs_full_vocab.png"),
    x: 1.0, y: 1.9, w: 11.3, h: 5.2, sizing: { type: "contain", w: 11.3, h: 5.2 }
  });
});

// 39 — Param allocation image
add(() => {
  const s = pres.addSlide();
  title(s, "Paraméterallokáció: hova kerül a kapacitás");
  subtitle(s, "Redukció nélkül, csak input, illetve input + output");
  s.addImage({
    path: IMG("02_parameter_allocation.png"),
    x: 1.5, y: 1.9, w: 10.3, h: 5.2, sizing: { type: "contain", w: 10.3, h: 5.2 }
  });
});

// 40 — Summary
add(() => {
  const s = pres.addSlide();
  title(s, "Összegzés");
  subtitle(s, "Saját LLM nagyobb modellből, általános pipeline");
  bullets(s, [
    "1. domain definiálása",
    "2. tanító adat generálása a tanár modellből",
    "3. transformer student modell konfigurálása",
    "4. tanítási regimen konfigurálása (KL/CE, temperature, schedule)",
    "5. tanítás (akár experiment batchekben) és kiértékelés"
  ]);
});

// 41 — Tooling
add(() => {
  const s = pres.addSlide();
  title(s, "Tooling");
  subtitle(s, "Eredmény-elemzés és kontrollált kísérletek");
  bullets(s, [
    "Run-onkénti log + checkpoint, kísérlet összehasonlítás",
    "Skálázási és vocabulary ábrák, annealing összehasonlítás",
    "Top-k elemzés, confusion matrix",
    "Graceful shutdown, queue futtatás, HTML report"
  ]);
});

// 42 — Thanks
add(() => {
  const s = pres.addSlide();
  darkBg(s);
  s.addText("Köszönöm a figyelmet!", {
    x: 0.6, y: 2.6, w: W - 1.2, h: 1.4,
    fontSize: 60, bold: true, fontFace: FH, color: WHITE, align: "center", valign: "middle"
  });
  s.addShape(pres.ShapeType.line, {
    x: W / 2 - 0.6, y: 4.1, w: 1.2, h: 0,
    line: { color: ACCENT, width: 3 }
  });
  s.addText("Kérdések?", {
    x: 0.6, y: 4.4, w: W - 1.2, h: 0.8,
    fontSize: 28, italic: true, fontFace: FB, color: ICE, align: "center"
  });
});

// Slide indices (0-based) that should NOT show a page number:
// title (1), section header (18), closing (50)
const NO_PAGE_NUM = new Set([0, 17, slides.length - 1]);

slides.forEach((fn, index) => {
  fn();
  if (NO_PAGE_NUM.has(index)) return;
  const slide = pres.slides[pres.slides.length - 1];
  slide.addText(`${index + 1} / ${slides.length}`, {
    x: W - 1.4, y: H - 0.45, w: 1.1, h: 0.35,
    fontSize: 14, fontFace: FB, color: MUTED, align: "right", valign: "middle"
  });
});

await pres.writeFile({ fileName: path.join(__dirname, "slides.pptx") });
console.log("Wrote slides.pptx (" + slides.length + " slides)");
