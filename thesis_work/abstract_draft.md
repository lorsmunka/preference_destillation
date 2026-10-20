Absztrakt – javaslat (2026-10-20). Nem a dolgozat fájlja: ellenőrzésre. A dolgozat (thesis_work/new_thesis/thesis.md) változatlan.

# A változat: a mostani absztrakt, kiegészítve a feladatkiírás szerint

A nagy nyelvi modellek általános képességei sok szűk, ismétlődő feladatnál aránytalanul drágák. A dolgozat egy saját fejlesztésű keretrendszert mutat be, amellyel egy nagy nyelvi modell adott szakterületre vonatkozó tudása kis méretű, nulláról tanított nyelvi modellekbe és ágensekbe desztillálható. A dolgozat konkrét domainenkénti megvalósításokat, általános észrevételeket és eredményeket is bemutat.

A dolgozat több tanítómodellt (Gemma 3 4B, illetve Gemma 4 12B), több alapoktól tanított, pár millió paraméteres tanulómodellt és változó entrópiájú szűk domaineket vizsgál. Két ágenst is megvalósít: egy matematikatanító ágenst és egy programkódban kereső és azt magyarázó ágenst. Egy ágens a desztillált tanulómodellből és egy keretprogramból áll. A keretprogram kezeli a beszélgetést, ellenőrzi a modell válaszait, és hiba esetén újrakérdez.

A tanulómodell architektúrája korszerű LLM-megoldásokat használ, a tanítómodell architektúráját követve. A tanulómodell és annak betanítása a következő fő paraméterek mentén konfigurálható: az architektúra (rétegek száma, rejtett dimenzió, figyelem fejek száma); a KL-divergencia és a cross-entropy veszteség aránya és annak ütemezése a tanítás során; a desztillációs hőmérséklet a tanuló és a tanító oldalon külön ütemezve; a tanító eloszlásához illesztett divergencia típusa (forward KL, reverse KL, Jensen-Shannon, TVD); a tanító entrópiájához kötött veszteségsúlyozás; valamint a figyelem rétegekre adott, ütemezhető zaj. A tanítómodell teljes (vizsgált) kimeneti eloszlása minden lépésnél mentésre kerül, így ezek a döntések új adatgenerálás nélkül vizsgálhatók.

A keretrendszer moduláris: egy új szakterület vagy ágens külön modulként adható hozzá. A rendszer a desztilláció lefolyásáról részletes elemzést ad. A feladatpontosságot mindig a leggyakoribb válasz pontosságához méri, és összeveti a tanító és a tanuló eloszlását (entrópia, top-k egyezés, perplexitás). A rendszer helyes működését egységtesztek ellenőrzik.

[Eredmények: 1–2 mondat a felhős futások után.]

# B változat: rövidebb, egy bekezdés (a paraméterlista a specifikáció fejezetbe kerül)

A nagy nyelvi modellek általános képességei sok szűk, ismétlődő feladatnál aránytalanul drágák. A dolgozat egy saját fejlesztésű keretrendszert mutat be, amellyel egy nagy nyelvi modell adott szakterületre vonatkozó tudása kis méretű, nulláról tanított nyelvi modellekbe és ágensekbe desztillálható. A dolgozat több tanítómodellt (Gemma 3 4B, illetve Gemma 4 12B), pár millió paraméteres tanulómodelleket és változó entrópiájú szűk domaineket vizsgál. Két ágenst is megvalósít: egy matematikatanító ágenst és egy programkódban kereső és azt magyarázó ágenst. Mindkettő a desztillált tanulómodellből és egy keretprogramból áll, amely ellenőrzi a modell válaszait. A tanulómodell architektúrája és betanítása finoman szabályozható: az architektúra, a veszteségek aránya és ütemezése, a desztillációs hőmérséklet és a divergencia típusa is beállítható. A tanítómodell teljes (vizsgált) kimeneti eloszlása minden lépésnél mentésre kerül, így ezek a döntések új adatgenerálás nélkül vizsgálhatók. A keretrendszer moduláris, a desztilláció lefolyásáról részletes elemzést ad, és helyes működését egységtesztek ellenőrzik. [Eredmények: 1–2 mondat a felhős futások után.]

# Ami új a mostanihoz képest (A változat)

- 1. bekezdés: a keretrendszer mondata (a feladatkiírás szavaival: „adott szakterületre vonatkozó tudása … desztillálható”).
- 2. bekezdés: „pár millió paraméteres” a „pár millió paramétertől százmillió paraméterig” helyett (a legnagyobb eddig tanított tanuló kb. 6 millió paraméteres); a két ágens és a keretprogram.
- 4. bekezdés: modularitás, elemzés, tesztek (a feladatkiírás mindhármat kéri).
- Az eredmény mondata még hiányzik.
