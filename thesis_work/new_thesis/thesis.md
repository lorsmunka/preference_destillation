Absztrakt (magyar) + Abstract (angol)

Magyar cím: Nagy nyelvi modellek domain-specifikus desztillációja alapoktól tanított kis modellekbe
Angol cím: Domain-Specific Distillation of Large Language Models into Small Models Trained from Scratch

A nagy nyelvi modellek megjelenésével az intelligencia, a problémamegoldás és a döntéshozatal soha nem látott egyszerűséggel vált beépíthetővé a különböző szoftveres megoldásokba. Ezzel egyidejűleg az ilyen alkalmazások számításigénye és így futtatásának költsége is jelentősen megnőtt. Sok esetben fontos az ilyen modelleket lokálisan futtatni, akár szenzitív adatok, offline elérhetőség vagy megbízhatóság miatt. Ezeknél a feladatoknál viszont általában nincs szükség a modell teljes tudására, hiszen csak egy bizonyos feladatot lát el.
A dolgozat a domain-specifikus tudás desztillálását vizsgálja alapoktól tanított, jelentősen csökkentett paraméterszámú modellekbe és ügynökökbe. A vizsgált modellek elkészítéséhez és kiértékeléséhez használt saját keretrendszer jelentősen leegyszerűsíti a desztillálás folyamatát: a domain definiálását követően mindössze konfigurálással beállítható a tanítóadatok előállítása, maga a tanítás és a tanulómodell architektúrája. A tanítóadatok újrahasználhatóak több modell betanítására is.
A dolgozat több domaint is vizsgál: alapfeladatokat (text sentiment classification, posztgenerálás), illetve ügynökfeladatokat is (matematikai feladatok megoldása és magyarázata, kódbázisban végrehajtott keresés és magyarázás), és ezek mentén igyekszik megválaszolni olyan kérdéseket, mint például: kis méretű tanulómodellek mennyire képesek elsajátítani a tanítómodell viselkedését? Különböző entrópiájú domaineknél hogyan érdemes a különböző veszteségeket használni vagy vegyíteni? Milyen módszerrel érdemes ügynököket desztillálni? Hogyan javul a teljesítmény a modell méretének függvényében? Milyen következtetések vonhatóak le nagyobb büdzséjű desztillálásokra?
A dolgozat több tanítómodellt (Gemma 3 4B, Gemma 4 12B) használ, öt domaint vizsgál részletesen, és ezek közül kettőnél saját modellhám (harness) segítségével önálló ügynököket hoz létre.

1. BEVEZETÉS

1.1 A megoldandó probléma, motiváció
Az utóbbi években a mesterséges inteligencia és ezen belül természetes nyelv feldolgozására képes modellek iránti igény jelentősen megnőtt. Ilyen modelleknek gyakori felhasználási területe lehet a kódírás, döntések programatikus meghozatala vagy a tanulás támogatása. Az ilyen feladatok ellátásához jelenleg a legegyszerűbb és sok esetben az egyetlen elérhető mód, ha a különböző felhőszolgáltatóktól vásárlunk hozzáférést, legyen egy előfizetés vagy API-hívás alapú számlázás. Ez viszont sok problémát felvet, például szenzitív adatoknál vagy kritikus infrastruktúra kódjánál nem feltétlenül használhatóak az ilyen megoldások, hiszen adatszivárgás léphet fel. Egy másik probléma lehet a megbízhatóság, hiszen ha a modellre infrastruktúraként tekintünk, sebezhetőséget jelenthet, ha a modell válaszminősége a kontrollunkon kívül változik, például mert a szolgáltató eltérő architektúrájú hardverről szolgálja ki a modellt, vagy mert új védőkorlátokat vezet be. Újabb probléma lehet az ilyen modellek költsége vagy elérhetősége. Ahogy a legmodernebb és legjobb modellek egyre drágább áron vagy csak előfizetések mögé rejtve lesznek elérhetőek, nem feltétlenül lesz az intelligencia mindenki számára elérhető.
Nyílt súlyú modellek is elérhetőek, igaz, a legtöbb benchmarkon és valós tapasztalatok alapján a zászlóshajó zárt súlyú modellektől enyhén elmaradnak. Ha felhőszolgáltató infrastruktúrájáról szolgáljuk ki ezeket, a fent említett problémák csak részben mérsékelhetőek, lokális hardveren a hihetetlen memória és grafikus számításigény miatt pedig sokszor nem lehetséges a megfelelő minőségben kiszolgálni ezeket a modelleket.
Erre kínál megoldást a desztillálás, amikor is egy nagyobb modell viselkedését próbáljuk meg közelíteni egy jelentősen kisebb számítási kapacitást igénylő tanulómodellel. Ez a folyamat veszteségesen tömöríti a tanármodell tudását, tehát általában elmondható, hogy a kisebb tanulómodell válaszminősége (a tömörítés mértékétől függően) rosszabb lesz.
A modern LLM-ek egyik legértékesebb tulajdonsága, hogy egy modell szinte bármire használható. Ha erről hajlandóak vagyunk részben lemondani, és csak egy specifikus képességét szeretnénk megtartani, a tömörítés során sokkal nagyobb hatékonysággal tudjuk megőrizni az adott domainen az eredeti modell válaszminőségét.
A dolgozat célja egy olyan általános megoldás bemutatása, amely lehetővé teszi a domain-specifikus desztillálást. Mindezt olyan példákon keresztül, melyek bemutatják a módszer hatékonyságát és előnyeit. Bemutatásra kerül például egy olyan matematika feladat megoldó és magyarázó ügynök, amely akár egy középkategóriás okostelefonon is offline elfut, vagy akár egy középkategóriás laptopon is lokálisan futtatható kódértelmező ügynök.

1.2 A dolgozat célja
A dolgozat célja egy olyan keretrendszer megtervezése és implementálása, amellyel egy nagy nyelvi modell adott szakterületre vonatkozó tudása kis méretű, nulláról tanított nyelvi modellekbe vagy ágensekbe desztillálható. A dolgozat vizsgálja a tudásdesztilláció és a kis méretű nagy nyelvi modellek tanításának módszereit, a nyelvi modellekre épülő ágensek felépítését, valamint a rendelkezésre álló eszközöket. A keretrendszer funkcionalitása lefedi a tanítóadatok előállítását a tanármodellel, a tanulómodell tanítását és a tanítása kiértékelését. A keretrendszer lehetővé teszi a kis modell architektúrájának és a tanítási folyamat finom szabályozását, és a desztilláció lefolyásáról részletes elemzést ad. A rendszer moduláris szemlélettel lett megtervezve, új szakterületek és ágensek külön modulként hozzáadhatóak.

1.3 A dolgozat felépítése

2. A PROBLÉMA ELEMZÉSE, A SPECIFIKÁCIÓ KIDOLGOZÁSA

Az LLM-ek egyre nagyobb szerepet játszanak a különböző digitális rendszerekben. Gyorsíthatnak vagy automatizálhatnak bizonyos szoftverfejlesztési és kiberbiztonsági feladatokat, tehát eszközként válhatnak az állami, vállalati vagy személyes infrastruktúra részévé. Vagy lehetnek egy szoftver beépített elemei, például különböző moderációs eszközökben. Az is előfordulhat, hogy az LLM maga a szoftveres megoldás fő része, például egy oktató-magyarázó alkalmazásban. Az 1.1 fejezetben részletesebben kifejtett kiszámíthatósági, megbízhatósági és költségbeli okok miatt sok esetben ideális vagy szükségszerű lenne a lokális kiszolgálás. Szintén az 1.1 fejezetben kifejtett óriási hardverigény vagy a meglévő hardveren elfutó, de jelentősen rosszabb válaszminőségű modell miatt a lokálisan futtatott általános LLM-ek nem feltétlenül jelentenek megoldást. A következőkben bemutatásra kerülnek a jelenlegi módszerek, azoknak a hiányosságai illetve a domain-specifikus desztilláció által kínált megoldás.

2.1 Általános desztilláció
A tudásdesztilláció (knowlage distillation) egy tömörítési módszer, amely egy kisebb modellel próbálja minél pontosabban közelíteni a desztillálni kívánt nagyobb modellt. A kis modellt tanulómodellnek, a nagy modellt tanármodellnek hívják. A klasszikus tudásdesztilláció alapötletét Hinton és társai [1] fogalmazták meg, melynek lényege, hogy a tanulómodellt nemcsak a helyes válaszra tanítjuk meg, hanem a tanármodell teljes kimeneti eloszlására is. A tanulómodell így nemcsak a helyes választ, de a tanármodell bizonytalanságát, úgynevezett "dark knowledge"-t is képes megtanulni. Ez a bizonytalanság érdekes többletinformációt is magában hordoz, hiszen azt is megmutatja, hogy melyik rossz válaszok állnak közel a jóhoz. Ráadásul a nagy nyelvi modellek működéséből fakadóan kifejezetten fontos ez az információ, ugyanis gyakran előfordul, hogy egy LLM használata közben nem a legvalószínűbb, hanem valamelyik más valószínű szóval folytatja a válaszadást.
Egy másik gyakran használt tudásdesztillálási módszer nagy nyelvi modelleknél az eloszlás nélkül, csak a tanármodell kimeneti szövegén történő tanítás, az úgynevezett szekvenciaszintű desztilláció [2].
A nagy nyelvi modelleknél a desztillációt ma jellemzően általános céllal használják, tehát a tanulómodelltől azt várjuk, hogy a tanár minden képességét megtartsa. Erre példa a Gemma 2 modellcsalád [3], ahol a 2 és 9 milliárd paraméteres modelleket már desztillációval tanították. A szerzők kísérletében egy 7 milliárd paraméteres tanárból desztillált 2 milliárd paraméteres modell három benchmark átlagában 67,7 pontot ért el, szemben az ugyanannyi tokenen, desztilláció nélkül tanított változat 60,3 pontjával.
Vannak természetesen specifikusabban desztillált modellek, ilyenek például a DeepSeek-R1-Distill modellek [4], amelyeknél nagyjából 800 ezer, főként a DeepSeek-R1 érvelő (reasoning) modell által generált példán tanították tovább a kisebb, 1,5 és 70 milliárd paraméter közötti Qwen és Llama modelleket, kifejezetten az érvelési képességek átadására fókuszálva. Ebben a kísérletben eloszlás nélkül, csak a tanármodell szövegén tanították a tanulómodelleket.
Busbridge és társai [5] szerint a desztillált modell teljesítménye előre jelezhető a tanulómodell méretéből és a tanításra fordított számítási keretből, és ha a modell általános képességeit nézzük, akkor ez a tömörítés veszteséges. Ha viszont lemondunk az általánosságról, és csak egy szűk feladatra (domain-re) szeretnénk fókuszálni, a szükséges tudás egy jóval kisebb modellbe is beférhet.

2.2 Kvantálás
A desztillálás egy olyan tömörítőmódszer, amely a paraméterszám csökkentésére koncentrál, ezzel szemben a kvantálás (quantization) a meglévő súlyok helyigényét csökkenti. Míg egy nem kvantált LLM súlyait általában 16 biten szokták tárolni, kvantáláskor ez lecsökkenhet 8 vagy 4 bites precizitásra. A kvantálás a modell teljesítményének romlásával járhat, viszont így jelentősen nagyobb paraméterszámú modelleket is képesek vagyunk futtatni lokálisan, ezért az ilyen modellek futtatására ez az egyik legelterjedtebb módszer.
Előnye, hogy nem szükséges újratanítani a modellt: a GPTQ módszerrel [6] például egy 175 milliárd paraméteres modell nagyjából négy GPU-óra alatt 3 vagy 4 bitesre kvantálható, elhanyagolható pontosságvesztéssel, így egyetlen GPU-n is futtatható.
A kvantálás viszont csak a modell számábrázolásának pontosságát csökkenti, a modell felépítése és így a paraméterek száma változatlan marad, ugyanúgy a teljes modelltudás lesz tömörítve, csak kevesebb memóriát foglal. Dettmers és Zettlemoyer [7] kutatásából az is kiderül, hogy az adott memóriakeret mellett szinte mindig a 4 bites pontosság adja a legjobb eredményt. Tehát a kvantálással nagyjából négyszeres memóriacsökkenés érhető el, a dolgozatban vizsgált domain-specifikus desztillálással viszont több százszoros, akár több ezerszeres tömörítés is elérhető. A desztillálás és kvantálás nem feltétlenül kizáró optimalizációs módszerek, tehát egy desztillált modellt lehet kvantálni is.

2.3 Modellfinomhangolás
A finomhangolás (fine-tuning) során egy már előtanított nagy nyelvi modellt tanítunk tovább egy szűkebb szakterületre, a domainnek megfelelő adatokon vagy környezetben (environment), például megerősítéses tanulással (RL). Előnye, hogy nem kell a modellt a nulláról tanítani, hiszen az alapvető nyelvi képességeket a modell már elsajátította. Ez a technika viszont a már létező előtanított (pre-trained) modellekhez köt, amelyek sok esetben nem is léteznek a dolgozatban vizsgált méretekben.
A finomhangolás költsége jelentősen csökkenthető paraméterhatékony módszerekkel. Ezek közül az egyik legelterjedtebb a LoRA [8], amely az eredeti súlyokat befagyasztja, és csak kis, alacsony rangú mátrixokat tanít mellettük. A szerzők ezzel a módszerrel a 175 milliárd paraméteres GPT-3 esetén a tízezred részére csökkentették a tanítandó paraméterek számát és a harmadára a szükséges GPU-memóriát. Az így kapott mátrixok utólag pedig beolvaszthatóak az eredeti súlyokba, így a folyamat végére nem változik a modell mérete.
A QLoRA [9] ezt a 2.2 fejezetben bemutatott kvantálással kombinálja. A módszer a modell súlyainak 4 bitesre kvantálása után, befagyasztott modell mellett tanítja a LoRA-mátrixokat. Így egy 65 milliárd paraméteres modell finomhangolásának memóriaigénye több mint 780 GB-ról 48 GB alá csökken, lényeges teljesítményromlás nélkül. A finomhangolás viszont csak a súlyok értékét változtatja meg, azok számát és elrendezését nem, tehát a finomhangolt modell futtatásához ugyanakkora hardver kell mint az álltalános nem finomhagot modellhez.
Ezenfelül a domainre hangolás az általános képességek rovására is mehet: Biderman és társai [10] például a Llama-2-7B modellt finomhangolták tovább programozási és matematikai feladatokra, és azt tapasztalták, hogy a modell teljes finomhangolásakor jobban tanul a célterületen, viszont többet felejt a többi mért képességéből. Ezzel szemben a LoRA kevesebbet felejt, de kevesebbet is tanul.

2.4 Domain-specifikus desztillálás alapoktól tanított kis méretű modelleknél

2.5 Ügynökdesztillálás
2.6 Funkcionális és nem funkcionális követelmények
2.7 Szereplők, használati esetek

3. SZAKIRODALMI ÁTTEKINTÉS – LEHETSÉGES MEGKÖZELÍTÉSEK ÉS MEGOLDÁSOK

3.1 Alapfogalmak ← your "fogalmak" section belongs here
3.1.1 …
3.1.2 …
3.2 Meglévő megoldások, hasonló rendszerek
3.3 Technológiák, módszerek összevetése
3.4 Következtetések

4. A MEGOLDÁSI MÓDSZER KIVÁLASZTÁSA ÉS INDOKLÁSA

4.1 Döntési szempontok
4.2 A választott megoldás

5. RÉSZLETES SPECIFIKÁCIÓ

6. TERVEZÉS – MUNKAFÁZISOK ÉS TAPASZTALATOK

6.1 Architektúra
6.2 Adatmodell
6.3 Felhasználói felület terve

7. MEGVALÓSÍTÁS

7.1 Fejlesztői környezet, eszközök
7.2 A fő komponensek megvalósítása

8. TESZTELÉS

8.1 Tesztstratégia
8.2 Tesztesetek és eredmények

9. EREDMÉNYEK BEMUTATÁSA, ÉRTÉKELÉSE, ÖSSZEVETÉS HASONLÓ RENDSZEREKKEL

10. A MEGVALÓSÍTÁS ELEMZÉSE, ALKALMAZÁSA, TOVÁBBFEJLESZTÉSI LEHETŐSÉGEK

11. ÖSSZEFOGLALÁS (magyarul)

12. SUMMARY (angolul)

13. IRODALOMJEGYZÉK

[1] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," presented at the NIPS 2014 Deep Learning Workshop, 2015. [Online]. Available: https://arxiv.org/abs/1503.02531
[2] Y. Kim and A. M. Rush, "Sequence-level knowledge distillation," in Proc. 2016 Conf. Empirical Methods Natural Lang. Process. (EMNLP), Austin, TX, USA, 2016, pp. 1317–1327, doi: 10.18653/v1/D16-1139.
[3] Gemma Team, "Gemma 2: Improving open language models at a practical size," 2024, arXiv:2408.00118. [Online]. Available: https://arxiv.org/abs/2408.00118
[4] D. Guo et al., "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning," Nature, vol. 645, pp. 633–638, 2025, doi: 10.1038/s41586-025-09422-z.
[5] D. Busbridge, A. Shidani, F. Weers, J. Ramapuram, E. Littwin, and R. Webb, "Distillation scaling laws," in Proc. 42nd Int. Conf. Mach. Learn. (ICML), 2025. [Online]. Available: https://arxiv.org/abs/2502.08606
[6] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, "GPTQ: Accurate post-training quantization for generative pre-trained transformers," in Proc. 11th Int. Conf. Learn. Represent. (ICLR), 2023. [Online]. Available: https://arxiv.org/abs/2210.17323
[7] T. Dettmers and L. Zettlemoyer, "The case for 4-bit precision: k-bit inference scaling laws," in Proc. 40th Int. Conf. Mach. Learn. (ICML), 2023, pp. 7750–7774.
[8] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-rank adaptation of large language models," in Proc. 10th Int. Conf. Learn. Represent. (ICLR), 2022. [Online]. Available: https://arxiv.org/abs/2106.09685
[9] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, 2023. [Online]. Available: https://arxiv.org/abs/2305.14314
[10] D. Biderman et al., "LoRA learns less and forgets less," Trans. Mach. Learn. Res., 2024. [Online]. Available: https://arxiv.org/abs/2405.09673

14. MELLÉKLETEK
