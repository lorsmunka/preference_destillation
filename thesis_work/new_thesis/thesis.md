Absztrakt (magyar) + Abstract (angol)

Magyar cím: Nagy nyelvi modellek domain-specifikus desztillációja alapoktól tanított kis modellekbe
Angol cím: Domain-Specific Distillation of Large Language Models into Small Models Trained from Scratch

A nagy nyelvi modellek megjelenésével az intelligencia, a problémamegoldás és a döntéshozatal soha nem látott egyszerűséggel vált beépíthetővé a különböző szoftweres megoldásokba. Ezzel egyidejűleg az ilyen alkalmazások számításigénye és így futtatásának költsége is jelentősen megnőtt. Sok esetben fontos az ilyen modelleket lokálisan futtatni, akár szenzitív adatok, offline elérhetőség vagy megbízhatóság miatt. Ezeknél a feladatoknál viszont általában nincs szükség a modell teljes tudására, hiszen csak egy bizonyos feladatot lát el.
A dolgozat a domain-specifikus tudás desztillálását vizsgálja alapoktól tanított, jelentősen csökkentett paraméterszámú modellekbe és ügynökökbe. A vizsgált modellek elkészítéséhez és kiértékeléséhez használt saját keretrendszer jelentősen leegyszerűsíti a desztillálás folyamatát: a domain definiálását követően mindössze konfigurálással beállítható a tanítóadatok előállítása, maga a tanítás és a tanulómodell architektúrája. A tanítóadatok újrahasználhatóak több modell betanítására is.
A dolgozat több domaint is vizsgál: alap feladatokat (text sentiment classification, poszt generálás), illetve ügynök feladatokat is (matematikai feladatok megoldása és magyarázata, kódbázisban végrehajtott keresés és magyarázás), és ezek mentén igyekszik megválaszolni olyan kérdéseket, mint például: kis méretű tanulómodellek mennyire képesek elsajátítani a tanítómodell viselkedését? Különböző entrópiájú domaineknél hogyan érdemes a különböző veszteségeket használni vagy vegyíteni? Milyen módszerrel érdemes ügynököket desztillálni? Hogyan javul a teljesítmény a modell méretének függvényében? Milyen következtetések vonhatóak le nagyobb büdzséjű desztillálásokra?
A dolgozat több tanítómodellt (Gemma 3 4B, Gemma 4 12B) használ, öt domaint vizsgál részletesen, és ezek közül kettőnél saját modellhám (harness) segítségével önálló ügynököket hoz létre.

1. BEVEZETÉS

1.1 A megoldandó probléma, motiváció
Az utóbbi években a mesterséges inteligencia és ezen belül természetes nyelv feldolgozására képes modellek iránti igény jelentősen megnőtt. Ilyen modellekenek gyakori felhasználsi területe lehet a kódírás, döntések progrematikus meghozatala, vagy a tanulás támogatása. Az ilyen feladatok ellátásához jelnleg a legegyszerűbb és sok esetben az egyetlen elérhető mód ha a különböző felhő szolgáltatóktól vásárlunk hozzáférést, legyen egy előfizetés, vagy API hívás alapú számlázás. Ez viszont sok problémát felvet, például szenzitív adatoknál, vagy kritkus infrastruktúra kódjánál nem feltélenül használhatóak az ilyen megoldások, hiszen adatszivárgás léphet fel. Egy másik probléma lehet a megbízhatóság, hiszen ha a modellre infrastruktúraként tekintünk sebezhetőséget jelenthet ha modell válaszminősége a kontrolunkon kívül változik, például mert a szolgáltató eltérő architektúrájú hardwerről szolgálja ki a modelt, vagy mert új vídőkorlátokat vezet be. Újabb problélma lehet az ilyen modellek költsége vagy elérhetősége. Ahogy a legmodernebb és legjobb modellek egyre drágább áron, vagy csak előfizetések mögé rejtve lesznek elérhetőek nem feltétlenül lesz az intellgiencia mindenki számára elérhető.
Nyílt sulyú modellek is elérhetőek, igaz, a legtöbb banchmárkon és valós tapasztalatok alapján a zázlóshalyó zárt súlyú modellektől egyhén elmaradnak. Ha felhőszolgáltató infrastruktúrájáról szolgáljuk ki ezeket a fent említett problémák csak részben mérsékelhetőek, lokális hardweren a hihetleten memória és grafikis számítás igény miatt pedig sokszor nem lehetséges a megfelelő minőségben kiszolgálni ezeket a modelleket.
Erre kínál megoldást a desztillálás, amikor is egy nagyobb model viselkedését próbáljuk meg közelíteni egy jelentősen kisebb számítási kapacitást igénylő tanulómodellek. Ez a folyamat veszteségesen tömöríti a tanár model tudását, tehát álltalában elmondható, hogy a kisebb tanulómodell válaszminősge (a tömörítés mértékétől függően) roszabb lesz.
A modern LLM-ek egyik legértékesebb tulajdonsága, hogy egy model szinte bármire használható. Ha erről hajlandóak vagyun részben lemondani, és csak egy specifikus képességét szeretnénk megtartani, a tömörítés során sokkal nagyobb hatékonysággal tudjuk megőrizni az adott domainen az eredeti model válaszminőségét.
A dolgozat célja egy olyan álltalános megoldás bemutatása, amely lehetővé teszi a domain specifikus desztillálást. Mindezt olyan példákon keresztül, melyek bemutatják a módszer hatékonyságát és előnyeinyeit. Bemutatásra kerül például egy olyan matematika feladat megoldó és magyarázó ügynök amely akár egy középkategóriás okostelefonoin is offline elfut, vagy akár egy középkategóriás laptopon is lokálisan futtatható kód értelmező ügynök.

1.2 A dolgozat célja
A dolgozat célja egy olyen keretrendszer megtervezése és implementálása, amellyel egy nagy nyelvi modell adott szakterületre vonatkozó tudása kis méretű, nulláról tanított nyelvi modellekbe vagy ágensekbe desztillálható. A dolgozat vizsgálja a tudásdesztilláció és a kis méretű nagy nyelvi modellek tanításának módszereit, a nyelvi modellekre épülő ágensek felépítését, valamint a rendelkezésre álló eszközöket. A keretrendszer funkcionalitása lefedi a tanítóadatok előállítását a tanármodellel, a tanulómodell tanítását és a tanítása kiértékelését. A keretrendszer lehetővé teszi a kis modell architektúrájának és a tanítási folyamat finom szabályozását, és a desztilláció lefolyásáról részletes elemzést ad. A rendszert moduláris szemlélettel lett megtervezve, új szakterületek és ágensek külön modulként hozzáadhatóak.

1.3 A dolgozat felépítése

2. A PROBLÉMA ELEMZÉSE, A SPECIFIKÁCIÓ KIDOLGOZÁSA

Az LLM-ek egyre nagyobb szerepet játszanak a különböző digitális rendszerekben. Gyorsíthat vagy autómatizálhat bizonyos szoftverfejlesztési és kiberbiztonsági feladatokat, tehát eszközként válhat az állami, vállalati, vagy személyes infrastruktúra részévé. Vagy lehet egy szoftwer beépített eleme, például különböző moderációs eszközökben. Az is előfordulhat hogy az LLM maga a szoftweres megoldás fő része, például egy oktató-magyarázó alkalmazásban. Az 1.1 fejezetben részletesebben kifejtett kiszámíthatósági, megbízhatósági és költségbeli okok miatt sok esetben ideális vagy szükségszerű lenne a lokális kiszolgálás. Szintén az 1.1 fejezeteben kifejtett óriási hardwerigény, vagy a meglévő hardveren elfutú, de jelentősen roszabb válaszminőségű model miatt a lokálisan futatott állatlános LLM-ek nem feltétlenül jelentenek megoldást. A következőkben bemutatásra kerülnek a jelenlegi módszerek, azonak a hiányoságai illetve a domain specifikus desztilláció álltal kínált megoldás.

2.1 Álltalános desztiláció
2.2 Qvantálás
2.3 model finomhangolás
2.4 Domain specifikus desztillálás
2.5 Ügynök desztilláálás
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

14. MELLÉKLETEK
