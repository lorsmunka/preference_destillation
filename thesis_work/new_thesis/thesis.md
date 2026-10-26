Absztrakt (magyar) + Abstract (angol)

Magyar cím: Nagy nyelvi modellek domain-specifikus desztillációja alapoktól tanított kis modellekbe
Angol cím: Domain-Specific Distillation of Large Language Models into Small Models Trained from Scratch

A nagy nyelvi modellek megjelenésével az intelligencia, a problémamegoldás és a döntéshozatal soha nem látott egyszerűséggel vált beépíthetővé a különböző szoftweres megoldásokba. Ezzel egyidejűleg az ilyen alkalmazások számításigénye és így futtatásának költsége is jelentősen megnőtt. Sok esetben fontos az ilyen modelleket lokálisan futtatni, akár szenzitív adatok, offline elérhetőség vagy megbízhatóság miatt. Ezeknél a feladatoknál viszont általában nincs szükség a modell teljes tudására, hiszen csak egy bizonyos feladatot lát el.
A dolgozat a domain-specifikus tudás desztillálását vizsgálja alapoktól tanított, jelentősen csökkentett paraméterszámú modellekbe és ügynökökbe. A vizsgált modellek elkészítéséhez és kiértékeléséhez használt saját keretrendszer jelentősen leegyszerűsíti a desztillálás folyamatát: a domain definiálását követően mindössze konfigurálással beállítható a tanítóadatok előállítása, maga a tanítás és a tanulómodell architektúrája. A tanítóadatok újrahasználhatóak több modell betanítására is.
A dolgozat több domaint is vizsgál: alap feladatokat (text sentiment classification, poszt generálás), illetve ügynök feladatokat is (matematikai feladatok megoldása és magyarázata, kódbázisban végrehajtott keresés és magyarázás), és ezek mentén igyekszik megválaszolni olyan kérdéseket, mint például: kis méretű tanulómodellek mennyire képesek elsajátítani a tanítómodell viselkedését? Különböző entrópiájú domaineknél hogyan érdemes a különböző veszteségeket használni vagy vegyíteni? Milyen módszerrel érdemes ügynököket desztillálni? Hogyan javul a teljesítmény a modell méretének függvényében? Milyen következtetések vonhatóak le nagyobb büdzséjű desztillálásokra?
A dolgozat több tanítómodellt (Gemma 3 4B, Gemma 4 12B) használ, öt domaint vizsgál részletesen, és ezek közül kettőnél saját modellhám (harness) segítségével önálló ügynököket hoz létre.

1. BEVEZETÉS

1.1 A megoldandó probléma, motiváció

1.2 A dolgozat célja (a feladatlap pontjaihoz kötve)
1.3 A dolgozat felépítése

2. A PROBLÉMA ELEMZÉSE, A SPECIFIKÁCIÓ KIDOLGOZÁSA

2.1 A problématér / jelenlegi helyzet
2.2 Követelmények (funkcionális, nem funkcionális)
2.3 Szereplők, használati esetek

3. SZAKIRODALMI ÁTTEKINTÉS – lehetséges megközelítések és megoldások
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

6. TERVEZÉS – munkafázisok és tapasztalatok
   6.1 Architektúra
   6.2 Adatmodell
   6.3 Felhasználói felület terve

7. MEGVALÓSÍTÁS
   7.1 Fejlesztői környezet, eszközök
   7.2 A fő komponensek megvalósítása

8. TESZTELÉS
   8.1 Tesztstratégia
   8.2 Tesztesetek és eredmények

9. EREDMÉNYEK BEMUTATÁSA, ÉRTÉKELÉSE, összevetés hasonló rendszerekkel

10. A MEGVALÓSÍTÁS ELEMZÉSE, ALKALMAZÁSA, TOVÁBBFEJLESZTÉSI LEHETŐSÉGEK

11. ÖSSZEFOGLALÁS (magyarul)

12. SUMMARY (angolul)

13. IRODALOMJEGYZÉK

14. MELLÉKLETEK
