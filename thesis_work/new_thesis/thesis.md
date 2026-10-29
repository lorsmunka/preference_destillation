Absztrakt (magyar) + Abstract (angol)

Magyar cím: Nagy nyelvi modellek domain-specifikus desztillációja alapoktól tanított kis modellekbe
Angol cím: Domain-Specific Distillation of Large Language Models into Small Models Trained from Scratch

A nagy nyelvi modellek megjelenésével az intelligencia, a problémamegoldás és a döntéshozatal soha nem látott egyszerűséggel vált beépíthetővé a különböző szoftveres megoldásokba. Ezzel egyidejűleg az ilyen alkalmazások számításigénye és így futtatásának költsége is jelentősen megnőtt. Sok esetben fontos az ilyen modelleket lokálisan futtatni, akár szenzitív adatok, offline elérhetőség vagy megbízhatóság miatt. Ezeknél a feladatoknál viszont általában nincs szükség a modell teljes tudására, hiszen csak egy bizonyos feladatot lát el.
A dolgozat a domain-specifikus tudás desztillálását vizsgálja alapoktól tanított, jelentősen csökkentett paraméterszámú modellekbe és ügynökökbe. A vizsgált modellek elkészítéséhez és kiértékeléséhez használt saját keretrendszer jelentősen leegyszerűsíti a desztillálás folyamatát: a domain definiálását követően mindössze konfigurálással beállítható a tanítóadatok előállítása, maga a tanítás és a tanulómodell architektúrája. A tanítóadatok újrahasználhatóak több modell betanítására is.
A dolgozat több domaint is vizsgál: alapfeladatokat (text sentiment classification, posztgenerálás), illetve ügynökfeladatokat is (matematikai feladatok megoldása és magyarázata, kódbázisban végrehajtott keresés és magyarázás), és ezek mentén igyekszik megválaszolni olyan kérdéseket, mint például: kis méretű tanulómodellek mennyire képesek elsajátítani a tanármodell viselkedését? Különböző entrópiájú domaineknél hogyan érdemes a különböző veszteségeket használni vagy vegyíteni? Milyen módszerrel érdemes ügynököket desztillálni? Hogyan javul a teljesítmény a modell méretének függvényében? Milyen következtetések vonhatóak le nagyobb büdzséjű desztillálásokra?
A dolgozat több tanármodellt (Gemma 3 4B, Gemma 4 12B) használ, öt domaint vizsgál részletesen, és ezek közül kettőnél saját modellhám (harness) segítségével önálló ügynököket hoz létre.

1. BEVEZETÉS

1.1 A megoldandó probléma, motiváció
Az utóbbi években a mesterséges intelligencia és ezen belül a természetes nyelv feldolgozására képes modellek iránti igény jelentősen megnőtt egyéni és vállalati körökben is [1]. Ilyen modelleknek gyakori felhasználási területe lehet a kódírás, döntések programatikus meghozatala vagy a tanulás támogatása. Az ilyen feladatok ellátásához jelenleg a legegyszerűbb és sok esetben az egyetlen elérhető mód, ha a különböző felhőszolgáltatóktól vásárlunk hozzáférést, legyen egy előfizetés vagy API-hívás alapú számlázás. Ez viszont sok problémát felvet, például szenzitív adatoknál vagy kritikus infrastruktúra kódjánál nem feltétlenül használhatóak az ilyen megoldások, hiszen az adatok egy külső szolgáltató szerverein kerülnek feldolgozásra, tehát kikerülnek a vállalat ellenőrzése alól. Egy másik probléma lehet a megbízhatóság, hiszen ha a modellre infrastruktúraként tekintünk, sebezhetőséget jelenthet, ha a modell válaszminősége a kontrollunkon kívül változik, például mert a szolgáltató eltérő architektúrájú hardverről szolgálja ki a modellt [2], vagy mert új védőkorlátokat vezet be [3]. Újabb probléma lehet az ilyen modellek költsége vagy elérhetősége [3]. Ahogy a legmodernebb és legjobb modellek egyre drágább áron vagy csak előfizetések mögé rejtve lesznek elérhetőek, nem feltétlenül lesz az intelligencia mindenki számára elérhető.
Nyílt súlyú modellek is elérhetőek, igaz, a legtöbb benchmarkon és valós tapasztalatok alapján a zászlóshajó zárt súlyú modellektől jelentősen elmaradnak, legalábbis ez tapasztalható 2026 októberéig [4]. Ha felhőszolgáltató infrastruktúrájáról szolgáljuk ki ezeket, a fent említett problémák csak részben mérsékelhetőek, lokális hardveren a hihetetlen memória és grafikus számításigény miatt pedig sokszor nem lehetséges a megfelelő minőségben kiszolgálni ezeket a modelleket. A 2026 júliusában megjelent Kimi K3, az egyik legjobb benchmarkeredményt elérő nyílt súlyú modell [5], 2,8 billió paraméterből áll, amelyből tokenenként 104 milliárd aktív, és a súlyai még a kiadott, 4 bites (MXFP4) formátumban is nagyjából 1,56 TB-ot foglalnak [6]. Pusztán a súlyok betöltéséhez tehát 80 GB-os A100 adatközponti minőségű GPU-kból is legalább húszra lenne szükség.
Erre kínál megoldást a desztillálás, amikor is egy nagyobb modell viselkedését próbáljuk meg közelíteni egy jelentősen kisebb számítási kapacitást igénylő tanulómodellel. Ez a folyamat veszteségesen tömöríti a tanármodell tudását, tehát általában elmondható, hogy a kisebb tanulómodell válaszminősége (a tömörítés mértékétől függően) rosszabb lesz.
A modern LLM-ek egyik legértékesebb tulajdonsága, hogy egy modell szinte bármire használható. Ha erről hajlandóak vagyunk részben lemondani, és csak egy specifikus képességét szeretnénk megtartani, a tömörítés során sokkal nagyobb hatékonysággal tudjuk megőrizni az adott domainen az eredeti modell válaszminőségét.
A dolgozat ezért egy olyan általános megoldást mutat be, amely lehetővé teszi a domain-specifikus desztillálást. Mindezt olyan példákon keresztül, melyek bemutatják a módszer hatékonyságát és előnyeit. Bemutatásra kerül például egy olyan matematikai feladatokat megoldó és magyarázó ügynök, amely akár egy középkategóriás okostelefonon is offline elfut, vagy akár egy középkategóriás laptopon is lokálisan futtatható kódmagyarázó ügynök.

1.2 A dolgozat célja
A dolgozat célja egy olyan keretrendszer megtervezése és implementálása, amellyel egy nagy nyelvi modell adott szakterületre vonatkozó tudása kis méretű, alapoktól tanított nyelvi modellekbe vagy ágensekbe desztillálható. A dolgozat vizsgálja a tudásdesztilláció és a kis méretű nagy nyelvi modellek tanításának módszereit, a nyelvi modellekre épülő ágensek felépítését, valamint a rendelkezésre álló eszközöket. A keretrendszer funkcionalitása lefedi a tanítóadatok előállítását a tanármodellel, a tanulómodell tanítását és a tanítás kiértékelését. A keretrendszer lehetővé teszi a kis modell architektúrájának és a tanítási folyamat finom szabályozását, és a desztilláció lefolyásáról részletes elemzést ad. A rendszer moduláris szemlélettel lett megtervezve, új szakterületek és ágensek külön modulként hozzáadhatóak.

1.3 A dolgozat felépítése

2. A PROBLÉMA ELEMZÉSE, A SPECIFIKÁCIÓ KIDOLGOZÁSA

Az LLM-ek egyre nagyobb szerepet játszanak a különböző digitális rendszerekben. Gyorsíthatnak vagy automatizálhatnak bizonyos szoftverfejlesztési és kiberbiztonsági feladatokat, tehát eszközként válhatnak az állami, vállalati vagy személyes infrastruktúra részévé. Vagy lehetnek egy szoftver beépített elemei, például különböző moderációs eszközökben. Az is előfordulhat, hogy az LLM maga a szoftveres megoldás fő része, például egy oktató-magyarázó alkalmazásban. Az 1.1 fejezetben részletesebben kifejtett kiszámíthatósági, megbízhatósági és költségbeli okok miatt sok esetben ideális vagy szükségszerű lenne a lokális kiszolgálás. Szintén az 1.1 fejezetben kifejtett óriási hardverigény vagy a meglévő hardveren elfutó, de jelentősen rosszabb válaszminőségű modell miatt a lokálisan futtatott általános LLM-ek nem feltétlenül jelentenek megoldást.

2.1 A probléma elemzése
A megoldandó feladat tehát egy nagy modell (későbbiekben tanármodell) sokrétű tudásának egy domainre koncentráltan hatékonyan kinyerni. Gyakorlatilag a tanármodell oly módú tömörítése, ahol nem számít a modell teljesítményének romlása más területeken, csak az számít, hogy a vizsgált területen a lehető legjobban másolja a tanárát.
A probléma több részre bontható. Először is szükség van domainek definiálására. Itt megadhatjuk azt az utasítást, amit a tanármodell követve a tanítóadatokat elő tudja állítani. Továbbá beállításra kerülnek a be- és kimeneti szótár tokenjei, a megállást jelző token, illetve ha ügynökdesztillálásról van szó, akkor a modellhám is.
Ha van egy domainünk, ahhoz, hogy a tanár viselkedése másolható legyen, szükségünk van a tanártól adatokra. A tanítóadatok a következőképpen kerülnek előállításra: a keretrendszert használva bekonfigurálhatunk akár több adatgenerálási feladatot egy sorban, amelyek szekvenciálisan végrehajtódnak. Minden adatgenerálási feladat meghatározza, hogy melyik tanármodellt használva melyik domainen mennyi példát generáljon a keretrendszer. Az adatgenerálás során a tanármodell megkap egy feladatot, amelyre lépésenként (autoregresszív módon) választ ad. Minden lépésben mentésre kerül a tanár kimeneti eloszlása, vagyis hogy a tanármodell milyen valószínűséggel hogyan folytatná a szöveget. Egy később kifejtett optimalizációs stratégiának köszönhetően (be- és kimeneti szótár csökkentés) nem szükséges a teljes eloszlást elmenteni. Ez tovább optimalizálható: az xy módszerrel tovább csökkenthető, amelynek során csak a legfontosabb n token kerül mentésre. Szükségesség híján a dolgozat ezt a stratégiát nem vizsgálja. A tanármodell vagy a feladat befejezését jelző, domainben konfigurált tokent használva állítja le a válaszadást, vagy ha túllép egy válaszhossz-keretet, a modell megállításra, a példa pedig elvetésre kerül.
A folyamat végén kapni fogunk egy mappát a futás nevével, amelyben JSONL kötegeket kapunk. A JSONL fájlon belül alapértelmezett beállítások mellett 32 JSON-objektumot találunk, amelyek egy-egy probléma tanítóadatát tárolják. Egy ilyen probléma adatait tartalmazó JSON tartalmazza a feladatot szövegként, a modell válaszát szövegként, illetve egy steps listát, amelyben lépésenként el van mentve a tanármodell által legvalószínűbb következő tokennek vélt token található, illetve a domainen vizsgált kimeneti szótár tokeneinek eloszlása. Összességében tehát ez a folyamat adatgenerálási feladatonként külön mappába, rendszerezve tárolja az adatokat olyan formában, hogy az x. fejezetben bemutatott desztillációs tanítás végrehajtható legyen a tanulómodellen.
A következő kérdés a tanulómodell architektúrájának megválasztása. A keretrendszer alapértelmezett modellje egy sűrű Gemma 3 4B megoldásait használó transformer modell, amelynek konfigurálhatjuk rejtett dimenzióját (hidden dimension), figyelemfejeinek (attention head) és rétegeinek (layers) számát sok egyéb mellett. Ezenfelül saját modellt is átadhatunk a keretrendszernek, ha saját architektúrával szeretnénk kísérletezni, hiszen a keretrendszernek a könnyű, gyors, kontrollált és költséghatékony kísérletezésnek az elősegítése is célja.
A tanítóadatok előállításához hasonlóan a tanítás is egy soralapú szekvenciális rendszerre épül. A tanítóadatok előállításához és a diákmodellhez hasonlóan a desztilláció ezen része is teljes mértékben konfigurálható kell hogy legyen, hiszen a kísérleteket csak így lehet nagy számban, megbízhatóan és pontosan végrehajtani. Beállítható kell legyen a vizsgált veszteségvegyítésen felül számos más paraméter.
A modell betanulása után pedig fontos lépés a tanítás és a modell eredményeinek a kielemzése. A keretrendszer alaposan naplózza az összes fontos paramétert adatgenerálás és tanítás közben is. Visszakövethetjük a különböző folyamatokkal eltöltött időt, tanulás folyamán a veszteség- és pontosságértékeket, epochok közben a mini tesztadatsoron történő kiértékelést, epochok végén a teljes tesztadatokon történő értékelést.
Az utolsó problémakör pedig a kész modellek integrálása különböző felületekbe, legyen az egy terminálos alkalmazás, amely kielemezi egy bemeneti szöveg hangulatát, vagy egy webalkalmazás, amely mutatja a modell és a modellhám által tett lépéseket egy chatfelületen.

2.2 Funkcionális és nem funkcionális követelmények - DRAFT, review before subbmitting
A rendszerrel szemben támasztott követelmények két csoportra bonthatóak. A funkcionális követelmények azt írják le, hogy mit kell tudnia a rendszernek, a nem funkcionális követelmények pedig azt, hogy milyen minőségben és milyen korlátok között. A követelmények a keretrendszerre és a rá épülő két ügynökre vonatkoznak, teljesülésüket a 8. fejezet ellenőrzi.
Funkcionális követelmények:
F1 – Új domain külön modulként hozzáadható, amely megadja a feladatokat, a kiértékelés módját, valamint a domain be- és kimeneti szótárát.
Nem funkcionális követelmények:
NF1 – Megbízhatóság: megszakításkor legfeljebb az utolsó mentés óta végzett munka vész el, a felhalmozott adatfájlok soha nem íródnak felül, és a keretrendszer elutasítja a nem egyező szótárral készült tanítóadatokat, illetve a nem egyező állapotból történő folytatást.

2.3 Használati esetek - DRAFT, review before subbmitting
A dolgozatban leírt rendszerek és azok eredményei többféle módon is felhasználhatóak.
Használhatják kutatók különböző architektúrák szisztematikus tesztelésére desztillált adatokon, domainenként. Használhatják különböző desztillációs módszerek és tömörítő módszerek tesztelésére és összehasonlítására is.
Használhatják fejlesztők, akik különböző vállalati, infrastrukturális vagy személyes feladatokra vagy problémákra keresnek egy hatékony, lokálisan futtatható megoldást.
A kész modelleket pedig természetesen használhatják végfelhasználók különböző felhasználási területeken.
A főbb használati esetek szereplőnként:
Kutató vagy fejlesztő:
HE1 – Tanítóadatok előállítása: kiválasztja a tanármodellt, és konfigurációval elindítja az adatok előállítását.

3. SZAKIRODALMI ÁTTEKINTÉS – LEHETSÉGES MEGKÖZELÍTÉSEK ÉS MEGOLDÁSOK

A következőkben bemutatásra kerülnek a jelenlegi módszerek, azok hiányosságai, illetve a domain-specifikus desztilláció által kínált megoldás.

3.1 Alapfogalmak
3.1.1 Nagy nyelvi modellek (LLM), transformer
LLM
Transformer (decoder-only)
NF4, bf16 (lásd 3.2.2)
3.1.2 Token, be- és kimeneti szótár
Token, tokenizálás
Be- és kimeneti szótár
3.1.3 Autoregresszív generálás, kimeneti eloszlás, hőmérséklet
Autoregresszív generálás
Kimeneti eloszlás (softmax)
Hőmérséklet
3.1.4 Veszteségek
Veszteség
Kereszt-entrópia (cross-entropy)
Kullback–Leibler-divergencia
A veszteségek vegyítése
3.1.5 Ügynök, hám, kontextus
Ügynök
Hám (harness)
Kontextus
3.1.6 Kiértékelés
Feladatpontosság
Helyes előzménnyel mért tokenpontosság (teacher forcing)
Saját előzménnyel mért tokenpontosság

3.2 Meglévő megoldások, hasonló rendszerek
3.2.1 Általános desztilláció
A tudásdesztilláció (knowledge distillation) egy tömörítési módszer, amely egy kisebb modellel próbálja minél pontosabban közelíteni a desztillálni kívánt nagyobb modellt. A kis modellt tanulómodellnek, a nagy modellt tanármodellnek hívják. A klasszikus tudásdesztilláció alapötletét Hinton és társai [7] fogalmazták meg, melynek lényege, hogy a tanulómodellt nemcsak a helyes válaszra tanítjuk meg, hanem a tanármodell teljes kimeneti eloszlására is. A tanulómodell így nemcsak a helyes választ, de a tanármodell bizonytalanságát, úgynevezett "dark knowledge"-t is képes megtanulni. Ez a bizonytalanság érdekes többletinformációt is magában hordoz, hiszen azt is megmutatja, hogy melyik rossz válaszok állnak közel a jóhoz. Ráadásul a nagy nyelvi modellek működéséből fakadóan kifejezetten fontos ez az információ, ugyanis gyakran előfordul, hogy egy LLM használata közben nem a legvalószínűbb, hanem valamelyik más valószínű szóval folytatja a válaszadást.
Egy másik gyakran használt tudásdesztillálási módszer nagy nyelvi modelleknél az eloszlás nélkül, csak a tanármodell kimeneti szövegén történő tanítás, az úgynevezett szekvenciaszintű desztilláció [8].
A nagy nyelvi modelleknél a desztillációt ma jellemzően általános céllal használják, tehát a tanulómodelltől azt várjuk, hogy a tanár minden képességét megtartsa. Erre példa a Gemma 2 modellcsalád [9], ahol a 2 és 9 milliárd paraméteres modelleket már desztillációval tanították. A szerzők kísérletében egy 7 milliárd paraméteres tanárból desztillált 2 milliárd paraméteres modell három benchmark átlagában 67,7 pontot ért el, szemben az ugyanannyi tokenen, desztilláció nélkül tanított változat 60,3 pontjával.
Vannak természetesen specifikusabban desztillált modellek, ilyenek például a DeepSeek-R1-Distill modellek [10], amelyeknél nagyjából 800 ezer, főként a DeepSeek-R1 érvelő (reasoning) modell által generált példán tanították tovább a kisebb, 1,5 és 70 milliárd paraméter közötti Qwen és Llama modelleket, kifejezetten az érvelési képességek átadására fókuszálva. Ebben a kísérletben eloszlás nélkül, csak a tanármodell szövegén tanították a tanulómodelleket.
Busbridge és társai [11] szerint a desztillált modell teljesítménye előre jelezhető a tanulómodell méretéből és a tanításra fordított számítási keretből, és ha a modell általános képességeit nézzük, akkor ez a tömörítés veszteséges. Ha viszont lemondunk az általánosságról, és csak egy szűk feladatra (domain-re) szeretnénk fókuszálni, a szükséges tudás egy jóval kisebb modellbe is beférhet.

3.2.2 Kvantálás
A desztillálás egy olyan tömörítőmódszer, amely a paraméterszám csökkentésére koncentrál, ezzel szemben a kvantálás (quantization) a meglévő súlyok helyigényét csökkenti. Míg egy nem kvantált LLM súlyait általában 16 biten szokták tárolni, kvantáláskor ez lecsökkenhet 8 vagy 4 bites precizitásra. A kvantálás a modell teljesítményének romlásával járhat, viszont így jelentősen nagyobb paraméterszámú modelleket is képesek vagyunk futtatni lokálisan, ezért az ilyen modellek futtatására ez az egyik legelterjedtebb módszer.
Előnye, hogy nem szükséges újratanítani a modellt: a GPTQ módszerrel [12] például egy 175 milliárd paraméteres modell nagyjából négy GPU-óra alatt 3 vagy 4 bitesre kvantálható, elhanyagolható pontosságvesztéssel, így egyetlen GPU-n is futtatható.
A kvantálás viszont csak a modell számábrázolásának pontosságát csökkenti, a modell felépítése és így a paraméterek száma változatlan marad, ugyanúgy a teljes modelltudás lesz tömörítve, csak kevesebb memóriát foglal. Dettmers és Zettlemoyer [13] kutatásából az is kiderül, hogy az adott memóriakeret mellett szinte mindig a 4 bites pontosság adja a legjobb eredményt. Tehát a kvantálással nagyjából négyszeres memóriacsökkenés érhető el, a dolgozatban vizsgált domain-specifikus desztillálásnál viszont a tanulómodellek a tanármodellnél akár százszor–húszezerszer kevesebb paraméterből is állhatnak. A desztillálás és kvantálás nem feltétlenül kizáró optimalizációs módszerek, tehát egy desztillált modellt lehet kvantálni is.

3.2.3 Modellfinomhangolás
A finomhangolás (fine-tuning) során egy már előtanított nagy nyelvi modellt tanítunk tovább egy szűkebb szakterületre, a domainnek megfelelő adatokon vagy környezetben (environment), például megerősítéses tanulással (RL). Előnye, hogy nem kell a modellt alapoktól tanítani, hiszen az alapvető nyelvi képességeket a modell már elsajátította. Ez a technika viszont a már létező előtanított (pre-trained) modellekhez köt, amelyek sok esetben nem is léteznek a dolgozatban vizsgált méretekben.
A finomhangolás költsége jelentősen csökkenthető paraméterhatékony módszerekkel. Ezek közül az egyik legelterjedtebb a LoRA [14], amely az eredeti súlyokat befagyasztja, és csak kis, alacsony rangú mátrixokat tanít mellettük. A szerzők ezzel a módszerrel a 175 milliárd paraméteres GPT-3 esetén a tízezred részére csökkentették a tanítandó paraméterek számát és a harmadára a szükséges GPU-memóriát. Az így kapott mátrixok utólag pedig beolvaszthatóak az eredeti súlyokba, így a folyamat végére nem változik a modell mérete.
A QLoRA [15] ezt a 3.2.2 fejezetben bemutatott kvantálással kombinálja. A módszer a modell súlyainak 4 bitesre kvantálása után, befagyasztott modell mellett tanítja a LoRA-mátrixokat. Így egy 65 milliárd paraméteres modell finomhangolásának memóriaigénye több mint 780 GB-ról 48 GB alá csökken, lényeges teljesítményromlás nélkül. A finomhangolás viszont csak a súlyok értékét változtatja meg, azok számát és elrendezését nem, tehát a finomhangolt modell futtatásához ugyanakkora hardver kell, mint az általános, nem finomhangolt modellhez.
Ezenfelül a domainre hangolás az általános képességek rovására is mehet: Biderman és társai [16] például a Llama-2-7B modellt finomhangolták tovább programozási és matematikai feladatokra, és azt tapasztalták, hogy a modell teljes finomhangolásakor jobban tanul a célterületen, viszont többet felejt a többi mért képességéből. Ezzel szemben a LoRA kevesebbet felejt, de kevesebbet is tanul.

3.2.4 Ügynökdesztillálás
A mai modern LLM-rendszerek nemcsak válaszolnak, hanem cselekedni is tudnak. Ez a modellhámnak (harness) köszönhető, amelynek az LLM szöveges formában, például tagek segítségével képes utasításokat adni, a hám pedig végrehajtja ezeket. Ennek egyik meghatározó korai példája a ReAct [17], ahol a modell felváltva generál gondolatmenetet és cselekvéseket, a cselekvésekkel pedig külső forrásokat, például egy egyszerű Wikipédia-API-t ér el. A modellek cselekvőképessége, vagyis hogy ügynökként tudnak viselkedni, a lehetőségek új osztályát nyitja meg. Erre példa a Claude Code [18], amely lehetővé teszi, hogy a modell fájlokat hozzon létre és szerkesszen, kezelje a saját kontextusát, és Bash-parancsokkal vezérelje a számítogép bizonyos funkcióit. A dolgozatban különös figyelmet kapott ez a terület, hiszen meglepően hasznos tud lenni egy cselekvésre képes kis LLM is.
Az ügynökök desztillálása már létező kutatási terület: Kang és társai [19] például egy 32 milliárd paraméteres tanármodell ügynökként végzett lépéssorozatain finomhangoltak 0,5 és 7 milliárd paraméter közötti előtanított modelleket, és már a 0,5 milliárd paraméteres ügynök is versenyképes volt a következő méretkategóriába tartozó, csak a tanár gondolatmenetén (chain-of-thought) tanított modellel. Ezek a tanulómodellek viszont előtanított modellek, a dolgozatban vizsgált ügynökök ennél jóval kisebbek, és alapoktól tanulnak.

3.3 Technológiák, módszerek összevetése
3.4 Következtetések

4. A MEGOLDÁSI MÓDSZER KIVÁLASZTÁSA ÉS INDOKLÁSA

4.1 Domain-specifikus desztillálás alapoktól tanított kis méretű modelleknél
A fenti módszerekkel szemben a dolgozatban tárgyalt, alapoktól tanított kis méretű modellekbe történő domain-specifikus desztillálás lehetővé teszi, hogy egy, az adott feladatra szabott, a nagy nyelvi modellek világában kis méretűnek számító modellel megtartsuk a tanármodell domain-specifikus tudásának nagy részét. Tekinthetünk a módszerre egy olyan tömörítési eljárásként, amelynek nem célja a modell teljes tudásának megőrzése, a tanulómodell a teljes kapacitását egy specifikus, szűk területre fordíthatja, így a méretéhez képest kifejezetten jó teljesítményt nyújthat. Mivel a modell alapoktól tanul, ezért a tanulómodell architektúráját is szabadon választhatjuk. A dolgozatban tárgyalt megoldás ezt ki is használja: például a modell be- és kimeneti szótárának redukciójával jelentősen csökkenthető a szükséges paraméterek száma, hiszen kis modelleknél a szótárhoz tartozó beágyazások a paraméterek jelentős részét teszik ki. A Gemma 3 270M modell 270 millió paraméteréből például 170 millió a 256 ezer tokenes szókincs beágyazásaihoz tartozik [20]. Az alapoktól tanulás hátránya, hogy a modell nem indul "alap megértéssel". Ez viszont a megfelelő veszteségek vegyítésével, például Kullback–Leibler-divergencia alapú veszteséggel, a tanármodell "dark knowledge"-ének átvételével valamelyest javítható. Hinton és társai [7] egy beszédfelismerési kísérletében például a tanármodell eloszlásán (soft targets) tanított modell a tanítóadatok mindössze 3%-ával is 57,0%-os pontosságot ért el, szemben a csak a helyes válaszokon tanított változat 44,5%-ával és a teljes adathalmazon tanított modell 58,9%-ával. Ráadásul ezek a modellek olyan kis méretűek is lehetnek, hogy a tanítóadatok előállításának és a modell betanításának költségeit akár egy magánszemély is fedezni tudja: a dolgozat két ügynökének tanítóadatai összesen kevesebb mint 150 dollárnyi bérelt GPU-időbe (A100 és RTX PRO 6000) kerültek, a tanulómodellek tanítása pedig többnyire lokálisan, egy RTX 4070 videokártyán, néhány tíz óra alatt történt.
A dolgozatban bemutatott keretrendszer a teljes desztillációs folyamatot kezeli, melynek lépései a következők:
(1) a domain definiálása (feladatok, kiértékelés, be- és kimeneti szótár stb.),
(2) a tanítóadatok előállítása (konfigurációval, soralapú megoldással, ahol többek között a tanármodell is beállítható),
(3) a tanulómodell definiálása és tanítása (a tanítás folyamata szintén soralapú, számos paraméter konfigurálható),
(4) a kiértékelés.

4.2 Ügynökdesztillálás
A dolgozat két fő példán keresztül mutatja be az ügynökdesztillálást. Az egyik egy egyszerű matematikai feladatokat megoldó és magyarázó LLM, ahol az ügynöknek kezelnie kell a saját kontextusát: el kell döntenie, hogy egy kérdésre válaszol, vagy a következő feladatra tér át, illetve el kell utasítania a feladathoz nem kapcsolódó üzeneteket. A másik vizsgált ágens egy olyan modell és hám kombinációja, amely képes egy 12–15 fájlból álló HTML- vagy Python-projektben megtalálni és szükség esetén megmagyarázni bizonyos kódrészeket.
Könnyen látható, hogy ezek a feladatkörök egyszerűen bővíthetőek és skálázhatóak úgy, hogy rendkívül hasznosak legyenek. Képzeljünk csak el egy offline, ingyenesen, akár telefonon is futtatható matematikai magyarázó ügynököt, amely a lemaradó gyerekeket segíthetné a tanórára való készülésben. Vagy egy olyan kódoló ügynököt, amely lokálisan, egy laptop videokártyáján pillanatok alatt ellátja az egyszerűbb feladatokat, ezzel megspórolva a külső szolgáltató költségét.

5. RÉSZLETES SPECIFIKÁCIÓ

6. TERVEZÉS – MUNKAFÁZISOK ÉS TAPASZTALATOK
7. MEGVALÓSÍTÁS

8. TESZTELÉS

9. EREDMÉNYEK BEMUTATÁSA, ÉRTÉKELÉSE, ÖSSZEVETÉS HASONLÓ RENDSZEREKKEL

10. A MEGVALÓSÍTÁS ELEMZÉSE, ALKALMAZÁSA, TOVÁBBFEJLESZTÉSI LEHETŐSÉGEK

11. ÖSSZEFOGLALÁS (magyarul)

12. SUMMARY (angolul)

13. IRODALOMJEGYZÉK

[1] J. S. Allen, "Monitoring AI adoption in the U.S. economy," FEDS Notes, Board of Governors of the Federal Reserve System, Washington, DC, USA, Apr. 3, 2026, doi: 10.17016/2380-7172.4032.
[2] Anthropic, "A postmortem of three recent issues," Anthropic Engineering, Sep. 17, 2025. Accessed: Oct. 29, 2026. [Online]. Available: https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues
[3] Anthropic, "Redeploying Fable 5," Anthropic News, Jun. 30, 2026. Accessed: Oct. 29, 2026. [Online]. Available: https://www.anthropic.com/news/redeploying-fable-5
[4] Artificial Analysis, "Announcing Artificial Analysis Intelligence Index v5," Oct. 27, 2026. Accessed: Oct. 29, 2026. [Online]. Available: https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v5
[5] Artificial Analysis, "Kimi K3 achieves #3 in the Artificial Analysis Intelligence Index, comparable to Opus 4.8 and GPT-5.5," Jul. 17, 2026. Accessed: Oct. 29, 2026. [Online]. Available: https://artificialanalysis.ai/articles/kimi-k3-achieves-3-in-the-artificial-analysis-intelligence-index-comparable-to-opus-4-8-and-gpt-5-5
[6] Moonshot AI, "Kimi-K3," Hugging Face model card, 2026. Accessed: Oct. 29, 2026. [Online]. Available: https://huggingface.co/moonshotai/Kimi-K3
[7] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," presented at the NIPS 2014 Deep Learning Workshop, 2015. [Online]. Available: https://arxiv.org/abs/1503.02531
[8] Y. Kim and A. M. Rush, "Sequence-level knowledge distillation," in Proc. 2016 Conf. Empirical Methods Natural Lang. Process. (EMNLP), Austin, TX, USA, 2016, pp. 1317–1327, doi: 10.18653/v1/D16-1139.
[9] Gemma Team, "Gemma 2: Improving open language models at a practical size," 2024, arXiv:2408.00118. [Online]. Available: https://arxiv.org/abs/2408.00118
[10] D. Guo et al., "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning," Nature, vol. 645, pp. 633–638, 2025, doi: 10.1038/s41586-025-09422-z.
[11] D. Busbridge, A. Shidani, F. Weers, J. Ramapuram, E. Littwin, and R. Webb, "Distillation scaling laws," in Proc. 42nd Int. Conf. Mach. Learn. (ICML), 2025. [Online]. Available: https://arxiv.org/abs/2502.08606
[12] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, "GPTQ: Accurate post-training quantization for generative pre-trained transformers," in Proc. 11th Int. Conf. Learn. Represent. (ICLR), 2023. [Online]. Available: https://arxiv.org/abs/2210.17323
[13] T. Dettmers and L. Zettlemoyer, "The case for 4-bit precision: k-bit inference scaling laws," in Proc. 40th Int. Conf. Mach. Learn. (ICML), 2023, pp. 7750–7774.
[14] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-rank adaptation of large language models," in Proc. 10th Int. Conf. Learn. Represent. (ICLR), 2022. [Online]. Available: https://arxiv.org/abs/2106.09685
[15] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, 2023. [Online]. Available: https://arxiv.org/abs/2305.14314
[16] D. Biderman et al., "LoRA learns less and forgets less," Trans. Mach. Learn. Res., 2024. [Online]. Available: https://arxiv.org/abs/2405.09673
[17] S. Yao et al., "ReAct: Synergizing reasoning and acting in language models," in Proc. 11th Int. Conf. Learn. Represent. (ICLR), 2023. [Online]. Available: https://arxiv.org/abs/2210.03629
[18] Anthropic, "Claude Code overview," Claude Code Docs. Accessed: Oct. 28, 2026. [Online]. Available: https://code.claude.com/docs/en/overview
[19] M. Kang, J. Jeong, S. Lee, J. Cho, and S. J. Hwang, "Distilling LLM agent into small models with retrieval and code tools," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 38, 2025. [Online]. Available: https://arxiv.org/abs/2505.17612
[20] O. Lacombe, K. Kenealy, K. Black, R. Kumar, F. Visin, and J. Zhang, "Introducing Gemma 3 270M: The compact model for hyper-efficient AI," Google Developers Blog, Aug. 14, 2025. Accessed: Oct. 28, 2026. [Online]. Available: https://developers.googleblog.com/en/introducing-gemma-3-270m/

14. MELLÉKLETEK
