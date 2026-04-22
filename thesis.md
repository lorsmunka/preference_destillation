Domain-specifikus LLM desztillációs pipeline

Abstract
A nagy nyelvi modellek általános képességei sok feladatnál hasznosak, de gyakran túl nagy erőforrásköltséggel járnak olyan ismétlődő, szűk domainhez kötött feladatoknál, ahol a lehetséges bemenetek és kimenetek jól körülhatárolhatók. A dolgozat egy könnyen módosítható, több domainre alkalmazható desztillációs pipeline-t mutat be, amely nagyobb teacher modellek domain-specifikus tudását kisebb autoregresszív transformer student modellekbe desztillálja. A rendszer hosszabb, felügyelet nélküli kutatási futtatásokra is alkalmas: konfigurációval és kisebb domain-specifikus kódkiegészítésekkel cserélhető a teacher modell, a domain, a kimeneti szótár mérete, az architektúra és a tanítási stratégia.
A dolgozat három fő optimalizációs irányt vizsgál. Az első a domain-specifikus bemeneti szótár csökkentése, amely a kisebb modellek egyik aránytalanul nagy paraméterblokkjának, az embedding rétegnek a méretét csökkenti. A második a kimeneti szótár és az output projection szűkítése, amely jelentős paramétercsökkenést eredményez, de a KL divergenciára épülő desztilláció miatt óvatos kompromisszumot igényel: túl szűk kimeneti szótár esetén a teacher eloszlásának információtartalma csökkenhet. A harmadik a paraméterezhető tanítási pipeline, amely támogatja a Kullback-Leibler divergencia és cross-entropy veszteségek arányának annealingjét, valamint a distillation temperature annealinget is.
A pipeline több feladattípuson kerül vizsgálatra: Reddit kommentek strukturált JSON alapú sentiment-kategorizálásán, matematikai szöveges feladatok strukturált megoldásán, valamint Reddit kommentből kiinduló rövid posztgeneráláson. A strukturált feladatok azt mutatják, hogy a csökkentett ki- és bemeneti szótárú autoregresszív student modellek stabilan képesek kötött formátumú kimenetek előállítására. A szabadabb posztgenerálási domain ezzel szemben azt vizsgálja, hogy a teacher eloszlásának követése milyen többletinformációt ad a pusztán következő tokenre optimalizáló cross-entropy veszteséghez képest. A dolgozat eredményei alapján a domain-specifikus vocabulary redukció és a loss-, valamint temperature-annealinget használó desztillációs tanítás ígéretes irány a nagy modellek ismétlődő, szűk feladatokra specializált kiváltására. A módszer általánosíthatósága további domaineken, teacher modelleken és skálázási kísérleteken keresztül vizsgálandó.

1. Bevezetés
A nagy nyelvi modellek (LLM-ek) az elmúlt években jelentős áttörést hoztak a természetes nyelvfeldolgozásban. Képesek szöveget generálni, klasszifikálni, strukturált outputot előállítani és összetett instrukciókat követni. Ugyanakkor ezek a modellek erőforrásigényesek: nagy memóriaigényt, jelentős számítási kapacitást és sok esetben külső API-függőséget igényelnek. Számos alkalmazásban nincs szükség a teljes általános intelligenciára, hanem csak egy gyakran ismételt, szűk domainhez kötött részfeladat megbízható végrehajtására.
A dolgozat célja egy olyan desztillációs pipeline kifejlesztése és vizsgálata, amely nagy nyelvi modellek domain-specifikus viselkedését kisebb, hatékonyabb autoregresszív student modellekbe ülteti át. A rendszer nem egyetlen feladatra írt klasszifikátor, hanem konfigurálható kutatási eszköz: a domain, a teacher modell, a prompt, a stop token, a kimeneti szótár, a student architektúra és a tanítási stratégia külön módosítható. Ez lehetővé teszi, hogy a dolgozat ne csak egy adott domaint vagy modellt, hanem különböző optimalizációs döntések hatását is vizsgálja.

1.1 Probléma és motiváció
Az LLM-ek használata egyszerű, jól definiált feladatokra gyakran nem gazdaságos. Egy sentiment klasszifikáció, egy strukturált JSON válasz generálása, egy kötött matematikai levezetés vagy egy rövid domain-specifikus szöveggenerálási lépés nem feltétlenül igényli egy többmilliárd paraméteres általános célú modell teljes kapacitását. Ennek ellenére sok alkalmazás ilyen modelleket használ API hívásokon keresztül, magas költség, nagyobb válaszidő és külső szolgáltatótól való függés mellett. A mixture of experts (MoE) modellek részben csökkenthetik az aktív számítási igényt, de a teljes paraméterkészlet tárolása továbbra is jelentős infrastruktúrát igényel.
A desztilláció alternatívát kínál: egy kisebb modell megtanulhatja a nagyobb modell viselkedését egy adott feladatra, és azt töredék erőforrással képes végrehajtani [2]. Ez lehetővé teszi:
- on-device futtatást (pl. telefonon, edge eszközökön)
- jelentősen alacsonyabb latenciát és költséget
- függetlenséget külső API-któl
A dolgozat három domaint vizsgál: Reddit kommentek strukturált sentiment klasszifikációját, matematikai szöveges feladatok megoldását és Reddit posztgenerálást. A három domain eltérő kimeneti szerkezetet képvisel: kötött JSON objektumot, determinisztikus számítási scaffoldot és szabadabb természetes nyelvi generálást. Ez a különbség fontos, mert a desztillációs módszerek hatása nem feltétlenül azonos kevés lehetséges tokent tartalmazó kimeneteknél és nagyobb, puhább eloszlással rendelkező generatív feladatoknál. Továbbá a predikció "puhasága" vagy "keménysége", vagyis a következő token jóslásának eloszlása is nagyban befolyásolja az ideális tanítási beállítások kiválasztását.
A desztilláció előtt költségelemzés szükséges: elég gyakran hívják-e az adott részfeladatot ahhoz, hogy megérje a desztillációs befektetés? Szükséges-e edge eszközön vagy lokálisan futtatni? Elég szűk-e a domain ahhoz, hogy a bemeneti és kimeneti vocabulary jelentősen csökkenthető legyen? Szükséges-e autoregresszív transformer, vagy egy egyszerűbb klasszifikátor is elegendő lenne?
A dolgozat olvasása segíthet megérteni, hogy milyen jellegű problémákon vizsgálható a bemutatott desztillációs megközelítés, a projekt pedig kiindulási pontként szolgálhat hasonló modellprototípusok készítéséhez vagy a végső tanítás előkészítéséhez.

1.2 Vizsgált optimalizációk
A dolgozat fókusza a bemeneti vocabulary redukció, a kimeneti vocabulary redukció és a paraméterezhető desztillációs tanítás együttes vizsgálata. A bemeneti redukció a domainen nem használt embedding paramétereket hagyja el, ami domainfüggően akár 98% feletti input embedding redukciót is jelenthet, míg a kimeneti redukció a feladathoz szükséges tokenekre és átgondolt tippeléssel kiválasztott auxiliary tokenekre szűkíti az output projectiont. A tanítási pipeline a KL divergencia és cross-entropy veszteségek arányát, valamint a distillation temperature-t is egymástól függetlenül konfigurálhatóvá teszi.
A dolgozat hozzájárulása nem egy teljesen új modellarchitektúra vagy egyetlen általánosan optimális tanítási eljárás, hanem ezeknek a részben ismert és kutatott technikáknak és módszereknek a több domainen vizsgált, gyakorlati kombinációja.

2. Irodalomkutatás

2.1 Transformer architektúra
A modern nagy nyelvi modellek alapját a Vaswani et al. (2017) által bevezetett Transformer architektúra adja [1]. A kulcsinnovációk: self-attention mechanizmus (a szekvencia bármely pozíciójából közvetlenül figyel bármely másikra), multi-head attention (párhuzamos attention mechanizmusok különböző aspektusokra), és pozícionális kódolás. Az eredeti architektúra 6 encoder és 6 decoder réteget használt. A dolgozat student modellje decoder-only transformer, hasonlóan a modern LLM-ekhez (GPT, Gemma, Claude).
A Transformer architektúra megértéséhez a 3Blue1Brown YouTube csatorna [24] és a Welch Labs alapozó videói [25] szolgáltak kiindulópontként. Az első saját transformer implementáció ezek alapján készült, majd a működő modell a Gemma architektúrájához lett igazítva a könnyebb desztilláció érdekében.

2.2 Knowledge Distillation
A knowledge distillation fogalmát Hinton et al. (2015) vezette be [2]. A módszer lényege: egy nagyobb "teacher" modell tudását kisebb "student" modellbe tömörítjük. A kulcs a "soft targets" használata, a teacher softmax kimenetét magas temperatur-rel lágyítva kapjuk az eloszlást, amely gazdagabb információt tartalmaz a hard labels-nél. Hinton ezt "dark knowledge"-nek nevezte: a teacher bizonytalanságai és a hibás osztályok közötti preferenciái is átadódnak.
A desztillációs loss általános formája: L = α·L_CE + (1-α)·L_KL, ahol L_CE a cross-entropy a ground truth címkékkel, L_KL pedig a Kullback-Leibler divergencia a teacher és student eloszlások között [3]. A soft targets információgazdagsága miatt kevesebb adat és tanítási idő szükséges.

2.3 LLM-specifikus desztilláció
A DistilBERT [4] 40%-kal kisebb modellt ért el a BERT-hez képest, 97%-os teljesítménymegőrzéssel. A TinyBERT [5] továbbment: transformer rétegek, embedding-ek és predikciós rétegek együttes desztillációjával 7,5x kisebb, 9,4x gyorsabb modellt hozott létre 96,8%-os GLUE teljesítménnyel.
A MiniLLM [6] megállapítása szerint autoregresszív generálásnál a forward KL divergencia helyett reverse KL előnyösebb, mert megakadályozza, hogy a student túlbecsülje a teacher alacsony valószínűségű régióit. A dolgozat forward KL-t használ, de a CE loss dominanciájának növelése a tanítás végén hasonló hatást ér el.

2.4 Curriculum Learning és Loss Annealing
A curriculum learning, ahol a modell először könnyebb, majd fokozatosan nehezebb példákon tanul, bevett módszer [7]. A dolgozat egy speciális curriculum-ot alkalmaz: nem a példák nehézségét, hanem a loss komponensek arányát változtatja konfigurálható kezdő- és végértékkel. Ez az "Annealing Knowledge Distillation" [8] megközelítéshez hasonlít, ahol a temperature-t csökkentik a tanítás során.
A "Curriculum Temperature for Knowledge Distillation" [9] kimutatta, hogy a student modellek lágyabb eloszlásokból profitálnak a tanítás elején, de élesebb eloszlásokra van szükségük később. A dolgozat ezt implementálja: mind a KL/CE arány, mind a desztillációs temperature cosine decay-t követ konfigurálható start→end értékekkel.

2.5 Csökkentett kimeneti szótár
A vocabulary reduction desztillációban kevésbé kutatott terület. A "Knowledge Distillation with Reduction of Vocabulary" [10] 17-49x tömörítést ért el orosz nyelvi modelleken. A "Fast Vocabulary Transfer" [11] a vocabulary cseréjét vizsgálta desztilláció során.
A dolgozat megközelítése eltér: nem a teljes vocabulary-t csökkenti, hanem a kimeneti szótárat korlátozza a feladathoz szükséges tokenekre (strukturált domaineknél ~525, post generation esetén 26 659 token a 262 144-ből), és a bemeneti szótárat is a domainen előforduló tokenekre szűkíti (reddit sentiment: 68 328, math: 1 282, post generation: 72 133 token). Ez a task-specifikus ki- és bemeneti vocabulary kombináció kevésbé kutatott terület a szakirodalomban.

2.6 Architektúrai komponensek
A student modell a következő, szakirodalomban megalapozott komponenseket használja:
- RMSNorm [12]: Zhang és Sennrich (2019). A LayerNorm egyszerűsített változata, amely csak RMS-sel normalizál, mean subtraction nélkül. 7-64%-kal gyorsabb, azonos teljesítmény mellett. A modern LLM-ek (LLaMA, Gemma) ezt használják.
- RoPE [13]: Su et al. (2021). Rotary Position Embedding - a pozíció információt rotációs mátrixokkal kódolja. Paramétermentes, jól skálázódik hosszú kontextusra.
- GeGLU [14]: Shazeer (2020). A feed-forward rétegekben GELU aktivációt gated mechanizmussal kombinál. A Gemma és más modern LLM-ek ezt használják.

2.7 Optimalizáció
AdamW [15]: Loshchilov és Hutter (2017) javított Adam változata, ahol a weight decay különválik a gradiens alapú frissítéstől. A transformer tanítás de facto standard optimizere.
Cosine Annealing [16]: Szintén Loshchilov és Hutter munkája (2016). A learning rate koszinusz görbe mentén csökken, opcionálisan warm restarts-szal. A dolgozat ezt használja a learning rate-re és a KL/CE arány változtatására is.

2.8 Scaling Laws
A Chinchilla scaling law (Hoffmann et al., 2022) [17] megállapította, hogy compute-optimális tanításhoz minden modell paraméterre ~20 token szükséges. A dolgozat 5M-127M paraméteres modelleket vizsgál ~500 ezer példával. A desztilláció más dinamikát követ — a soft labels információgazdagabbak, így potenciálisan kevesebb adat is elegendő lehet, ami a domain-specifikus feladat és a csökkentett vocabulary mellett működhet.

2.9 Toxicitás klasszifikáció
A Reddit és általános toxicitás klasszifikáció aktív kutatási terület. A Jigsaw Toxic Comment Classification Challenge [18] alapvető benchmark. A transformer modellek (BERT, RoBERTa) state-of-the-art eredményeket érnek el [19]. A Detoxify projekt [20] széles körben használt pre-trained modelleket kínál.
A dolgozat megközelítése eltér: nem fine-tuned classifier, hanem generatív, autoregresszív JSON output. Ez komplexebb, de a kódbázis rugalmas - könnyen átírható más domain-specifikus feladatokra (pl. email klasszifikáció, matematikai feladatok) a kimeneti vocabulary és prompt cseréjével.

2.10 Gemma modell
A Gemma 3 modellcsalád (Google DeepMind, 2025) [21] maga is desztillációval készült - egy nagyobb teacher modellből tanult. A 4B variáns 4,3 milliárd paraméterrel rendelkezik, 4 trillió tokenen pretrained. A dolgozat ezt használja teacher modellként.

2.11 Autoregresszív generálás vs klasszifikáció
A dolgozat tudatosan választ autoregresszív generálást a klasszifikáció helyett. Egyszerű sentiment analysis-hez általában encoder modelleket (BERT) vagy classifier head-eket használnak [22]. A T5 modell [23] bizonyította, hogy minden NLP feladat text-to-text formában kezelhető, beleértve a klasszifikációt is. A dolgozat megközelítése ezt követi: a cél nem az optimális klasszifikáció, hanem annak bizonyítása, hogy a desztillációs pipeline működik, olyan feladatokra is, ahol az autóregreszsív transformer modellek előnyösebbek. 

2.12 A dolgozat pozícionálása
A dolgozat több ismert technikát kombinál:
- Standard elemek: KL+CE loss, AdamW, Cosine Annealing, Transformer architektúra, RMSNorm, RoPE, GeGLU
- Kevésbé standard: dinamikus KL/CE arányváltozás curriculum-ként
- Újszerű kombináció: task-specifikus ki- és bemeneti vocabulary redukció (89,8-99,8% output, 72,5-99,5% input) + autoregresszív domain-specifikus output generálás

3. Implementáció
A projekt 3 részre bontható: data extraction pipeline, tanítás és a modell(ek). A data extraction pipeline egy tanármodellt használva állítja elő és menti el a desztillációs tanításhoz szükséges adatokat (ami akár több modell tanításához is újrahasználható). A tanítás és modell jobban összefügg, a tanítás egy speciális curriculum, amely célja először a teacher modell következő token predikció disztribúciójának a student modellhez való közelítése, majd ezt követően a következő helyes token fontosságára nagyobb hangsúlyt fektetve megerősíteni a modellt, hogy ne csak hasonlóan gondolkodjon, mint a teacher modell, de a helyes döntést is hozza meg [2][8]. Több fajta student modellt is lehet használni, a dolgozat viszont elsősorban kisebb multi-headed attentiont használó Transformer modellekre fókuszál [1], amelyek ugyanazzal az autoregresszív viselkedéssel állítják elő a kimeneti választ, mint a manapság leginkább elterjedt nagy nyelvi modellek (pl.: OpenAI GPT, Google Gemini, Anthropic Claude modellek) és a desztillációhoz használt teacher modellek maguk is.

3.1 Vizsgált domainek
A projekt több domainen keresztül vizsgálja ugyanazt az alapötletet: teacher modell által generált autoregresszív, tokenenként mentett logit eloszlások alapján tanítani egy kisebb student modellt. A domainek szándékosan eltérő kimeneti struktúrát használnak, mert más kompromisszumok jelennek meg egy kötött JSON objektumnál, egy determinisztikus számítási scaffoldnál és egy szabadabb természetes nyelvi generálási feladatnál.

3.1.1 Reddit komment sentiment klasszifikáció
A Reddit sentiment domain rövid, 3 és 25 token közötti kommenteket használ bemenetként. A bemeneti adatokból kiszűrésre kerülnek a speciális tagek, URL-ek és zavaró karakterek, hogy a domain zajmentes és könnyebben kontrollálható legyen. A teacher modell kimenete egy strukturált JSON objektum, amely a kommentet négy mező mentén jellemzi: tone, sentiment, safety és toxicity. A generálás a lezáró kapcsos zárójelig tart. Ez a domain elsősorban azt vizsgálja, hogy egy csökkentett vocabulary-jű autoregresszív student modell képes-e stabilan megtanulni a kötött JSON formátumot és a teacher klasszifikációs preferenciáit.

3.1.2 Matematikai szöveges feladatok
A math word problem domain generált matematikai szöveges feladatokat használ. A bemenet templatekből, véletlenszerű nevekből, tárgyakból és számokból épül fel, 13 művelettípussal: alap aritmetikai, összetett és összehasonlító feladatokkal. Minden feladathoz scaffold tartozik, például: "Problem: ...\nA=?\nB=?\nC=A+B=?\nSolution: ?;". A teacher modell a konkrét számítási lépéseket tölti ki, a kimenet pedig pontosvesszővel zárul. Ez a domain a strukturált, nem JSON-alapú kimenetet és a nyelvi-matematikai gondolkodás kapcsolatát vizsgálja.

3.1.3 Reddit posztgenerálás
A post generation domainben az a feladat, hogy a modell egy Reddit kommentből rövid, plauzibilis Reddit posztot generáljon, amelyre a komment válaszként illeszkedhetne. A bemenet itt is rövid Reddit komment, de a kimenet már nem kötött JSON vagy számítási scaffold kitöltése, hanem természetes nyelvi szöveg, amely a "<end>" lezáró markerrel zárul. Emiatt a kimeneti vocabulary jóval nagyobb, és a teacher eloszlásában több hasznos puha információ maradhat. Ez a domain stressztesztként szolgált az infrastruktúrának (óriási desztillációs adatmennyiség), és erős bizonyítékként szolgál a pure KL és KL/CE annealing veszteségszámítás előnyeire. A domain értékelése emiatt később nem csak pontos tokenegyezésre, hanem top-k és teacher-student eloszlás-összehasonlításra is támaszkodik.

1. Táblázat: domainek vocabulary és adatméret jellemzői

| Mutató | Reddit sentiment | Math word problem | Reddit post generation |
|---|---:|---:|---:|
| Bemeneti vocabulary | 68 328 | 1 282 | 72 133 |
| Bemeneti redukció | 73,9% | 99,5% | 72,5% |
| Kimeneti vocabulary | 525 | 528 | 26 659 |
| Kimeneti redukció | 99,8% | 99,8% | 89,8% |
| Teljes paraméterszám (80h/5L modell) | ~6,02M | ~0,66M | ~8,44M |
| Teljes modell redukció (80h/5L) | 85,9% | 98,5% | 80,2% |
| Átlagos generált lépés / példa | 38,9 | 32,0 | 62,9 |
| Logit érték / példa | ~20,4k | ~16,9k | ~1,68M |
| Batch fájlméret (32 példa) | kb. 5,9 MB | kb. 5,5 MB | kb. 512 MB |

A vocabulary redukciók a Gemma 262 144 tokenes vocabulary-jéhez képest értendők. A teljes modell redukció egy azonos transformer body-val rendelkező 80 hidden dimenziós, 5 rétegű modellhez viszonyít, ahol redukció nélkül a bemeneti és kimeneti vocabulary is 262 144 token. Ebben az összehasonlításban a redukció nélküli modell 42,72M paraméteres. A post generation redukált bemeneti vocabulary a Reddit kommentek, a 10 ezer generált mintaposzt, valamint a prompt, lezáró és egyéb technikai tokenek uniója. Az átlagos generált lépésszám az adott domain egy runjának első 5 train batchéből számolt átlag. A "logit érték / példa" az átlagos generált lépésszám és a kimeneti vocabulary méretének szorzata, ezért nem a JSONL fájlméretet, hanem a desztillációs példa információs sűrűségét mutatja. A batch fájlméret sor korábbi mérések alapján a desztillációs JSONL batchek nagyságrendjét mutatja. Ez alapján látszik, hogy a post generation domain nemcsak hosszabb kimeneteket használ, hanem nagyságrendekkel nagyobb teacher eloszlást is tárol minden példához.

3.2 Data extraction pipeline
A projekt során a Google Gemma 3 4b open source modell szolgál teacher modellként [21]. A modell méretéhez képest elfogadható intelligenciával rendelkezik, képes megbízhatóan structured outputot előállítani. Mérete miatt kényelmesen elfér 12GB VRAM-on, és consumer grade videókártyán is használható inferenciára.
A data extraction során minden autoregresszív lépésnél mentésre kerülnek a domainhez definiált kimeneti vocabulary nyers logit értékei. A szükséges tokenek azok, amelyek abszolút szükségesek az összes lehetséges valid kimenet generálásához (reddit sentiment: 27, math: 30 token), míg az auxiliary tokenek (további whitespace, prompt tokenek, gyakori angol szavak, math-nál extra változónevek) nagyobb puha címkét biztosítanak a student modellnek, így jobban el tudja sajátítani a teacher "dark knowledge"-jét [2], illetve hasonlóbb top-k eloszlást tud produkálni nem determinisztikus generáláskor. A jelenlegi konfigurációkban ez reddit sentiment esetén 525, math esetén 528, post generation esetén 26,659 kimeneti tokent jelent, szemben Gemma 262,144 tokenjével. A strukturált domaineknél ez 99,8%-os csökkentés. Ezeknél a domaineknél a vocabulary szempontjából a különbség elhanyagolható, viszont desztillációs szempontból az ~525 tokenes puha címke ~19x annyi információt hordozhat, mint a ~30 tokenes szükséges minimum — ezért a nagyobb vocabulary meghagyható. A példák 32-esével kerülnek mentésre JSONL batchekbe.
Az, hogy Gemma 3 4b bizonyos esetekben hibás vagy vitatható választ ad, nem jelent problémát, hiszen a cél nem annak bizonyítása, hogy a desztillált modell emberi szempontból helyes választ adott-e, hanem hogy mennyire tudja a student modell követni a teacher modell preferenciáját. Ez azt is jelenti, hogy a student modell maximum annyira lehet jó, mint a teacher, hiszen a teacher hibáit is megtanulja.
3.3 A tanítás
A tanítás egyik legfontosabb része a veszteség, amit használva a modell optimalizál. A projekt két veszteség együttesét használja, változó arányban [3]. A cross entropy loss a teacher által generált konkrét következő tokenre optimalizál, míg a Kullback-Leibler divergencia a teacher és student eloszlások közötti különbséget számolja [2][3]. Ezen belül a projekt forward KL divergenciát használ (KL(teacher || student)), amely a student modellt a teacher teljes eloszlásának lefedésére ösztönzi [2][3]. A MiniLLM [6] kimutatta, hogy autoregresszív generálásnál a reverse KL előnyösebb lehet, mert megakadályozza a teacher alacsony valószínűségű régióinak túlbecslését; a projektben ezt a hatást részben a CE loss későbbi erősítése adja vissza.

A KL/CE arány és a distillation temperature együtt curriculum jellegű tanítási ütemezést alkot. A tanítás elején a magasabb KL súly és a magasabb temperature lágyabb teacher eloszlást ad, így a student nem csak a helyes tokent látja, hanem a teacher bizonytalanságát és alternatív preferenciáit is [2][9]. A tanítás végén a temperature csökken, a CE komponens nagyobb szerepet kaphat, és a modell erősebben a helyes következő-token predikció felé élesedik. A KL és CE veszteség aránya, valamint a temperature konfigurálható kezdő- és végértékkel rendelkezik (például KL 0,99→0,5 vagy temperature 5→3), és cosine decay-t követ a tanítás során [16]. Mindkét veszteség végig jelen van a gradiensben, így a modell nem vált hirtelen egyik tanulási célról a másikra [8][9].

A megfelelő schedule nem csak domainfüggő, hanem célfüggő is. Más beállítás kedvez annak, ha a cél egy minél magabiztosabb következő-token prediktor, más annak, ha a cél a teacher teljes eloszlásának követése, és megint más lehet a jó kompromisszum a kettő között. A strukturált JSON és math domaineknél sok következő-token eloszlás éles, ezért a KL, CE és vegyes veszteségek közötti különbség több futás után is kicsi maradhat. Ezt az egységesen 8000 tanító példán történő sentiment és math kísérletek is mutatják: mindkét domainen pure CE, pure KL és KL 0,99→0,5 beállításból 10-10 futás készült, de a különbségek több metrikán is a szórással összemérhetőek maradtak. Free-form post generation esetén több plauzibilis folytatás versenyez, ezért a teacher eloszlásának követése több információt hordoz. A kísérletek alapján az agresszív KL-súly csökkentés (például 0,9→0,1) könnyen gyengítheti a KL korábbi eloszláskövető hatását, míg a magas KL súlyról mérsékeltebb értékre annealelő beállítások (például 0,99→0,5 vagy 0,99→0,6) jobban megőrzik a distribution matching előnyét. Ez nem univerzális szabály, hanem a veszteségek skálájából, a kimeneti vocabulary méretéből, a teacher eloszlás élességéből és a konkrét alkalmazási célból következő gyakorlati tapasztalat.

| Domain / beállítás | Student accuracy | Classification accuracy | Teacher-forced accuracy |
|---|---:|---:|---:|
| Sentiment, 8000 példa, pure CE | 88,60% ± 0,37% | 64,56% ± 2,29% | 97,68% ± 0,34% |
| Sentiment, 8000 példa, pure KL | 88,52% ± 0,36% | 64,12% ± 1,99% | 97,78% ± 0,05% |
| Sentiment, 8000 példa, KL 0,99→0,50 | 88,63% ± 0,46% | 65,03% ± 2,71% | 97,76% ± 0,12% |
| Math, 8000 példa, pure CE | 65,44% ± 1,68% | 21,66% ± 1,70% | 78,08% ± 0,30% |
| Math, 8000 példa, pure KL | 66,43% ± 2,08% | 21,92% ± 1,86% | 78,35% ± 0,42% |
| Math, 8000 példa, KL 0,99→0,50 | 65,34% ± 2,13% | 22,83% ± 0,55% | 78,65% ± 0,58% |

A táblázatban látható, hogy a strukturált kimenetű domaineknél a CE, KL és vegyes beállítások közötti különbség többnyire a futások közötti szórással összemérhető.

A post generation domain ezzel szemben lényegesen nehezebben skálázható. Egy 32 példás post generation batch körülbelül 512 MB, miközben a strukturált sentiment és math batchek mérete jellemzően 5-6 MB. Emiatt a free-form generálás statisztikailag rigorózusabb tesztelését jelenleg főleg a distillation data előállításának és a tanításnak az idő- és tárhelyigénye korlátozza. A korai kísérletek alapján a KL/CE annealing és temperature annealing kombinációja free-form szövegen jelentős javulást mutat, mert egyszerre támogatja a következő-token predikció élesítését és a teacher eloszlásának követését. Ez a "best of both worlds" jellegű eredmény ígéretes, de további futtatások szükségesek ahhoz, hogy statisztikailag is erősen alátámasztható legyen. Ez a munka jelenleg is aktívan folyik.
Nagy nyelvi modelleknél szinte kizárólagosan AdamW-t használnak optimizerként [15], hiszen stabilabb tanulást eredményez (főleg nagyobb modelleknél) és jobban kezeli a sparse adatokat (mint például a nyelv). Ráadásul a modern AI/LLM infrastruktúra többnyire erre az optimizerre van optimalizálva, ez az industry standard.
A learning rate lineáris warmup fázissal indul (a teljes tanítás konfigurálható hányadán), majd cosine decay-t követ [16]. A warmup stabilizálja a korai gradienseket, a cosine decay pedig fokozatosan csökkenti a learning rate-et a fine-tuning fázisban.
A tanítás folyamán folyamatosan megjelenik a konzolon a train loss, a train accuracy és az adott példa vagy batch ideje. Ezeknek az adatoknak a fontosabb része mentésre is kerül.
Minden epoch végén készül checkpoint és tesztelésre kerül a modell a teljes adathalmaz konfigurálható hányadán (alapértelmezetten 2%), ami erre a célra lett félretéve. Emellett mini-eval fut konfigurálható gyakorisággal (alapértelmezetten 1000 batchenként), amely konfigurálható számú teszt batch-en (alapértelmezetten 10) méri a modell aktuális teljesítményét — ez lehetővé teszi a konvergencia epoch közbeni követését.
A program kezeli a graceful shutdownt, tehát epochok között is el tudja menteni a progresst.
3.4 Student modell
A kiinduló pilot modellnél fontos volt, hogy a lehető legnagyobb eséllyel legyen sikeres a desztilláció. A fő kérdések:
- Működik-e a pipeline end-to-end? Elérhető-e közel 100%-os student only accuracy?
- Tud-e koherens JSON-t generálni a saját modell? Egyáltalán a rendelkezésre álló erőforrásokon betanítható-e?
- Működik-e a csökkentett kimeneti szótár?
Ezért a legkedvezőbb körülmények biztosítása volt a cél: nagy puha címke (525 token), elegendő paraméter (~537M), Gemma-szerű architektúra. A kimeneti szótár mérete lineárisan skálázódik, így a nagyobb puha címke meghagyható - segít a modellnek magába szívni a teacher "dark knowledge"-ét [2]. Az attention számítás négyzetesen skálázódik a context length-szel, így a 75 token ideális balance, de nagyobb sem lenne feltétlenül probléma.
A pilot sikere után a projekt több modellméret és konfiguráció szisztematikus tesztelésére bővült, 5M-tól 127M paraméterig.
3.4.1 Architektúra és Gemma hasonlóságok
A student modell architektúrája paraméterezhető: a hidden dimenzió, rétegszám és attention head-ek száma konfigurálható. A transformer body hasznos kapacitását elsősorban az attention és feed-forward rétegek adják, míg a bemeneti embedding és a kimeneti projekció paraméterszáma közvetlenül a vocabulary méretétől függ. A bemeneti embedding paraméterszáma input_vocab × h, a kimeneti projekcióé output_vocab × h + output_vocab, ahol h a hidden dimenzió. A transformer body közelítő paraméterszáma 16 × L × h², ahol L a rétegszám.

Ez különösen kisebb student modelleknél fontos, mert teljes Gemma vocabulary mellett a modell nagy része olyan tokenek reprezentációjára megy el, amelyek a domainben nem fordulnak elő. Az összehasonlításhoz használt 80 hidden dimenziós, 5 rétegű konfiguráció transformer body-ja 512 880 paraméter. Teljes bemeneti és teljes kimeneti vocabulary mellett az erre épülő modell 42,72M paraméteres. Ugyanez a hasznos transformer body redukált vocabulary-vel reddit sentiment esetén 6,02M, math word problem esetén 0,66M, post generation esetén 8,44M paraméteres modellt eredményez. A csökkenés tehát nem abból származik, hogy a modell gyengébb attention vagy feed-forward rétegeket kap, hanem abból, hogy a domainben nem használt vocabulary paraméterek kikerülnek.

A pilot modell ~537M paraméterből állt (1024 dim, 16 réteg, 16 head), ahol az input embedding önmagában ~268M paramétert jelentett (262 144 token × 1024 dim), vagyis a paraméterek ~50%-a. Összehasonlításképp, Gemma 3 4b embedding rétege önmagában ~671M paraméter (262 144 text token + 64 speciális token = 262 208 × 2560 dim), tehát a pilot modell teljes paraméterbüdzséje kisebb volt, mint a teacher modell embedding rétege. Ez rámutatott, hogy kisebb modelleknél a teljes bemeneti szótár aránytalanul nagy költség. Bemeneti oldalon a redukció domainen belül nem vesz el hasznos kapacitást, mert a domainben nem előforduló tokenek embedding sorai nem vesznek részt a számításban. A korlát inference időben jelenik meg: ha a redukált bemeneti vocabulary-vel futó modell ismeretlen token ID-t kap, a jelenlegi implementáció hibát dob. Gyakorlati rendszerben ezt előfeldolgozással, lokális elutasítással vagy nagyobb modellhez irányítással lehet kezelni; ez deployment döntés, nem automatikus döntéstámogató funkció a jelenlegi pipeline-ban. A kimeneti oldalon óvatosabb kompromisszum szükséges, mert a KL-alapú desztillációhoz a teacher eloszlásának információtartalma is fontos; ezért a minimálisan szükséges tokenek mellett auxiliary tokenek is maradnak.
A modell a következő architekturális elemeket veszi át Gemmától:
- RMSNorm [12]: a hagyományos LayerNorm helyett, ahogy Gemma is használja
- RoPE (Rotary Position Embedding) [13]: pozíció kódolásra, Gemma-kompatibilis
- GeGLU-szerű aktiváció [14]: GELU(gate) * linear, hasonló a Gemma megoldásához
- Ugyanaz a tokenizer: a bemeneti tokenek azonos reprezentációt kapnak
Ezek a hasonlóságok biztosítják, hogy a student modell architektúrája közel áll a teacher modelléhez, így a desztilláció hatékonyabb.
3.4.2 Optimalizációk
A modell bemeneti szótára alapértelmezetten megegyezik Gemma 3 4b-vel (262 144 token), de a domain-specifikus bemeneti szótár jelentősen csökkenti ezt: reddit sentiment esetén 68 328 tokenre (73,9%-os csökkentés), matematikához 1 282 tokenre (99,5%-os csökkentés), post generation esetén 72 133 tokenre (72,5%-os csökkentés). A kimeneti szótár szintén domainfüggő: 525 token reddit sentimenthez (27 szükséges + 498 auxiliary), 528 token matematikához (30 szükséges + 498 auxiliary), és 26 659 token post generationhöz. A strukturált domaineknél ez 99,8%-os, post generation esetén 89,8%-os kimeneti vocabulary redukciót jelent.
A tesztelt modellek 5M-tól 127M paraméterig terjednek, szemben a teacher modell 4,3 milliárd paraméterével, tehát 34x-tól 860x-os méretcsökkentés érhető el.

3.4.3 Kimeneti vocabulary és KL kompromisszum
A kimeneti vocabulary redukció közvetlen paramétercsökkentést ad, mert a kimeneti projekció mérete output_vocab × h + output_vocab. Ez első ránézésre egyszerűbbnek tűnik, mint a bemeneti vocabulary redukció: ha a feladat csak néhány tíz vagy száz tokent használ, a teljes 262 144 tokenes kimeneti réteg felesleges. Desztillációs tanításnál azonban a kimeneti szótár nem csökkenthető következmények nélkül a minimálisan valid tokenekre, mert a KL divergencia nem csak a helyes következő tokent, hanem a teacher teljes eloszlását próbálja követni [2][3]. Ha túl kevés token marad a kimeneti eloszlásban, a soft target közelebb kerül egy hard labelhez, és csökken a "dark knowledge" átadásának lehetősége [2], valamint a helyes top-k disztribúció követése is.

Ezért a projekt a szükséges tokenek mellett auxiliary tokeneket is megtart. A strukturált domaineknél ezek főleg whitespace, prompt-hoz kapcsolódó tokenek, gyakori angol szavak és domain-specifikus kiegészítők. Paraméterszámban az 525 vagy 528 tokenes kimeneti réteg továbbra is elhanyagolható a teljes Gemma vocabulary-hez képest, és elméletileg gazdagabb eloszlást ad, mint a 27-30 tokenes minimum. A kísérletek alapján azonban strukturált outputnál ezt a többletet nehéz volt statisztikailag jelentős javulásként kimutatni a pure KL, pure CE és vegyes veszteségfüggvények között. Ezekben a feladatokban sok lépésnél a teacher eloszlása nagyon éles: JSON mezőnevek, zárójelek, írásjelek, számjegyek vagy scaffold elemek következnek, ahol kevés valódi alternatíva versenyez. Ilyen eloszlásnál az auxiliary tokenek és a KL loss többletinformációja kevésbé látszik a determinisztikus tokenválasztást mérő tokenpontosságban.

Post generation esetén más a helyzet: a kimeneti vocabulary 26 659 token, mert természetes nyelvi generálásnál több plauzibilis folytatás versenyez egymással. Itt a nagyobb output vocabulary a feladat természetéből következő kompromisszum: még mindig 89,8%-kal kisebb, mint a teljes Gemma kimeneti tér, de elég nagy ahhoz, hogy a KL loss értelmes eloszlási információt kapjon. Ezért a kimeneti vocabulary méretét és a KL/CE arányt nem érdemes domainfüggetlen konstansként kezelni: a döntés attól is függ, hogy a teacher következő-token eloszlása éles, strukturált vagy több lehetséges folytatást tartalmazó természetes nyelvi eloszlás.

3.4.4 Modell részletek
A projekt során tesztelt modellek hidden dimenziója 20-tól 384-ig terjed, rétegszámuk 2-től 12-ig, head számuk 1-től 6-ig. A skálázási kísérletek 7 modellméretet vizsgáltak teljes bemeneti szótárral, a csökkentett bemeneti szótár kísérletekben további 3 konfiguráció készült azonos paraméterszámú, de mélyebb architektúrával — a felszabaduló embedding paraméterek a transformer body-ra fordíthatók.
4. Módszertan
4.1 Kísérletek
A projekt minden tanítási futtatásról külön alkönyvtárat tart fenn a run könyvtáron belül, amelyben a konfiguráció, a modell metaadatai, a tanítási logok, az epoch végi kiértékelések és a checkpointok elkülönítve tárolódnak. Ez azért fontos, mert a dolgozat nem egyetlen végső modellt vizsgál, hanem több domainen, több modellméreten és több tanítási stratégián keresztül hasonlítja össze a desztillációs döntéseket. A jelenlegi experiment registry több mint 100 futtatást tartalmaz: skálázási, loss- és curriculum-beállításokat és vocabulary redukciós változatokat több domainen. A futtatások fő csoportjai a sentiment domainen végzett skálázási és vocabulary redukciós kísérletek, a sentiment és math domaineken futtatott CE/KL/vegyes loss összehasonlítások, valamint a post generation domainen végzett free-form generálási kísérletek.

A math domainnél a csökkentett 1282 tokenes bemeneti vocabulary paraméterszám szempontjából kiszámolt és támogatott konfiguráció, de a jelenlegi math futtatások teljes Gemma bemeneti vocabulary-t használtak. A post generation domainnél szintén rendelkezésre áll a redukált bemeneti vocabulary számítása, de az eddigi post generation futtatások még teljes bemeneti vocabulary-vel készültek.

4.2 Kutatási eszköztár
A projekt tooling rétege jelenleg nem letisztult, hanem egymás után épült ki, ahogy a kutatási igények változtak. Emiatt több egymást részben lefedő, illetve mostanra részben elavult funkció is található benne. Jelenleg körülbelül 4-6 külön rendszer van legalább részben fenntartva modellfuttatásra, modell- és logelemzésre, valamint kísérleti riportok készítésére. A jövőben ezeknek az eszközöknek az egységesítése fontos fejlesztési irány.

Az eszköztár legfontosabb pillére a log-alapú vizualizáció. A data generation és training folyamat JSONL logokat, állapotfájlokat, run metaadatokat és checkpointokat ment. Ezekből készülnek a training progress ábrák, az epoch végi accuracy/loss görbék, a mini-eval görbék és a confusion matrix jellegű elemzések. Erre épül az experiment analysis réteg is, amely több run összehasonlítását, skálázási ábrákat, vocabulary összehasonlítást, annealing összehasonlítást és HTM oldalon olvasható riportot generál. Az újabb eszközök már nem csak különálló futtatások, hanem egy kutatási kérdéshez tartozó modellcsoportok közös vizsgálatát is támogatják, ami biztosabb alapot ad a következtetések levonásához.

Emellett több célzott elemzőeszköz készült: checkpointonkénti inference-szimuláció, kézi példaelemzés, run-összehasonlítás, skálázási görbe illesztés, valamint a post generation kísérletekhez top-k target accuracy, teacher-student top-k overlap és mean target rank számítás. A data generation és training oldal queue fájlokból indítható, így több futtatás egymás után, felügyelet nélkül végrehajtható. A graceful shutdown, a temp checkpointok és a logger state fájlok lehetővé teszik a hosszabb futtatások megszakítását és folytatását. A jelenlegi állapot a dolgozat kísérleteinek reprodukálását és elemzését támogatja, de tisztább újrafelhasználható eszköztárhoz további egységesítés szükséges.

4.3 Metrikák
A kísérletek értelmezéséhez külön kell választani a tokenpontosságot és a domain-specifikus feladatmetrikákat. A teacher-forced accuracy token szintű metrika: a modell minden lépésben a helyes korábbi tokeneket kapja kontextusként, ezért ez optimista mérés, főleg azt mutatja, hogy lokálisan megtanulta-e a következő tokeneket. A student-only accuracy szintén token szintű, de a modell saját korábbi predikcióit kapja vissza, ezért jobban közelíti az autoregresszív inference közbeni viselkedést.

A "classification accuracy" név a logokban kompatibilitási okból maradt meg, de domainenként eltérő task accuracy-t jelent. Reddit sentiment esetén a generált JSON négy mezőjének (tone, sentiment, safety, toxicity) kategóriahelyességét méri. Math word problem esetén a "Solution:" mezőből kinyert végső megoldás egyezését méri. Post generation esetén nem szemantikai minőséget mér, hanem strukturális completion rate-et: megjelenik-e a "<end>" lezáró marker a generált válaszban. Emiatt a három domain task accuracy értékei nem közvetlenül összehasonlíthatóak egymással.

Free-form generálásnál további metrikák szükségesek, mert a tokenpontos egyezés túl szigorú lehet több plauzibilis folytatás mellett. A top-k target accuracy azt méri, hogy a teacher által generált cél token szerepel-e a student legvalószínűbb tokenjei között. A teacher-student top-k overlap azt mutatja, mennyire hasonló a két modell valószínű tokenhalmaza. A mean target rank a cél token átlagos rangját méri a student eloszlásában, ahol az alacsonyabb érték jobb. Ezek a metrikák teacher-forced kontextusban értelmezendők: nem teljes szabad generálási minőséget mérnek, hanem a következő-token eloszlás hasonlóságát.

4.4 Post generation top-k elemzés
A post generation domainben a szabadabb kimenet miatt a student minőségét nem elég csak pontos tokenegyezéssel mérni. Egy rövid Reddit poszt többféleképpen is lehet elfogadható válasz egy adott komment előzményeként, ezért a teacher eloszlásának követése önmagában is fontos jel. A top-k elemzés három 10 epochos post generation futtatást hasonlít össze ugyanazon test split 24 batchén (768 példa, 47 911 generált tokenlépés).

| Beállítás | Top-20 cél-token találat | Teacher-student top-20 overlap | Átlagos cél-token rang |
|---|---:|---:|---:|
| KL 0,99→0,50 + temperature 5→3 | 69,08% | 34,61% | 486,31 |
| Pure CE, temperature 1 | 67,55% | 26,69% | 1068,92 |
| Pure KL, temperature 5 | 64,42% | 37,31% | 464,70 |

Az eredmény azt mutatja, hogy a pure KL követi legjobban a teacher valószínű tokenhalmazát: ennek a legmagasabb a top-20 overlapje és a legalacsonyabb az átlagos cél-token rangja. A pure CE ezzel szemben jobb cél-token találatot ad, de sokkal gyengébben őrzi meg a teacher eloszlásának szerkezetét. A KL/CE és temperature annealinget kombináló beállítás ebben a mérésben a legmagasabb top-20 cél-token találatot adja, miközben az overlap lényegesen közelebb marad a pure KL-hez, mint a pure CE-hez. Ez támasztja alá a "best of both worlds" értelmezést: a modell magabiztosabb következő-token predikciót kap, de nem veszti el teljesen a KL által tanított eloszláskövetést. Mivel ez egy kisebb, adat- és tárhelyigényes free-form kísérleti sorozat, az eredmény ígéretes, de további futtatásokkal kell statisztikailag erősebben alátámasztani.

4.5 Skálázás és bemeneti vocabulary redukció
A reddit sentiment domainen végzett skálázási futtatások azt mutatják, hogy a strukturált JSON feladat viszonylag korán telítődik. Teljes Gemma bemeneti vocabulary mellett az 5,27M paraméteres modell 92,55% student-only és 86,00% task accuracy-t ér el, míg a 129,19M paraméteres modell 93,25% student-only és 87,42% task accuracy-t. Ez nem azt jelenti, hogy a nagyobb modell haszontalan, hanem azt, hogy ezen a kötött, erősen strukturált feladaton a többletparaméterek hozama gyorsan csökken.

Az alábbi táblázat egy-egy reprezentatív futtatást mutat, nem több seedből számolt átlag.

| Beállítás | Paraméter | Student-only accuracy | Task accuracy |
|---|---:|---:|---:|
| Teljes input vocabulary | 5,27M | 92,55% | 86,00% |
| Redukált input vocabulary | 4,93M | 92,87% | 86,33% |
| Teljes input vocabulary | 10,56M | 92,83% | 86,06% |
| Redukált input vocabulary | 10,13M | 93,22% | 87,20% |
| Teljes input vocabulary | 34,67M | 93,09% | 86,84% |
| Redukált input vocabulary | 36,79M | 93,16% | 87,22% |

A táblázat nem kontrollált architektúra-ablációként értelmezendő, mert a redukált bemeneti vocabulary-vel futtatott modellek más depth/width arányt használnak. A lényeg az, hogy a bemeneti embeddingből felszabadított paraméterkeret hasznos transformer body kapacitásra fordítható, és hasonló teljes paraméterszám mellett legalább versenyképes eredményt ad. Ez a bemeneti vocabulary redukció gyakorlati értékét támasztja alá: nem pusztán paramétert töröl, hanem lehetővé teszi, hogy a kisebb modellben nagyobb arány jusson a tényleges számítást végző rétegekre.

4.6 Pilot run
A teljes modell betanítások előtt saját hardveren pilot runok futottak, hogy kiderüljön, a kód jól működik-e és a modell elkezd-e konvergálni. Egy 30 perces teszt pilot 10 batch adaton már megmutatta, hogy a modell képes helyes JSON outputot generálni. Egy 3 órás pilot run 100 batch adaton, 3,5 epoch után egy 10 batch-es (10*32 példa) teszten a student only accuracy átlagosan 87,69%, míg a teacher forced accuracy átlagosan 98,97% volt. Példa a tesztből, ahol a modell tökéletesen eltalálta a teacher modell klasszifikációját (student only accuracy: 100%):

Bemenet: "F**k you and your family"
Kimenet:
```json
{
  "tone": "aggressive",
  "sentiment": "negative",
  "safety": "harmful",
  "toxicity": "toxic"
}
```

Példa, ahol a modell hibázott (student only accuracy: 69,23%, teacher forced accuracy: 97,44%):

Bemenet: "Bro no way you did this to me again"
Modell kimenet:
```json
{
  "tone": "neutral",
  "sentiment": "neutral",
  "safety": "safe",
  "toxicity": "respectful"
}
```
Teacher kimenet (ground truth):
```json
{
  "tone": "aggressive",
  "sentiment": "negative",
  "safety": "harmful",
  "toxicity": "toxic"
}
```

A második példán látható, hogy a modell hibázott a klasszifikációban, de valid JSON struktúrát generált. Érdemes megjegyezni, hogy a teacher klasszifikációja is vitatható, kontextus nélkül a mondat inkább frusztrált/játékos, mint toxic.
4.7 Korábbi pilot
Egy korábbi, nagyobb léptékű pilot run során 97 órán keresztül tanult a modell. A loss folyamatosan csökkent, a teacher forced accuracy 80-90% körül volt, viszont a student only accuracy csak 10-20% maradt. A nagy különbség a két metrika között gyanús volt. A generált JSON outputok hibásak voltak, annak ellenére, hogy a loss és a teacher forced accuracy jónak tűnt. Később kiderült, hogy egy encoding hiba okozta a problémát: néhány token (pl. „_\"") rosszul volt kódolva a training adatban és inference közben is. A hiba javítása után a 4.6-ban leírt eredmények születtek.
4.8 Pilot teszt elemzés
A pilot teszt összesen ~9 órát vett igénybe: 2,63 óra adatgenerálás (101 batch, 3232 mondat) és 6,25 óra tanítás (347 batch, 3 epoch). A 2. képen láthatók a tanítás metrikái.
A loss az első batcheknél élesen csökken (~25-ről ~2-re), majd tovább csökken, de fluktuálva. A fluktuáció oka, hogy a tanítás előrehaladtával a CE loss egyre nagyobb súlyt kap, ami élesebb, kevésbé sima gradienst eredményez. Az accuracy folyamatosan emelkedik, 35%-ról 95% fölé (ez tanítás közbeni, teacher forced accuracy).
Az aggregált train loss magasabb, mint az eval loss (epoch summary), tehát overfitting nem figyelhető meg, annak ellenére, hogy viszonylag kevés adaton tanult a modell.
A teacher-forced és student-only accuracy párhuzamosan emelkedik. A teacher-forced accuracy magasabb (~97%), mert a modell ilyenkor mindig a helyes kontextust kapja. A student-only accuracy alacsonyabb (~88%), mert bizonyos tokeneket (elsősorban kategória értékeket: "aggressive", "neutral" stb.) nehezebben talál el a modell. Ráadásul az autoregresszív generálás miatt, ha egy korai token hibás, az a teljes kimenet hibás predikciót okozhat.
Mindezek ellenére a student-only módban is az esetek túlnyomó többségében koherens, valid JSON generálódik.

2. kép

4.9 Kiértékelési példák
A 3. képen látható három kiértékelési példa. Minden példánál három szakasz jelenik meg: student (a modell saját predikciói alapján generál), teacher-forced (minden lépésnél a helyes eddigi tokeneket kapja kontextusként), és ground truth (a teacher modell eredeti kimenete).
A színkódolás: zöld a helyes, piros a hibás token.

Example 6: mindkét accuracy 100%, a teljes kimenet zöld. A modell tökéletesen követi a teacher predikciót.

Example 9: student only 71,79%, teacher-forced 92,31%. A JSON struktúra (kulcsok, zárójelek, kettőspontok) helyes, csak a kategória értékek hibásak (piros). A modell megtanulta a JSON formátumot, de a klasszifikációs döntésekben bizonytalan.

Example 19: mindkét accuracy 97,44%, egyetlen token hibás. A "sentiment" mezőnél "neutral" helyett "negative"-ot prediktál. Ez jellemző hiba: a modell a JSON szerkezetét stabilan generálja, a bizonytalanság a kategória értékeknél jelentkezik.

3. kép

4.10 Erőforrás becslés
A pipeline legnagyobb erőforrásigénye az adatgenerálás: a teacher modell (Gemma 3 4b) egy A100 GPU 20GB-s szeletén példánként ~3 másodperc alatt generál egy példát, tehát ~500 ezer példa előállítása ~417 óra. Ez a bottleneck nem változott a projekt során, mivel a teacher modell sebessége fix.
A tanítás erőforrásigénye viszont jelentősen csökkent a pilot óta. A korai becslés 537M paraméteres modellre 15 epochon 2600 óra GPU időt jósolt. A végső kísérletek 5-10M paraméteres modelleket használnak, és 3 epoch elegendő — ennél több túltanuláshoz vezet. A korai implementáció minden egyes tokenre külön forward passt futtatott (40-50 pass/példa), de mivel a causal mask biztosítja, hogy a későbbi tokenek nem befolyásolják a korábbiakat, a teljes szekvencia egyetlen forward passból kiértékelhető. Ez 40-50x gyorsulást eredményezett a loss számításban. 500 ezer példával, 32-es batch mérettel, saját hardveren (RTX 4070) ~0,5 mp/batch sebességgel egy modell 3 epoch tanítása ~6,5 óra. A teljes pipeline bottleneckje tehát egyértelműen az adatgenerálás, nem a tanítás.

5. Források

[1] Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS 2017. https://arxiv.org/abs/1706.03762
A Transformer architektúra alapcikke, amelyre az egész modern LLM ökoszisztéma épül.

[2] Hinton, G., Vinyals, O., Dean, J. (2015). "Distilling the Knowledge in a Neural Network." https://arxiv.org/abs/1503.02531
A knowledge distillation alapműve, a soft targets és dark knowledge koncepciók forrása.

[3] Hugging Face Blog. "Everything You Need to Know about Knowledge Distillation." https://huggingface.co/blog/Kseniase/kd
A desztillációs loss kombinációk gyakorlati áttekintése.

[4] Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." https://arxiv.org/abs/1910.01108
Az első sikeres nagyléptékű transformer desztilláció.

[5] Jiao, X., et al. (2019). "TinyBERT: Distilling BERT for Natural Language Understanding." https://arxiv.org/abs/1909.10351
Többszintű desztilláció (embedding, attention, prediction), referencia a 7,5x tömörítéshez.

[6] Gu, Y., et al. (2023). "MiniLLM: Knowledge Distillation of Large Language Models." https://arxiv.org/abs/2306.08543
LLM-specifikus desztilláció, reverse KL divergencia autoregresszív modellekhez.

[7] Bengio, Y., et al. (2009). "Curriculum Learning." ICML 2009. https://ronan.collobert.com/pub/2009_curriculum_icml.pdf
A curriculum learning alapelve - fokozatosan növekvő nehézség.

[8] Jafari, A., et al. (2021). "Annealing Knowledge Distillation." EACL 2021. https://aclanthology.org/2021.eacl-main.212/
Temperature annealing τ_max → 1, hasonló a dolgozat KL/CE arányváltozásához.

[9] Li, Z., et al. (2023). "Curriculum Temperature for Knowledge Distillation." AAAI 2023. https://ojs.aaai.org/index.php/AAAI/article/view/25236
A lágy → éles eloszlás tanítási stratégia elméleti megalapozása.

[10] Kolesnikova, A., et al. (2022). "Knowledge Distillation of Russian Language Models with Reduction of Vocabulary." https://arxiv.org/abs/2205.02340
A legközelebbi munka a vocabulary reduction desztillációhoz.

[11] Samenko, I., et al. (2024). "Fast Vocabulary Transfer for Language Model Compression." https://arxiv.org/abs/2402.09977
Vocabulary csere/transzfer a desztilláció során.

[12] Zhang, B., Sennrich, R. (2019). "Root Mean Square Layer Normalization." NeurIPS 2019. https://arxiv.org/abs/1910.07467
RMSNorm, a modern transformer architektúrák normalizációs rétege.

[13] Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." https://arxiv.org/abs/2104.09864
RoPE, a Gemma és más modern LLM-ek pozíció kódolása.

[14] Shazeer, N. (2020). "GLU Variants Improve Transformer." https://arxiv.org/abs/2002.05202
GeGLU/SwiGLU aktiváció, a Gemma feed-forward rétegeinek alapja.

[15] Loshchilov, I., Hutter, F. (2017). "Decoupled Weight Decay Regularization." https://arxiv.org/abs/1711.05101
AdamW optimizer, az LLM tanítás standard optimizere.

[16] Loshchilov, I., Hutter, F. (2016). "SGDR: Stochastic Gradient Descent with Warm Restarts." https://arxiv.org/abs/1608.03983
Cosine Annealing learning rate scheduler.

[17] Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." NeurIPS 2022. https://arxiv.org/abs/2203.15556
Chinchilla scaling - 20 token/paraméter szabály.

[18] Kaggle. "Jigsaw Toxic Comment Classification Challenge." https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
Alapvető toxicitás klasszifikáció benchmark.

[19] ResearchGate. "A Comprehensive Survey and Comparative Analysis of Toxic Comment Classification Techniques." https://www.researchgate.net/publication/391347534
Transformer modellek toxicitás detekciós teljesítményének áttekintése.

[20] Hanu, L. "Detoxify - Toxic Comment Classification." https://huggingface.co/unitary/toxic-bert
Gyakorlati implementáció, pre-trained toxicitás modellek.

[21] Google DeepMind. (2025). "Gemma 3 Technical Report." https://arxiv.org/abs/2503.19786
A teacher modell hivatalos dokumentációja.

[22] HuggingFace Transformers Documentation. "Summary of the Models." https://huggingface.co/transformers/v3.1.0/model_summary.html
Autoencoding vs autoregressive modellek klasszifikációs felhasználása.

[23] Raffel, C., et al. (2019). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." (T5) https://arxiv.org/abs/1910.10683
Text-to-text megközelítés klasszifikációra.

[24] 3Blue1Brown. "Neural Networks" és "Transformers" YouTube sorozat. https://www.youtube.com/c/3blue1brown
Vizuális magyarázatok a neurális hálózatok és transformer architektúra működéséről.

[25] Welch Labs. "Neural Networks Demystified" YouTube sorozat. https://www.youtube.com/c/WelchLabsVideo
Neurális hálózatok alapjai, a saját implementáció kiindulópontja.
