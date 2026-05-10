#set document(
  title: "[KNN] Projekt - extrakce metadat vědeckých článků",
  author: ("David Machů (xmachu05)", "Matěj Neděla (xnedel11)"),
  date: none,
)

#show raw: set block(width: 100%, fill: black.lighten(95%), inset: 5pt)
#show raw: set text(size: 8pt)

#title()

#align(right)[_David Machů (xmachu05), Matěj Neděla (xnedel11)_]

= Řešený problém

Projekt se zaměřuje na extrakci metadat z titulních stran vědeckých článků. Cílem je s pomocí existujících řešení poloautomaticky připravit datovou sadu obsahující název, autory, abstrakt, klíčová slova a instituce a následně natrénovat destilovanou neuronovou síť, která bude schopna tyto informace extrahovat z obrazové podoby titulní strany článku.

= Existující řešení

=== GROBID _(GeneRation Of BIbliographic Data)_

- aktivní projekt
- *výstup:* název, autoři (včetně e-mailů a institucí), abstrakt, klíčová slova, datum vydání, ...
- *výhody:* rychlé zpracování
- *nevýhody:* vyžaduje nativní PDF
- *GitHub repozitář:* `https://github.com/grobidOrg/grobid`

=== CERMINE _(Content ExtRactor and MINEr)_

- neaktivní projekt (poslední úpravy v roce 2018)
- *výstup:* název, autoři (včetně e-mailů a institucí), abstrakt, klíčová slova, datum vydání, ...
- *nevýhody:* vyžaduje nativní PDF, pomalé zpracování
- *GitHub repozitář:* `https://github.com/CeON/CERMINE`

=== Gemini, ChatGPT, ...

- *výstup:* flexibilní
- *výhody:* podpora obrazového PDF, rychlé zpracování

= Vstupní data

Titulní strana vědeckého článku v obrazové podobě, typicky ve formátu PNG.

= Výstupní data

Strukturovaná data ve formátu JSON, obsahující následující položky:

- název (`title`)
- autoři (`authors`) --- jméno (`firstName`), příjmení (`lastName`), e-mail (`email`), instituce (`institution`)
- abstrakt (`abstract`)
- klíčová slova (`keywords`)
- datum vydání (`date`)

= Testovací datová sada

=== Kaggle arXiv Dataset

Datová sada obsahující metadata všech vědeckých článků publikovaných na arXiv.org. Samotná PDF nejsou součástí datové sady, ale jsou jednoduše dostupná skrz URL `https://arxiv.org/pdf/{id}`.

- *počet záznamů:* ~2.9 milionů
- *velikost metadat:* ~5.1 GB
- *odkaz:* `https://www.kaggle.com/datasets/Cornell-University/arxiv`

```json
{
  "id": "0704.0002",
  "submitter": "Louis Theran",
  "authors": "Ileana Streinu and Louis Theran",
  "title": "Sparsity-certifying Graph Decompositions",
  "comments": "To appear in Graphs and Combinatorics",
  "journal-ref": null,
  "doi": null,
  "report-no": null,
  "categories": "math.CO cs.CG",
  "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
  "abstract": "We describe a new algorithm, the $(k,\\ell)$-pebble game with colors, ...",
  "versions": [
    { "version": "v1", "created": "Sat, 31 Mar 2007 02:26:18 GMT" },
    { "version": "v2", "created": "Sat, 13 Dec 2008 17:26:00 GMT" }
  ],
  "update_date": "2008-12-13",
  "authors_parsed": [
    ["Streinu", "Ileana", ""],
    ["Theran", "Louis", ""]
  ]
}
```

= Získání datové sady

Pro získání dat byl implementován samostatný skript v jazyce Python. Nástroj umožňuje parametrizaci rozsahu stahování (počet dokumentů a počáteční index), což zajišťuje rozšiřitelnost sady v budoucí fázi projektu. Vědecké články jsou čerpány ze serveru arXiv.org, přičemž proces stahování zahrnuje kromě uložení původního PDF souboru i extrakci titulní strany do formátu PNG a následnou serializaci metadat do formátu JSON. Implementace respektuje limity serveru arXiv.org, tedy omezení na 1 požadavek za 3 sekundy. Toto omezení představuje hlavní úzké hrdlo procesu a limituje celkovou rychlost stahování. Ukázka jednoho záznamu metadat je uvedena níže.

```json
{
  "idx": 1,
  "id": "0704.0002",
  "title": "Sparsity-certifying Graph Decompositions",
  "authors": [
    { "firstName": "Ileana", "lastName": "Streinu", "email": null, "institution": null },
    { "firstName": "Louis", "lastName": "Theran", "email": null, "institution": null }
  ],
  "abstract": "We describe a new algorithm, the $(k,\\ell)$-pebble game with colors, ...",
  "keywords": null,
  "date": "2008-12-13"
}
```

Dále by bylo vhodné doplnit chybějící parametry (e-mail a instituce autorů, klíčová slova) pomocí existujících řešení (např. GROBID) a dostupných nativních PDF dokumentů.

= Předzpracování datové sady

Před samotným trénováním byla datová sada převedena do konverzačního formátu očekávaného frameworkem `Qwen-VL-Series-Finetune`#footnote[`https://github.com/2U1/Qwen-VL-Series-Finetune`] <fn>. K tomu slouží skript `qwen-prepare-dataset.py`, jehož vstupem je soubor `metadata.json` s referenčními metadaty vědeckých článků a adresář `jpg` obsahující obrázky titulních stran. Každý článek je identifikován pomocí svého identifikátoru, podle kterého se dohledává odpovídající obrázek ve tvaru `<id>.jpg`.

Skript postupně načte všechny záznamy z metadat, ověří existenci příslušného obrázku a pro každý platný záznam vytvoří jednu trénovací položku. Tato položka obsahuje identifikátor článku, název obrázku a konverzaci složenou ze dvou zpráv. První zpráva reprezentuje dotaz uživatele a obsahuje značku `<image>` spolu s instrukcí pro model. Druhá zpráva reprezentuje očekávanou odpověď modelu, tedy referenční metadata serializovaná jako JSON řetězec.

Struktura jedné položky odpovídající požadavkům frameworku má následující podobu:

```json
{
  "id": "0704.0002",
  "image": "0704.0002.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nExtract metadata from this title page of a scientific ..."
    },
    {
      "from": "gpt",
      "value": "{\"title\": \"Sparsity-certifying Graph Decompositions\", ...}"
    }
  ]
}
```

Požadovaná odpověď modelu je v promptu definována následujícím JSON schématem:

```json
{
  "title": "",
  "authors": [
    {
      "firstName": "",
      "lastName": "",
      "email": null,
      "institution": []
    }
  ],
  "abstract": "",
  "keywords": [],
  "date": null
}
```

Model má z titulní strany extrahovat název článku, autory, abstrakt, klíčová slova a datum. Prompt zároveň omezuje výstup tak, aby model používal pouze informace viditelné na stránce, nevymýšlel chybějící hodnoty, chybějící e-mail nebo datum vracel jako `null`, chybějící instituce a klíčová slova jako prázdné pole a odpověď vracel pouze jako validní JSON.

Po vytvoření všech platných položek je datová sada zamíchána pomocí pevně zvoleného seedu a rozdělena na trénovací, validační a testovací část. Výchozí poměr je 80 % pro trénování, 10 % pro validaci a 10 % pro testování. Výsledkem předzpracování jsou soubory `train.json`, `eval.json` a `test.json`, které jsou následně použity při trénování modelu.

= Trénování modelů

Po převodu datové sady do konverzačního formátu bylo provedeno doladění vybraných modelů z rodiny Qwen-VL. Trénování bylo realizováno pomocí frameworku `Qwen-VL-Series-Finetune` a spouštěno jako dávková úloha v prostředí PBS pomocí skriptu `qwen-train.sh`. Pro každý experiment byly na lokální scratch prostor výpočetního uzlu zkopírovány soubory `train.json`, `eval.json` a archiv `jpg.tar` s obrázky titulních stran. Archiv s obrázky byl následně rozbalen a cesta k němu byla předána trénovacímu skriptu frameworku.

Trénování probíhalo na souboru `train.json`, který obsahoval přibližně 40 tisíc článků. Validační soubor `eval.json` měl původně přibližně 5 tisíc položek, ale pro rychlejší vyhodnocování byla použita jeho zmenšená varianta s 1000 náhodně vybranými články. Stejný postup byl použit také pro `train_bbox.json` a zmenšenou validační množinu odvozenou z `eval_bbox.json` u varianty s bouding boxy.

Doladění bylo provedeno metodou LoRA, tedy bez plného přetrénování všech parametrů modelu. V experimentech byla jazyková část modelu zmrazena (`freeze_llm=True`), zatímco vizuální část a slučovací projekce zůstaly trénovatelné. Pro LoRA byla použita hodnost 32, parametr `lora_alpha` nastavený na 64 a dropout 0,05. Trénování probíhalo po dobu jedné epochy s velikostí dávky 1 a akumulací gradientů přes 4 kroky. Optimalizace používala kosinový plán učící rychlosti s warm-up poměrem 0,03.

Pro porovnání byly trénovány čtyři varianty modelů:

- `Qwen2-VL-2B-Instruct`,
- `Qwen2-VL-7B-Instruct`,
- `Qwen2.5-VL-3B-Instruct`,
- `Qwen2.5-VL-7B-Instruct`.

Validace byla prováděna každých 1000 kroků a jako hlavní kritérium pro výběr nejlepšího checkpointu byla použita validační ztráta. Trénovací ztráta byla zaznamenávána průběžně a pro lepší čitelnost je v grafech zobrazena jako klouzavý průměr.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 12pt,

    image("assets/Qwen2-VL-2B_loss.png", width: 100%),
    image("assets/Qwen2-VL-7B_loss.png", width: 100%),
    image("assets/Qwen2.5-VL-3B_loss.png", width: 100%),
    image("assets/Qwen2.5-VL-7B_loss.png", width: 100%),
  ),
  caption: [Průběh trénovací a validační ztráty jednotlivých modelů.]
)

Z grafů je patrné, že u všech modelů dochází na začátku trénování k rychlému poklesu trénovací ztráty a následně k jejímu pozvolnému ustalování. Validační ztráta se v průběhu trénování také postupně snižuje, což naznačuje, že se modely učí požadovanému formátu odpovědi a extrakci metadat bez výrazného přeučení na trénovací data.

Nejnižší validační ztráty dosáhly větší modely. Model `Qwen2.5-VL-7B-Instruct` dosáhl nejlepší validační ztráty 0,1125 a model `Qwen2-VL-7B-Instruct` hodnoty 0,1144. Menší modely dosáhly vyšších hodnot, konkrétně `Qwen2.5-VL-3B-Instruct` 0,1234 a `Qwen2-VL-2B-Instruct` 0,1326.

= Obohacení datové sady o bounding boxy

Po základním trénování byla datová sada rozšířena také o anotace pozic extrahovaných hodnot na stránce. Cílem této úpravy bylo, aby model nevracel pouze textová metadata, ale pro vybrané položky také jejich bounding box v pixelových souřadnicích obrázku. Pro tuto variantu byl použit upravený předzpracovací skript `qwen-prepare-dataset-bbox.py`, který kromě souboru `metadata.json` a obrázků z adresáře `jpg` využívá také OCR výstupy ve formátu ALTO XML uložené v adresáři `alto`.

Oproti původní datové sadě se tedy změnila především očekávaná odpověď modelu. Textové hodnoty jsou nově doplněny o atribut `bbox`, který má tvar `[x1, y1, x2, y2]`. Pokud hodnota na stránce chybí nebo její pozici nelze spolehlivě určit, je bbox nastaven na `null`. Struktura požadovaného výstupu je následující:

```json
{
  "title": {
    "text": "",
    "bbox": null
  },
  "authors": [
    {
      "firstName": "",
      "lastName": "",
      "email": null,
      "institution": [],
      "bbox": null
    }
  ],
  "abstract": {
    "text": "",
    "bbox": null
  },
  "keywords": [
    {
      "text": "",
      "bbox": null
    }
  ],
  "date": {
    "text": null,
    "bbox": null
  }
}
```

Bounding boxy byly získávány z OCR výstupů ALTO, které obsahují jednotlivá rozpoznaná slova společně s jejich souřadnicemi na stránce. Pro obecné textové položky se metadata zarovnávají s OCR textem a výsledný bbox vzniká sjednocením bboxů odpovídajících slov. U abstraktu byla kvůli rychlosti použita zjednodušená metoda, která hledá začátek a konec abstraktu a z jejich pozic vytvoří společný bbox. Autoři jsou detekováni v oblasti mezi titulkem a abstraktem pomocí řádkového zpracování a porovnávání příjmení s OCR slovy.

Výsledkem tohoto předzpracování jsou soubory `train_bbox.json`, `eval_bbox.json` a `test_bbox.json`. Soubor `train_bbox.json` obsahoval přibližně 40 tisíc článků a z validačního souboru `eval_bbox.json` byla pro rychlejší vyhodnocování použita zmenšená varianta s 1000 náhodně vybranými položkami. Samotné trénování poté probíhalo stejným způsobem jako u varianty bez bounding boxů. Použit byl stejný framework, stejná metoda LoRA i stejné hlavní hyperparametry. Rozdíl byl pouze ve vstupních souborech, kdy trénovací skript `qwen-train.sh` používal `train_bbox.json` a `eval_bbox.json`, a ve výstupním adresáři, jehož název obsahoval příponu -bbox.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 12pt,

    image("assets/Qwen2-VL-2B-bbox_loss.png", width: 100%),
    image("assets/Qwen2-VL-7B-bbox_loss.png", width: 100%),
    image("assets/Qwen2.5-VL-3B-bbox_loss.png", width: 100%),
    image("assets/Qwen2.5-VL-7B-bbox_loss.png", width: 100%),
  ),
  caption: [Průběh trénovací a validační ztráty jednotlivých modelů při trénování s bounding boxy.]
)

Z průběhu ztráty je patrné, že i po rozšíření cílového výstupu o bounding boxy probíhalo trénování stabilně u všech modelů. Trénovací i validační ztráta se postupně snižovala, přičemž nejlepší validační ztráty dosáhl model `Qwen2.5-VL-7B-Instruct` s hodnotou 0,1648. Dále následovaly modely `Qwen2.5-VL-3B-Instruct` s hodnotou 0,1789, `Qwen2-VL-7B-Instruct` s hodnotou 0,1919 a `Qwen2-VL-2B-Instruct` s hodnotou 0,2180.

Pro přehlednější porovnání jsou na následujícím obrázku zobrazeny validační ztráty všech modelů pro obě varianty úlohy. Graf vlevo odpovídá trénování bez bounding boxů, graf vpravo variantě s bounding boxy.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 12pt,

    image("assets/summary_eval_loss_non_bbox.png", width: 100%),
    image("assets/summary_eval_loss_bbox.png", width: 100%),
  ),
  caption: [Porovnání validační ztráty jednotlivých modelů pro variantu bez bounding boxů a s bounding boxy.]
)

Z porovnání vyplývá, že přidání bounding boxů vede u všech modelů k vyšší validační ztrátě. Úloha je tedy náročnější, protože model musí kromě textových hodnot generovat také číselné souřadnice a dodržet složitější strukturu JSONu. Zároveň je patrné, že nejlepší výsledky dosahují novější nebo větší varianty modelů. U varianty s bounding boxy si modely řady `Qwen2.5-VL` vedly lépe než odpovídající modely řady `Qwen2-VL`, například `Qwen2.5-VL-3B-Instruct` dosáhl nižší validační ztráty než větší `Qwen2-VL-7B-Instruct`. Celkově nejlepší výsledek v obou variantách dosáhl model `Qwen2.5-VL-7B-Instruct`.

= Experimenty a vyhodnocení

První experiment byl zaměřen na porovnání přesnosti základního modelu (konkrétně `Qwen2.5-VL-3B-Instruct`) a jeho _fine-tuned_ varianty z předešlého kroku. Testovacím prostředím bylo Metacentrum poskytované organizací CESNET a výpočetní cluster s GPU o minimální kapacitě 16 GB VRAM. Datová sada pro testování obsahovala 200 náhodně vybraných záznamů, které nebyly použity během tréninku. Výsledky prvního experimentu jsou shrnuty v následující tabulce. Hodnoty jednotlivých metrik představují průměr ze všech záznamů, které daný model úspěšně zpracoval, tedy pro které vygeneroval validní JSON.

#table(
  columns: (1.2fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  [],
    [*Time* \ _average_],
    [*Valid JSON* \ _count (ratio)_],
    [*Title* \ _Levenshtein d._],
    [*Authors* \ _F1 score_],
    [*Abstract* \ _Rouge-L score_],
  [*Base model*],
    [1.565 s],
    [164 (82 %)],
    [0.985],
    [0.205],
    [0.899],
  [*Tuned model*],
    [2.381 s],
    [192 (96 %)],
    [0.990],
    [0.862],
    [0.929],
)

Podobnost názvu je měřena pomocí Levenshteinovy vzdálenosti, která kvantifikuje počet editací potřebných k transformaci jednoho řetězce na druhý, resp. pomocí její normalizované varianty. Pro vyhodnocení autorů je použito F1 skóre, které počítá harmonický průměr přesnosti (_precision_) a úplnosti (_recall_). Abstrakt je hodnocen pomocí Rouge-L skóre, které měří nejdelší společnou podposloupnost (LCS) mezi predikovaným a referenčním abstraktem.

Oproti základnímu modelu má náš _fine-tuned_ model vyšší úspěšnost v generování validních JSON struktur o 14 procentních bodů. Rozdíly v přesnosti extrakce názvu jsou zanedbatelné. U extrakce autorů je významný rozdíl v úspěšnosti _fine-tuned_ modelu, což je ale zapříčiněno především tím, že základní model často generoval autory ve špatném formátu.

= GitHub repozitář

`https://github.com/nedelamatej/knn-mis`
