#set document(
  title: "[KNN] Projekt - extrakce metadat vědeckých článků",
  author: ("David Machů (xmachu05)", "Matěj Neděla (xnedel11)"),
  date: none,
)

#set text(lang: "cs")

#import "@preview/oxifmt:0.2.1": strfmt

#show raw: set block(width: 100%, fill: black.lighten(95%), inset: 5pt)
#show raw: set text(size: 8pt)

#title()

#align(right)[_David Machů (xmachu05), Matěj Neděla (xnedel11)_]

= Řešený problém

Projekt se zaměřuje na extrakci metadat z titulních stran vědeckých článků. Cílem je s pomocí existujících řešení poloautomaticky připravit datovou sadu obsahující název, autory, abstrakt, klíčová slova a instituce a následně natrénovat neuronovou síť, která bude schopna tyto informace extrahovat z obrazové podoby titulní strany článku.

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

Titulní strana vědeckého článku v obrazové podobě, typicky ve formátu JPEG.

= Výstupní data

Strukturovaná data ve formátu JSON, obsahující následující položky:

- název (`title`)
- autoři (`authors`) --- jméno (`firstName`), příjmení (`lastName`), e-mail (`email`), instituce (`institution`)
- abstrakt (`abstract`)
- klíčová slova (`keywords`)
- datum vydání (`date`)

= Datová sada

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

Pro získání dat byl implementován samostatný skript v jazyce Python. Nástroj umožňuje parametrizaci rozsahu stahování (počet dokumentů a počáteční index), což zajišťuje rozšiřitelnost sady v budoucí fázi projektu. Vědecké články jsou čerpány ze serveru arXiv.org, přičemž proces stahování zahrnuje kromě uložení původního PDF souboru i extrakci titulní strany do formátu JPEG a následnou serializaci metadat do formátu JSON. Implementace respektuje limity serveru arXiv.org, tedy omezení na 1 požadavek za 3 sekundy. Toto omezení představuje hlavní úzké hrdlo procesu a limituje celkovou rychlost stahování. Ukázka jednoho záznamu metadat je uvedena níže:

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

= Rozšíření datové sady o chybějící položky

Datová sada, resp. metadata, jsou následně rozšířena o chybějící položky, kterými jsou e-maily autorů, instituce autorů a klíčová slova. Jelikož tyto údaje standardně nejsou součástí metadat uveřejňovaných na serveru arXiv.org, je nutné je získat jiným způsobem. Rozšíření datové sady probíhá pomocí již existujícího řešení GROBID, které je schopné tyto informace extrahovat z nám dostupných nativních PDF dokumentů. GROBID je do výše uvedeného skriptu integrován skrze Docker kontejner, přičemž komunikace se systémem GROBID probíhá prostřednictvím HTTP API. Výsledný skript je dostupný v souboru `arxiv-prepare-dataset.py`. Níže je zobrazena ukázka jednoho záznamu již obohaceného o chybějící položky:

```json
{
  "idx": 1,
  "id": "0704.0002",
  "title": "Sparsity-certifying Graph Decompositions",
  "authors": [
    { "firstName": "Ileana", "lastName": "Streinu", "email": "streinu@cs.smith.edu",
      "institution": ["Department of Computer Science, Smith College"] },
    { "firstName": "Louis", "lastName": "Theran", "email": "theran@cs.umass.edu",
      "institution": ["Department of Computer Science, University of Massachusetts ..."] }
  ],
  "abstract": "We describe a new algorithm, the $(k,\\ell)$-pebble game with colors, ...",
  "keywords": [],
  "date": "2008-12-13"
}
```

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

#pagebreak()

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

Model má z titulní strany extrahovat název článku, autory, abstrakt, klíčová slova a datum. Prompt zároveň omezuje výstup tak, aby model používal pouze informace viditelné na stránce, nevymýšlel chybějící hodnoty, chybějící e-mail a zároveň chybějící datum vracel jako `null`, chybějící instituce a klíčová slova jako prázdné pole a také vracel pouze odpověď ve formátu validního JSONu.

Po vytvoření všech platných položek je datová sada zamíchána pomocí pevně zvoleného seedu a rozdělena na trénovací, validační a testovací část. Výchozí poměr je 80 % pro trénování, 10 % pro validaci a 10 % pro testování. Výsledkem předzpracování jsou soubory `train.json`, `eval.json` a `test.json`, které jsou následně použity při trénování modelu. Jejich ukázky jsou dostupné v adresáři `data`.

= Trénování modelů

Po převodu datové sady do konverzačního formátu bylo provedeno doladění vybraných modelů z rodiny Qwen-VL. Trénování bylo realizováno pomocí frameworku `Qwen-VL-Series-Finetune` a spouštěno jako dávková úloha v prostředí PBS pomocí skriptu `qwen-train.sh`. Pro každý experiment byly na lokální scratch prostor výpočetního uzlu zkopírovány soubory `train.json`, `eval.json` a archiv `jpg.tar` s obrázky titulních stran. Archiv s obrázky byl následně rozbalen a cesta k němu byla předána trénovacímu skriptu frameworku.

Trénování probíhalo na souboru `train.json`, který obsahoval přibližně 40 tisíc článků. Validační soubor `eval.json` měl původně přibližně 5 tisíc položek, ale pro rychlejší vyhodnocování byla použita jeho zmenšená varianta s 1000 náhodně vybranými články. Stejný postup byl použit také pro `train_bbox.json` a zmenšenou validační množinu odvozenou z `eval_bbox.json` u varianty s bouding boxy.

Doladění bylo provedeno metodou LoRA, tedy bez plného přetrénování všech parametrů modelu. V experimentech byla jazyková část modelu zmrazena (`freeze_llm=True`), zatímco vizuální část a slučovací projekce zůstaly trénovatelné (`freeze_vision_tower=False`, `freeze_merger=False`). Pro LoRA byly použity parametry `lora_rank=32`, `lora_alpha=64` a `lora_dropout=0.05`. Trénování probíhalo po dobu jedné epochy s velikostí dávky 1 a akumulací gradientů přes 4 kroky. Optimalizace používala kosinový plán učící rychlosti (`lr_scheduler_type=cosine`) s warm-up poměrem 0,03.

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

= Rozšíření datové sady o bounding boxy

Po základním trénování byla datová sada rozšířena také o anotace pozic extrahovaných hodnot na stránce. Cílem této úpravy bylo, aby model nevracel pouze textová metadata, ale pro vybrané položky také jejich bounding box v pixelových souřadnicích obrázku. Pro tuto variantu byl použit upravený předzpracovací skript `qwen-prepare-dataset-bbox.py`, který kromě souboru `metadata.json` a obrázků z adresáře `jpg` využívá také OCR výstupy ve formátu ALTO XML uložené v adresáři `alto`.

Oproti původní datové sadě se tedy změnila především očekávaná odpověď modelu. Textové hodnoty jsou nově doplněny o atribut `bbox`, který má tvar `[x1, y1, x2, y2]`. Pokud hodnota na stránce chybí nebo její pozici nelze spolehlivě určit, je bbox nastaven na `null`.

#pagebreak()

Struktura požadovaného výstupu je následující:

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

Výsledkem tohoto předzpracování jsou soubory `train_bbox.json`, `eval_bbox.json` a `test_bbox.json`. Soubor `train_bbox.json` obsahoval přibližně 40 tisíc článků a z validačního souboru `eval_bbox.json` byla pro rychlejší vyhodnocování použita zmenšená varianta s 1000 náhodně vybranými položkami. Samotné trénování poté probíhalo stejným způsobem jako u varianty bez bounding boxů. Použit byl stejný framework, stejná metoda LoRA i stejné hlavní hyperparametry. Rozdíl byl pouze ve vstupních souborech, kdy trénovací skript `qwen-train.sh` používal `train_bbox.json` a `eval_bbox.json`, a ve výstupním adresáři, jehož název obsahoval příponu `-bbox`.

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

#let evalTable(b, t) = table(
  columns: (2fr, 1fr, 1fr, 1fr),
  align: (left, right, right, right),

  [#align(center)[*Metric*]],
  [#align(center)[*Base model*]],
  [#align(center)[*Tuned model*]],
  [#align(center)[*Improvement*]],

  [*Time* _(average)_],
  [#strfmt("{:.3} s", b.at(0))],
  [#strfmt("{:.3} s", t.at(0))],
  [],

  [*Valid JSON* _(ratio)_],
  [#strfmt("{:.2} %", b.at(1) * 100)],
  [#strfmt("{:.2} %", t.at(1) * 100)],
  [#strfmt("{:+.2} pp", (t.at(1) - b.at(1)) * 100)],

  [*Title* _(Levenshtein similarity)_],
  [#strfmt("{:.2} %", b.at(2) * 100)],
  [#strfmt("{:.2} %", t.at(2) * 100)],
  [#strfmt("{:+.2} pp", (t.at(2) - b.at(2)) * 100)],

  [*Authors first name* _(Lev. similarity)_],
  [#strfmt("{:.2} %", b.at(3) * 100)],
  [#strfmt("{:.2} %", t.at(3) * 100)],
  [#strfmt("{:+.2} pp", (t.at(3) - b.at(3)) * 100)],

  [*Authors last name* _(Lev. similarity)_],
  [#strfmt("{:.2} %", b.at(4) * 100)],
  [#strfmt("{:.2} %", t.at(4) * 100)],
  [#strfmt("{:+.2} pp", (t.at(4) - b.at(4)) * 100)],

  [*Authors email* _(Lev. similarity)_],
  [#strfmt("{:.2} %", b.at(5) * 100)],
  [#strfmt("{:.2} %", t.at(5) * 100)],
  [#strfmt("{:+.2} pp", (t.at(5) - b.at(5)) * 100)],

  [*Authors institution* _(F1 score)_],
  [#strfmt("{:.2} %", b.at(6) * 100)],
  [#strfmt("{:.2} %", t.at(6) * 100)],
  [#strfmt("{:+.2} pp", (t.at(6) - b.at(6)) * 100)],

  [*Abstract* _(Rouge-L score)_],
  [#strfmt("{:.2} %", b.at(7) * 100)],
  [#strfmt("{:.2} %", t.at(7) * 100)],
  [#strfmt("{:+.2} pp", (t.at(7) - b.at(7)) * 100)],

  [*Keywords* _(F1 score)_],
  [#strfmt("{:.2} %", b.at(8) * 100)],
  [#strfmt("{:.2} %", t.at(8) * 100)],
  [#strfmt("{:+.2} pp", (t.at(8) - b.at(8)) * 100)],

  [*Date* _(exact match)_],
  [#strfmt("{:.2} %", b.at(9) * 100)],
  [#strfmt("{:.2} %", t.at(9) * 100)],
  [#strfmt("{:+.2} pp", (t.at(9) - b.at(9)) * 100)],

  ..if b.len() > 10 and t.len() > 10 {
    (
      [*Title bbox* _(IoU)_],
      [#strfmt("{:.2} %", b.at(10) * 100)],
      [#strfmt("{:.2} %", t.at(10) * 100)],
      [#strfmt("{:+.2} pp", (t.at(10) - b.at(10)) * 100)],

      [*Authors bbox* _(IoU)_],
      [#strfmt("{:.2} %", b.at(11) * 100)],
      [#strfmt("{:.2} %", t.at(11) * 100)],
      [#strfmt("{:+.2} pp", (t.at(11) - b.at(11)) * 100)],

      [*Abstract bbox* _(IoU)_],
      [#strfmt("{:.2} %", b.at(12) * 100)],
      [#strfmt("{:.2} %", t.at(12) * 100)],
      [#strfmt("{:+.2} pp", (t.at(12) - b.at(12)) * 100)],

      [*Keywords bbox* _(IoU)_],
      [#strfmt("{:.2} %", b.at(13) * 100)],
      [#strfmt("{:.2} %", t.at(13) * 100)],
      [#strfmt("{:+.2} pp", (t.at(13) - b.at(13)) * 100)],
    )
  } else {
    ()
  }
)

= Experimenty a vyhodnocení

Jednotlivé experimenty porovnávaly hlavně přesnost základního _base_ modelu s našimi natrénovanými _fine-tuned_ variantami. Evaluace, stejně jako trénování, probíhala na výpočetních uzlech Metacentra poskytovaného organizací CESNET. Konkrétně evaluace byla spouštěna na clusteru s alespoň jedním GPU o minimální kapacitě 16 GB VRAM. Jednotlivé testy probíhaly nezávisle na jiných uzlech, kde se výpočetní prostředky mohly lišit, z čehož plyne, že srovnání časové náročnosti je přesné převážně pro jednotlivé skupiny testů (porovnání základního a natrénovaného modelu). Časy zpracování mezi jednotlivými skupiny jsou převážně orientační, jelikož nebylo možné zajistit identické podmínky pro každé spouštění evaluace. Testovací datová sada, která nebyla použita během tréninku, obsahovala ve většině případů 5 tisíc vzorků, ale pro rychlejší vyhodnocení bylo u některých experimentů použito i pouze 1000 záznamů z testovací množiny.

Při evaluaci byl nejprve spuštěn základní model a vyhodnocena jeho úspěšnost. Poté následovalo uvolnění paměti zabrané základním modelem a spuštění a evaluace natrénovaného modelu. Pro porovnání obou modelů byly použity stejné metriky, konkrétně:

- *Levenshteinova vzdálenost* (resp. její normalizovaná varianta, která kvantifikuje počet editací potřebných k transformaci jednoho řetězce na druhý) pro měření podobnosti většiny textových hodnot (titulek, jména autorů, e-maily),
- *Rouge-L skóre* (které měří nejdelší společnou podposloupnost (LCS) mezi predikovaným a referenčním textem) pro dlouhý text (abstrakt),
- *F1 skóre* (které počítá harmonický průměr přesnosti a úplnosti) pro seznamy hodnot (instituce, klíčová slova),
- *Přesná shoda* pro datum, jelikož datum bylo vyžadováno ve standardním formátu `YYYY-MM-DD`.

Při evaluaci modelů, které kromě textových hodnot generovaly také bounding boxy, byla použita metrika Intersection over Union (IoU) mezi predikovanými a referenčními obalovými obdélníky. Níže jsou uvedeny jednotlivé experimenty a jejich výsledky.

#pagebreak()

=== Qwen2-VL-2B bez bounding boxů

- Počet testovacích vzorků: 4968

#evalTable(
  (
    2.580955,
    0.626811,
    0.615840,
    0.530081,
    0.571376,
    0.579314,
    0.232090,
    0.578004,
    0.486867,
    0.051127
  ),
  (
    4.402444,
    0.856078,
    0.844718,
    0.770650,
    0.836489,
    0.818399,
    0.710760,
    0.812431,
    0.799294,
    0.774758
  )
)

Základní model měl velký problém s dodržením formátu a generováním validního JSONu, což se projevuje i v nízkých hodnotách ostatních metrik. Natrénovaný model generoval validní odpověď v 85 % případů, což je o téměř 23 procentních bodů více než u základního modelu. U trénovaného modelu lze vidět nárůst přesnosti získaných dat úměrný k poměru úspěšně zpracovaných vzorků, nicméně vyšší nárůst lze sledovat u detekce institucí autorů, klíčových slov a datumu, kde dosahoval trénovaný model výrazně lepších výsledků.

=== Qwen2-VL-7B bez bounding boxů

- Počet testovacích vzorků: 4968

#evalTable(
  (
    2.577001,
    0.711956,
    0.697405,
    0.609392,
    0.661348,
    0.708636,
    0.283542,
    0.655777,
    0.594057,
    0.568236
  ),
  (
    4.165789,
    0.863526,
    0.852486,
    0.779276,
    0.846328,
    0.835235,
    0.728724,
    0.816053,
    0.818215,
    0.782004
  )
)

Trénování modelu Qwen druhé generace s výrazně vyšším počtem parametrů vedlo ke značnému zlepšení základního modelu oproti variantě s 2B parametry. Nicméně _fine-tuned_ varianta dosahuje jen nepatrně vyšší přesnosti oproti menší variantě, obě dosahují přesnosti všech parametrů pohybující se kolem 80 %. Jejich časová náročnost je také srovnatelná.

#pagebreak()

=== Qwen2.5-VL-3B bez bounding boxů

- Počet testovacích vzorků: 4968

#evalTable(
  (
    4.435085,
    0.750402,
    0.735893,
    0.573546,
    0.640767,
    0.711307,
    0.319464,
    0.680893,
    0.611935,
    0.480072
  ),
  (
    7.701928,
    0.885668,
    0.873024,
    0.790163,
    0.859225,
    0.847260,
    0.717725,
    0.826086,
    0.818100,
    0.797101
  )
)

Novější generace modelu s 3B parametry dokáže konkurovat, ba dokonce předčít starší model s 7B parametry. Základní _base_ model dokáže vygenerovat validní odpověď pro tři čtvrtiny článků na vstupu, _fine-tuned_ model atakuje hranici téměř 89 %. Stejně jako u předchozích experimentů se pro netrénovanou variantu zdá být nejsložitější detekce institucí jednotlivých autorů, trénovaná varianta správně detekuje 72 % institucí, což je více než dvojnásobné zlepšení oproti základní verzi modelu.

=== Qwen2.5-VL-7B bez bounding boxů

- Počet testovacích vzorků: 4968

#evalTable(
  (
    3.869381,
    0.745169,
    0.731756,
    0.629014,
    0.686987,
    0.739191,
    0.348170,
    0.682756,
    0.617370,
    0.595813
  ),
  (
    4.449032,
    0.880233,
    0.868228,
    0.789050,
    0.855927,
    0.851523,
    0.725044,
    0.823396,
    0.823078,
    0.791666
  )
)

Při navýšení počtu parametrů na 7B se model z hlediska přesnosti extrakce textu oproti 3B variantě již výrazněji nezlepšil. Zaznamenaný pokles průměrné časové náročnosti je s největší pravděpodobností dán pouze alokací výkonnějšího výpočetního uzlu na Metacentru během konkrétního běhu experimentu, nikoliv architekturou modelu.

#pagebreak()

=== Qwen2-VL-2B s bounding boxy

- Počet testovacích vzorků: 4773

#evalTable(
  (
    3.854368,
    0.739786,
    0.723320,
    0.582469,
    0.632107,
    0.667437,
    0.317723,
    0.651229,
    0.519080,
    0.119002,
    0.051539,
    0.031003,
    0.000000,
    0.482748
  ),
  (
    7.715739,
    0.911586,
    0.899172,
    0.816475,
    0.890171,
    0.870095,
    0.750777,
    0.859389,
    0.842758,
    0.824638,
    0.742785,
    0.473249,
    0.703176,
    0.666070
  )
)

Základní model si s detekcí bounding boxů nedokázal téměř vůbec poradit. Trénováním došlo k výraznému zlepšení, přičemž _fine-tuned_ model dokázal extrahovat ohraničující obdélníky s překryvem (IoU) kolem 70 % u titulků, abstraktů a klíčových slov a 47 % u jmen autorů. Extrakce textových hodnot u trénovaného modelu dosahuje vyšší přesnosti než u trénování bez bounding boxů, což je ale způsobeno hlavně drobnou úpravou evaluačního skriptu, který u modelů s bounding boxy spouštěl s méně restriktivním omezením pro maximální počet tokenů v odpovědi.

#pagebreak()

=== Qwen2-VL-7B s bounding boxy

- Počet testovacích vzorků: 1000

#evalTable(
  (
    6.671840,
    0.887000,
    0.865003,
    0.738389,
    0.811491,
    0.874766,
    0.356105,
    0.810052,
    0.679849,
    0.152000,
    0.001485,
    0.006794,
    0.015765,
    0.577069
  ),
  (
    7.861250,
    0.932000,
    0.919203,
    0.834587,
    0.913086,
    0.901547,
    0.782541,
    0.873574,
    0.880672,
    0.842000,
    0.781033,
    0.512689,
    0.754864,
    0.734619
  )
)

Téměř čtyřnásobné zvětšení počtu parametrů vedlo ke zlepšení detekce bounding boxů o přibližně 5 procentních bodů oproti menšímu modelu.

=== Qwen2.5-VL-3B s bounding boxy

- Počet testovacích vzorků: 4497

#evalTable(
  (
    4.761227,
    0.915776,
    0.897931,
    0.715484,
    0.794660,
    0.864092,
    0.383455,
    0.834863,
    0.588910,
    0.070395,
    0.065158,
    0.035328,
    0.000000,
    0.587869
  ),
  (
    9.074043,
    0.942174,
    0.928428,
    0.836801,
    0.911336,
    0.905251,
    0.771667,
    0.873956,
    0.846688,
    0.847894,
    0.852377,
    0.750550,
    0.815005,
    0.771662
  )
)

Model novější generace s 3B parametry již dosahuje poměrně dobré přesnosti napříč všemi parametry. Validní odpověď je generována v 94 % případů, tomu odpovídá i úspěšnost extrakce jednotlivých textových hodnot, která se pohybuje kolem přibližně 85 %. Bounding boxy jsou detekovány s přibližně 80 % přesností, což je zároveň nejviditelnější přínos trénování, jelikož základnímu modelu se s detekcí bounding boxů vůbec nedařilo.

=== Qwen2.5-VL-7B s bounding boxy

- Počet testovacích vzorků: 1000

#evalTable(
  (
    7.399942,
    0.852000,
    0.831707,
    0.708436,
    0.773349,
    0.747101,
    0.364072,
    0.771629,
    0.659948,
    0.005000,
    0.009585,
    0.005447,
    0.016781,
    0.569476
  ),
  (
    12.454316,
    0.958000,
    0.941229,
    0.852365,
    0.931463,
    0.917157,
    0.790950,
    0.888605,
    0.866303,
    0.861000,
    0.854727,
    0.654448,
    0.795568,
    0.760279
  )
)

Nejlepších výsledků dosáhl natrénovaný model `Qwen2.5-VL-7B-Instruct`, který dokázal generovat validní odpověď pro 95,8 % testovacích vzorků. Přesnost extrakce textových hodnot se pohybovala kolem 90 % a detekce bounding boxů dosahovala přibližně 80 %.

= Závěr

V projektu se podařilo sestavit datovou sadu ze serveru arXiv.org o velikosti necelých 50 tisíc článků. Následně došlo k doplnění chybějících hodnot v této datové sadě a také rozšíření metadat o bounding boxy některých z textových hodnot.

Nad vytvořenou datovou sadou probíhalo trénování s využitím frameworku `Qwen-VL-Series-Finetune`#footnote[`https://github.com/2U1/Qwen-VL-Series-Finetune`] <fn> a výpočetního centra Metacentrum poskytovaného organizací CESNET. Trénováno bylo několik modelů, konkrétně dva modely z rodiny `Qwen2-VL` a dva modely z rodiny `Qwen2.5-VL`.

V souladu s teoretickými předpoklady se nejúspěšnějšími ukázaly modely novější generace. Trénování lze považovat za úspěšné, jelikož všechny natrénované modely dosahovaly výrazně lepších výsledků než jejich základní verze. Nejvýraznější zlepšení jsme zaznamenali v extrakci institucí autorů, což je náročný úkol vzhledem k faktu, že autoři jsou s jejich institucemi většinou provázáni pomocí superskriptů. U této hodnoty dosahovala většina natrénovaných modelů zlepšení o přibližně 40 procentních bodů.

Vůbec nejlépe se trénování osvědčilo v detekci ohraničujících obdélníků, jelikož základní modely s touto úlohou měly velké problémy. Trénováním jsme byli schopni dosáhnout přesnosti kolem 80 %.

= GitHub repozitář

`https://github.com/nedelamatej/knn-mis`
