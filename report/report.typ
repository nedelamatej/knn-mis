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

Před samotným trénováním byla datová sada převedena do formátu očekávaného frameworkem `Qwen-VL-Series-Finetune`#footnote[`https://github.com/2U1/Qwen-VL-Series-Finetune`] <fn>. Python skript načetl zdrojová metadata ze souboru a ke každému záznamu přiřadil odpovídající obrázek první strany článku ve formátu PNG. Pro každý vzorek byl vytvořen vstup tvořený instrukcí pro extrakci metadat a referenční výstup ve formátu JSON obsahující název článku, seznam autorů, abstrakt a klíčová slova.

```json
{
  "id": "0704.0002",
  "image": "0704.0002.png",
  "conversations": [
    { "from": "human", "value": "<image>\nExtract metadata from this first page of a ..." },
    { "from": "gpt", "value": "{\"title\": \"Sparsity-certifying Graph ..." }
  ]
}
```

= Trénování neuronové sítě

Pro trénování byl použit framework `Qwen-VL-Series-Finetune` @fn a jako základní model byl zvolen `Qwen2.5-VL-3B-Instruct`. Učení probíhalo na výpočetním clusteru prostřednictvím skriptu, který zajistil přípravu pracovního prostředí, přesun dat na lokální prostor a spuštění trénovacího skriptu. Model byl dolaďován metodou LoRA, která umožňuje efektivní adaptaci velkého předtrénovaného modelu při nižších paměťových i výpočetních nárocích. Pro trénování modelu bylo použito přibližně 16 000 vědeckých článků. Vstupem byly obrázky jejich titulních stran a cílem modelu bylo generovat metadata v definovaném JSON schématu.

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
