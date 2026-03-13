#set document(
  title: "[KNN] Projekt - extrakce metadat vědeckých článků",
  author: ("David Machů (xmachu05)", "Matěj Neděla (xnedel11)"),
  date: none,
)

#show raw: set block(width: 100%, fill: black.lighten(95%), inset: 5pt)
#show raw: set text(size: 8pt)

#title()

= Řešený problém

Projekt se zaměřuje na extrakci metadat z titulních stran vědeckých článků. Cílem je s pomocí existujících řešení poloautomaticky připravit datovou sadu obsahující název, autory, abstrakt, klíčová slova a instituce. A následně natrénovat destilovanou neuronovou síť, která bude schopna tyto informace extrahovat z obrazové podoby titulní strany článku.

Systém lze abstraktně modelovat jako pipeline na obrázku níže. Při inferenci je na vstupu PDF nebo PNG obrázek vědeckého článku. Z článku se získá titulní strana ve vizuální podobě a pomocí OCR také textový obsah stránky. VLM následně zpracuje obraz titulní strany spolu s textem a identifikuje požadovaná metadata. Výstupem je strukturovaný JSON soubor. Při tréninku se navíc používá referenční JSON s anotovanými metadaty, který slouží pro porovnání správnosti predikce.

#image("pipeline.svg")

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

= Testovací dataset

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

= Plán řešení a experimenty

=== Získání a předzpracování dat

V první fázi dojde k vybrání vzorku dat (přibližně 2000 článků) a stažení odpovídajících PDF dokumentů z arXiv.org skrz jejich API. Následně bude provedena extrakce titulních stran a jejich převod do jednotného obrazového formátu. Pomocí existujících řešení a dostupných nativních PDF dokumentů budou metadata rozšířena o chybějící parametry (např. e-mail a instituce autorů nebo klíčová slova).

=== Trénování neuronové sítě

Na takto vygenerovaném datasetu bude natrénována neuronová síť.

=== Experimenty a vyhodnocení

Na závěr dojde k vyhodnocení přesnosti obsahu pomocí exaktní textové shody (případně Levenštejnovy vzdálenosti) vůči referenčním metadatům a porovnání náročnosti, rychlosti a přesnosti mezi existujícími řešeními a výslednou neuronovou sítí.

= GitHub repozitář

`https://github.com/nedelamatej/knn-mis`

#v(1fr)

#align(right)[_David Machů (xmachu05), Matěj Neděla (xnedel11)_]
