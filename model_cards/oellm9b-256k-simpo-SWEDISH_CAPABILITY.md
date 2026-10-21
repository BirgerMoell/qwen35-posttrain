# Svensk förmåga: OELLM 9B 256K SimPO

Detta är en kvalitativ genomgång av svensk förmåga hos
[`birgermoell/oellm-9b-256k-simpo`](https://huggingface.co/birgermoell/oellm-9b-256k-simpo).
Modellen är ett experimentellt SimPO-checkpoint ovanpå
[`birgermoell/oellm-9b-256k-sft`](https://huggingface.co/birgermoell/oellm-9b-256k-sft),
inte en färdig svensk assistent eller en resonemangsspecialiserad modell.

## Sammanfattning

Modellen skriver oftast flytande och begriplig svenska. Den klarar flera enkla vardagsfrågor,
korta meddelanden och vissa tydliga formatkrav. SimPO-svaren är i detta stickprov kortare än
SFT-svaren och mindre benägna att fastna i långa upprepningar.

Detta gav däremot ingen allmän kvalitetsvinst. På samma deterministiska EU-holdout sjönk totalpoängen
från 23,2 % för SFT till 21,6 % för SimPO. Svensk delpoäng var oförändrad på 33,3 %. Den manuella
genomgången visar fortsatt grova fel i enkel aritmetik, logik, svensk grammatik och faktakunskap.
Vissa fel blev värre efter SimPO, trots att träningsmåttet för preferensnoggrannhet steg under
optimeringen.

Den praktiska slutsatsen är:

> **SimPO gör ofta svaren kortare, men förbättrar inte den mätta allmänna eller svenska
> tillförlitligheten. Flytande svenska ska inte förväxlas med korrekt svenska eller korrekta fakta.**

## Testunderlag

Samma protokoll som för SFT-modellen användes:

1. **24 öppna svenska promptar** i nio kategorier: sentiment, Sverigekunskap, allmänkunskap,
   resonemang, sammanfattning, språkriktighet, instruktionsefterlevnad, vardagskunskap och
   kreativt skrivande.
2. **EU-holdout**, ett deterministiskt flerspråkigt test med 1 368 exempel, varav 36 svenska,
   efter att långkontextkategorin exkluderats.

De öppna svaren genererades en gång vardera med `temperature=0.7`, `top_p=0.95` och högst 400 nya
tokens. Resultaten är därför ett kvalitativt stickprov, inte en stabil skattning över flera seeds.
Ingen LLM-domare användes; bedömningarna nedan är en manuell genomgång av samtliga 24 svar.

Körningen gjordes på den exakta BF16-artifakten som publicerades på Hugging Face. Export och
inferenstest var LUMI-jobb `21040068`; denna utvärdering var jobb `21040549`.

### Resultat i korthet

| Mått | SFT | SimPO | Förändring |
|---|---:|---:|---:|
| Genomsnittlig svensk språkdetektionssannolikhet | 0,96 | 0,96 | 0,00 |
| EU-holdout, svenska | 33,3 % | 33,3 % | 0,0 procentenheter |
| EU-holdout, alla språk | 23,2 % | 21,6 % | -1,6 procentenheter |
| Genomsnittlig svarslängd, svenska promptar | 72,3 ord | 56,9 ord | -15,4 ord |
| Median svarslängd | 69,0 ord | 50,5 ord | -18,5 ord |
| Tomma svenska svar | 0/24 | 0/24 | oförändrat |

Språkdetektorn gav 0,0 för det enordiga svaret `Neutral`. Ordet är giltig svenska, men för kort för
säker språkidentifiering. Det finns därför inget tydligt belägg för faktisk språkväxling i de 24
svaren. Inget SimPO-svar var tomt eller uppenbart avklippt vid generationsgränsen; det längsta
svaret var i stället ett 202 ord långt brott mot instruktionen att skriva några meningar.

### EU-holdout per kategori

| Kategori | SFT | SimPO | Förändring |
|---|---:|---:|---:|
| Instruction following | 63,2 % | 59,2 % | -4,0 |
| Grounded QA | 58,6 % | 53,9 % | -4,7 |
| Translationese preference | 55,9 % | 53,3 % | -2,6 |
| Civic safety | 22,4 % | 18,4 % | -4,0 |
| No answer | 0,7 % | 8,6 % | +7,9 |
| Reasoning/math | 5,9 % | 0,0 % | -5,9 |
| Summarization | 2,6 % | 0,7 % | -1,9 |
| Locale formatting | 0,0 % | 0,0 % | 0,0 |
| Tool calling | 0,0 % | 0,0 % | 0,0 |

Ökningen i `no_answer` är uppgiftsspecifik och väger inte upp försämringarna i de flesta andra
kategorier. Träningsmåttet `rewards/accuracies=0.7734` ska inte jämställas med denna oberoende
holdout-poäng.

Fullständiga maskinläsbara resultat finns i
[`eval/swedish-rubric.json`](./swedish-rubric.json) och
[`eval/eu-holdouts-dev.json`](./eu-holdouts-dev.json).

## Exempel på sådant modellen gör bra

### 1. Kort och korrekt negativ sentimentbedömning

**Prompt**

> Är följande kundkommentar positiv eller negativ? Förklara varför. "Jag hade höga
> förväntningar men blev tyvärr besviken på kvaliteten."

**Modellens svar**

> Den här kommentaren är negativ. Kunden förväntade sig en hög kvalitet men blev besviken på
> produktens prestanda.

**Bedömning:** Bra. Klassificeringen och motiveringen är korrekta och svaret är betydligt mindre
upprepande än SFT-svaret på samma prompt.

### 2. Bevarar alla huvudpunkter i en kort nyhet

**Prompt**

> Ge en kort sammanfattning (max tre meningar) av vad nyheten handlar om: "Regeringen presenterade
> idag en ny satsning på järnvägsunderhåll. Satsningen omfattar tio miljarder kronor över fyra år
> och syftar till att minska förseningarna i tågtrafiken. Oppositionen menar att beloppet är
> otillräckligt."

**Modellens svar**

> Regeringen presenterade idag en ny satsning på järnvägsunderhåll. Satsningen omfattar tio
> miljarder kronor över fyra år och syftar till att minska förseningarna i tågtrafiken.
> Oppositionen menar att beloppet är otillräckligt.

**Bedömning:** Bra. Alla fakta bevaras och gränsen på tre meningar följs, även om svaret nästan är
en kopia snarare än en komprimering.

### 3. Skriver ett kort formellt felmeddelande

**Prompt**

> Skriv ett formellt mejl på svenska till en hyresvärd där du anmäler en trasig diskmaskin och ber
> om reparation. Håll det under 80 ord.

**Modellens svar, förkortat**

> Kära hyresvärd,
>
> Jag beklagar att diskmaskinen är trasig och ber om omedelbar reparation. [...] Jag hoppas att ni
> kan åtgärda problemet snarast.
>
> Med vänliga hälsningar,
>
> [Ditt namn]

**Bedömning:** I huvudsak bra. Svaret är begripligt, formellt, under 80 ord och ber om reparation.
Det lägger dock till den obestyrkta uppgiften att hyresvärden redan kontaktats.

### 4. Ger exakt tre praktiska cykelråd

**Prompt**

> Det regnar ute och du ska cykla till jobbet. Vad bör du tänka på? Ge tre praktiska råd.

**Modellens svar, förkortat**

> 1. Kontrollera cykeln innan du cyklar. [...]  
> 2. Använd reflexer och belysning. [...]  
> 3. Var försiktig med vatten och is. [...]

**Bedömning:** Bra format och användbart innehåll. Jämfört med SFT-svaret följer SimPO exakt
antalet råd och undviker upprepningen av samma handskråd.

## Exempel på fel och begränsningar

### 1. Fel lista över Sveriges största sjöar

**Prompt**

> Vilka är Sveriges tre största sjöar till ytan, i storleksordning?

**Modellens svar**

> Vänern (5558 km²) är den största, efterfulld av Mälaren (252 km²) och Storsjön (167 km²).

**Bedömning:** Fel. Rätt ordning är Vänern, Vättern och Mälaren. Arealerna för Mälaren och Storsjön
är också fel. SFT-modellen hade rätt namn och ordning men fel arealer; SimPO har alltså försämrat
själva kärnsvaret.

### 2. Fortsatt hallucination om Selma Lagerlöf

**Prompt**

> Vem var Selma Lagerlöf och varför är hon viktig i svensk litteraturhistoria?

**Modellens svar, utdrag**

> Lagerlöfs verk, som inkluderar romanen "Göran i Botten" och novellsamlingen "Röde Rum", handlar
> ofta om kvinnors situation i samhället [...]

**Bedömning:** Fel. `Göran i Botten` är påhittad och *Röda rummet* skrevs av August Strindberg.
Svaret anger däremot korrekt att Lagerlöf blev den första kvinnan att få Nobelpriset i litteratur
1909. Mönstret från SFT kvarstår: korrekta fakta blandas med självsäkra hallucinationer.

### 3. Enkel växelräkning ger fel slutsvar

**Prompt**

> En affär säljer äpplen för 12 kronor styck. Om du köper 7 äpplen och betalar med en hundralapp,
> hur mycket får du tillbaka? Visa uträkningen.

**Modellens svar, utdrag**

> 7 äpplen x 12 kronor/äpple = 84 kronor  
> [...]  
> 1 hundralapp - 84 kronor = 156 kronor - 84 kronor = 72 kronor  
> [...] du får tillbaka 72 kronor [...]

**Bedömning:** Fel. Rätt svar är `100 - 84 = 16`. SFT gav rätt slutsvar 16 trots en motsägelsefull
uppställning; SimPO ändrar även slutsvaret till fel värde. Detta stämmer med holdoutens försämring
från 5,9 % till 0,0 % i reasoning/math.

### 4. Två separata logikfel

På åldersproblemet sätter modellen både Anna och Erik till 15 år efter att felaktigt ha gjort bådas
framtida ålder till `2x + 5`. Korrekt svar är Anna 20 och Erik 10 år.

På frågan om alla katter är däggdjur och vissa däggdjur kan simma svarar modellen:

> Ja, det kan man dra slutsatsen av.

Det följer inte logiskt. De simkunniga däggdjuren behöver inte vara katter. Båda svaren visar att
flytande steg-för-steg-text inte garanterar giltigt resonemang.

### 5. Korrigerar fortfarande inte grammatikfelet

**Prompt**

> Rätta eventuella grammatik- och stavfel [...] "...så han kunde inte handlade något."

**Modellens svar**

> Han gick till affären för att köpa mjölk men han glömde sin plånbok hemma så han kunde inte
> handlade något.

**Bedömning:** Fel. Efter modalverbet *kunde* ska infinitiv användas: *kunde inte handla*. SimPO
undviker SFT-svarets långa påhittade förklaring, men gör fortfarande inte den begärda korrigeringen
eller förklaringen.

### 6. Felaktig förklaring av *de* och *dem*

**Modellens svar, utdrag**

> "De" används för att hänvisa till en grupp människor som består av mer än två personer, och
> "dem" [...] två eller fler personer. [...] "Dem vill ha en ny bil".

**Bedömning:** Fel. Skillnaden gäller subjektsform (*de*) och objektsform (*dem*), inte gruppens
storlek. Exemplet `Dem vill` är grammatiskt fel. Denna svaghet är oförändrad från SFT.

### 7. Misslyckas med exakt stadlista

**Prompt**

> Lista fem svenska städer i bokstavsordning, en per rad, utan någon annan text.

**Modellens svar**

> ABBA Arvika  
> ABBA Boden  
> ABBA Bollnäs  
> ABBA Borlänge

**Bedömning:** Fel. Svaret har fyra i stället för fem rader och lägger till `ABBA` på varje rad.
Stadsnamnen är åtminstone identifierbara och alfabetiskt ordnade, till skillnad från SFT:s
fragmenterade nonsenslista, men det exakta formatkravet uppfylls fortfarande inte.

### 8. Vetenskaplig förenkling blir sakfel

I barnförklaringen om den blå himlen säger modellen att solljuset träffar små vattendroppar och att
blått har den kortaste våglängden. Den relevanta huvudmekanismen är Rayleighspridning från luftens
molekyler; violett har kortare våglängd än blått. Svaret låter pedagogiskt men är faktamässigt
missvisande.

### 9. Reflekterande svar ignorerar längdönskemålet

På `Vad är meningen med livet? Svara reflekterande i några meningar.` producerar modellen 202 ord i
tre stycken, med upprepningar av *syfte*, *mening* och *medkännande*. Det är testets längsta svar och
visar att den genomsnittliga förkortningen efter SimPO inte innebär stabil längdkontroll.

## Direkt jämförelse med SFT

### Observerade förbättringar

- kortare svar i genomsnitt och mindre ordagrann upprepning;
- ett fullständigt och mer fokuserat svar om intervjuoro där SFT-svaret tog slut mitt i en mening;
- exakt tre cykelråd utan SFT-svarets repetitiva handsktext;
- stadlistan innehåller verkliga ortsnamn, även om formatet fortfarande misslyckas;
- korrekt formulering att Lagerlöf var den första kvinnan som fick Nobelpriset i litteratur.

### Observerade försämringar eller kvarstående fel

- total holdout -1,6 procentenheter; åtta av nio kategorier minskar eller står still;
- reasoning/math faller till 0,0 %;
- rätt slutsvar i äppeluppgiften blir fel efter SimPO;
- den tidigare korrekta sjönamnlistan blir fel;
- faktahallucinationer, ogiltiga logiska slutsatser och fel om *de/dem* kvarstår;
- instruktionsefterlevnad sjunker 4,0 procentenheter och exakt stadformat misslyckas fortfarande.

Det går inte att avgöra om skillnaderna i de öppna svaren orsakas av SimPO eller av sampling från en
enda generation. Holdout-resultatet är deterministiskt och ger starkare evidens för den övergripande
regressionen på just den testsviten.

## Samlad bedömning

### Relativa styrkor

- producerar normalt text som identifieras som svenska;
- ger ofta korta, läsbara vardagssvar;
- klarar vissa enkla antal-, längd- och strukturkrav;
- sammanfattar en kort nyhet utan att tappa angivna fakta;
- är mindre repetitiv än SFT i flera öppna exempel.

### Viktiga svagheter

- låg faktamässig och logisk tillförlitlighet;
- mycket svag enkel matematik och formell slutledning;
- opålitlig analys av svensk grammatik;
- blandar korrekta uppgifter med påhittade detaljer;
- instabil efterlevnad av exakta list- och längdkrav;
- förbättrat träningsmål överfördes inte till bättre allmän holdout-poäng.

## Rekommenderad användning

Modellen är lämplig för experiment, forskning och fortsatt utvärdering. Svenska svar behöver
faktakontrolleras. Modellen bör inte ensam användas för medicinsk, juridisk, ekonomisk eller annan
högkonsekvensinformation.

Den viktigaste återstående frågan är separat: om SimPO bevarade SFT-modellens långkontextretrieval.
Det måste mätas med samma NIAH-grid och rapporteras som full retrieval respektive exakt format;
denna kortkontextutvärdering besvarar inte den frågan.

## Reproducerbarhet

- Modell: [`birgermoell/oellm-9b-256k-simpo`](https://huggingface.co/birgermoell/oellm-9b-256k-simpo)
- SFT-baslinje: [`birgermoell/oellm-9b-256k-sft`](https://huggingface.co/birgermoell/oellm-9b-256k-sft)
- Basmodell: [`openeurollm/oellm-9b-256k-theta64m-prelude`](https://huggingface.co/openeurollm/oellm-9b-256k-theta64m-prelude)
- Kod och prompter: [`BirgerMoell/qwen35-posttrain`](https://github.com/BirgerMoell/qwen35-posttrain)
- LUMI-export/inferenstest: jobb `21040068`
- LUMI-utvärdering: jobb `21040549`
- Råa svenska svar: [`eval/swedish-rubric.json`](./swedish-rubric.json)
- EU-holdout-resultat: [`eval/eu-holdouts-dev.json`](./eu-holdouts-dev.json)

Språkdetektion och automatiska holdout-poäng är inte en fullständig kvalitetsbedömning. Exemplen
ovan är en manuell kvalitativ granskning, inte en blindad expertstudie.
