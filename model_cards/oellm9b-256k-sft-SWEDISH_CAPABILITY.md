# Svensk förmåga: OELLM 9B 256K SFT

Detta är en kvalitativ genomgång av svensk förmåga hos
[`openeurollm/oellm-9b-256k-sft`](https://huggingface.co/openeurollm/oellm-9b-256k-sft).
Modellen är ett experimentellt SFT-checkpoint, inte en färdig svensk assistent eller en
resonemangsspecialiserad modell.

## Sammanfattning

Modellen kan skriva sammanhängande svenska, besvara enkla vardagsfrågor, följa vissa tydliga
formatkrav och göra korta sammanfattningar. Samtidigt är den faktamässiga och logiska
tillförlitligheten låg. Svaren kan se övertygande ut även när de innehåller påhittade fakta,
felaktig grammatikförklaring eller motsägelsefulla uträkningar.

Den praktiska slutsatsen är:

> **Svensk ytkompetens finns, men svaren behöver faktakontroll och bör inte användas som
> tillförlitliga utan verifiering.**

## Testunderlag

Två kompletterande tester användes:

1. **24 öppna svenska promptar** i nio kategorier: sentiment, Sverigekunskap, allmänkunskap,
   resonemang, sammanfattning, språkriktighet, instruktionsefterlevnad, vardagskunskap och
   kreativt skrivande.
2. **EU-holdout**, ett deterministiskt flerspråkigt test. Körningen omfattade 1 368 exempel,
   varav 36 svenska exempel efter att långkontextkategorin exkluderats.

De öppna svaren genererades med sampling (`temperature=0.7`, `top_p=0.95`) och högst 400 nya
tokens. Varje prompt kördes en gång. Resultatet mäter därför ett stickprov av modellens beteende,
inte en stabil eller statistiskt säker förmågenivå.

### Resultat i korthet

| Mått | Resultat | Tolkning |
|---|---:|---|
| Svenska öppna promptar | 24 | Kvalitativt stickprov |
| Genomsnittlig svensk språkdetektionssannolikhet | 0,96 | Svaren identifieras nästan alltid som svenska; detta mäter inte korrekthet |
| EU-holdout, svenska | 33,3 % | 12 av 36 deterministiskt rättade svenska exempel |
| EU-holdout, alla språk | 23,2 % | 1 368 exempel totalt |
| Instruction following, alla språk | 63,2 % | Starkaste kategorin tillsammans med grounded QA |
| Grounded QA, alla språk | 58,6 % | Kan använda given information i enklare frågor |
| Reasoning/math, alla språk | 5,9 % | Tydlig svaghet |
| Summarization, alla språk | 2,6 % | Den strikta automatiska rättningen visar låg robusthet |

Fullständiga maskinläsbara resultat finns i
[`eval/swedish-rubric.json`](./swedish-rubric.json) och
[`eval/eu-holdouts-dev.json`](./eu-holdouts-dev.json).

## Exempel på sådant modellen gör bra

### 1. Följer ett exakt listformat

**Prompt**

> Skriv en punktlista med exakt fyra tips för att sova bättre. Varje punkt ska vara högst tio
> ord lång.

**Modellens svar**

> - Håll en regelbunden sömnrutin.  
> - Skapa en avslappnande sovmiljö.  
> - Undvik koffein på kvällen.  
> - Använd en varm dusch för att lugna kroppen.

**Bedömning:** Bra. Svaret har exakt fyra punkter, samtliga under tio ord, och råden är relevanta.

### 2. Skriver ett kort formellt meddelande

**Prompt**

> Skriv ett formellt mejl på svenska till en hyresvärd där du anmäler en trasig diskmaskin och
> ber om reparation. Håll det under 80 ord.

**Modellens svar**

> Kära hyresvärd,
>
> Jag vill informera om att diskmaskinen i lägenheten [lägenhetsnummer] är trasig. Jag ber om
> att få en reparation snarast.
>
> Med vänliga hälsningar,  
> [Ditt namn]

**Bedömning:** Bra. Mejlet är begripligt, formellt, innehåller rätt begäran och håller sig tydligt
under ordgränsen.

### 3. Bevarar huvudpunkterna i en kort nyhetssammanfattning

**Prompt**

> Ge en kort sammanfattning (max tre meningar) av vad nyheten handlar om: "Regeringen
> presenterade idag en ny satsning på järnvägsunderhåll. Satsningen omfattar tio miljarder kronor
> över fyra år och syftar till att minska förseningarna i tågtrafiken. Oppositionen menar att
> beloppet är otillräckligt."

**Modellens svar**

> Regeringen presenterade idag en ny satsning på järnvägsunderhåll, som omfattar tio miljarder
> kronor över fyra år. Satsningen syftar till att minska förseningarna i tågtrafiken. Oppositionen
> menar att beloppet är otillräckligt.

**Bedömning:** Bra. Alla centrala uppgifter finns kvar och begränsningen på tre meningar följs.

### 4. Ger ett korrekt och användbart vardagssvar

**Prompt**

> Varför är det en dålig idé att lämna mjölk framme på köksbänken över natten?

**Modellens svar, förkortat**

> Att lämna mjölk framme på köksbänken över natten kan leda till att bakterier börjar växa i
> mjölken. [...] Det är därför viktigt att alltid kyla mjölk och andra mjölkprodukter efter
> användning.

**Bedömning:** Bra. Svaret förklarar den relevanta risken och ger ett praktiskt råd på naturlig
svenska.

## Exempel på fel och begränsningar

### 1. Korrekt namnlista följs av grovt felaktiga fakta

**Prompt**

> Vilka är Sveriges tre största sjöar till ytan, i storleksordning?

**Modellens svar**

> De tre största sjöarna till ytan i Sverige är Vänern, Vättern och Mälaren. Vänern är den största
> sjö som finns i Sverige, med en yta på 25 650 km², följt av Vättern med en yta på 12 388 km² och
> Mälaren med en yta på 10 907 km².

**Bedömning:** Blandat. Sjöarna och ordningen är korrekta, men alla tre arealerna är grovt
överdrivna. Exemplet visar ett viktigt riskmönster: modellen kan börja rätt och därefter lägga till
självsäkert formulerade hallucinationer som användaren inte bad om.

### 2. Påhittade verk av Selma Lagerlöf

**Prompt**

> Vem var Selma Lagerlöf och varför är hon viktig i svensk litteraturhistoria?

**Modellens svar, utdrag**

> Selma Lagerlöf var en svensk författare som skrev bl.a. romanen "Göran och Gösta" och
> "Trollkarlen från Österrike". [...] Dessutom var hon den första kvinnliga författaren i Sverige
> som vann Nobelpriset i litteratur, en ära som hon fick 1909.

**Bedömning:** Dåligt. De angivna boktitlarna är påhittade. Årtalet 1909 är korrekt, men den
viktiga beskrivningen är att Lagerlöf var den första kvinnan som tilldelades Nobelpriset i
litteratur, inte bara den första kvinnliga författaren i Sverige som fick det.

### 3. Rätt slutsvar trots motsägelsefull uträkning

**Prompt**

> En affär säljer äpplen för 12 kronor styck. Om du köper 7 äpplen och betalar med en hundralapp,
> hur mycket får du tillbaka? Visa uträkningen.

**Modellens svar, utdrag**

> 7 äpplen × 12 kr/äpple = 84 kr  
> 84 kr - 100 kr = -16 kr  
> Detta innebär att du får tillbaka 16 kronor.

**Bedömning:** Blandat men otillförlitligt. Slutsvaret 16 kronor är rätt, men subtraktionen är
uppställd åt fel håll och resultatet motsäger den avslutande meningen. Ett korrekt resonemang är
`100 - 84 = 16`.

### 4. Missar tidsförskjutningen i ett åldersproblem

**Prompt**

> Anna är dubbelt så gammal som Erik. Om fem år är deras sammanlagda ålder 40 år. Hur gamla är de
> nu? Resonera steg för steg.

**Modellens svar, utdrag**

> Anna + Erik = 40 [...] Erik = 40 / 3 [...] Erik 13 år [...] Anna 26 år [...] Anna och Erik båda
> 26 år gamla nu.

**Bedömning:** Fel. Modellen glömmer att båda personerna blir fem år äldre och producerar dessutom
en självmotsägande sammanfattning. Korrekt ekvation är `(2E + 5) + (E + 5) = 40`, vilket ger Erik
10 år och Anna 20 år.

### 5. Korrigerar inte ett grammatikfel

**Prompt**

> Rätta eventuella grammatik- och stavfel i meningen och förklara vad du ändrade: "Han gick till
> affären för att köpa mjölk men han glömde sin plånbok hemma så han kunde inte handlade något."

**Modellens svar, utdrag**

> Han gick till affären för att köpa mjölk men han glömde sin plånbok hemma så han kunde inte
> handlade något.
>
> Först och främst, jag bytte ut ordet "handlade" mot "handlade". [...] Sedan, jag ändrade
> "glömde" till "glömde".

**Bedömning:** Fel. Modellen upprepar originalfelet och hävdar att den gjort ändringar som inte har
gjorts. Korrekt verbform efter modalverbet *kunde* är *handla*: "...så han kunde inte handla
något."

### 6. Felaktig förklaring av *de* och *dem*

**Prompt**

> Förklara skillnaden mellan "de" och "dem" och ge ett exempel på korrekt användning av vardera.

**Modellens svar, utdrag**

> "De" används för att hänvisa till en grupp personer som är okända eller som inte har
> specificerats. [...] "Dem" används för att hänvisa till en specifik grupp personer [...] Till
> exempel: "Dem sa att de skulle komma till festen igår kväll."

**Bedömning:** Fel. Skillnaden gäller grammatisk funktion: *de* är subjektsform och *dem* är
objektsform. Exemplet "Dem sa ..." är fel; det ska vara "De sa ...".

### 7. Totalt format- och innehållsfel

**Prompt**

> Lista fem svenska städer i bokstavsordning, en per rad, utan någon annan text.

**Modellens svar**

> A-s-t-a-n  
> A-r-e-s-t-a-  
> B-y-s-t-a-  
> C-h-a-n-  
> D-e-l-f-o-  
> E-m-m-a-  
> F-r-o-

**Bedömning:** Fel. Svaret innehåller sju rader i stället för fem och inga identifierbara svenska
städer. Detta är ett tydligt exempel på att modellen inte är robust mot exakta listkrav.

## Samlad bedömning

### Relativa styrkor

- producerar vanligtvis text som identifieras som svenska;
- hanterar enkla vardagsfrågor och korta formella texter;
- kan följa vissa explicita längd- och formatkrav;
- kan göra enklare, källbundna sammanfattningar;
- har användbar grundläggande samtalsstil.

### Viktiga svagheter

- hallucinerar fakta och kan blanda korrekta och felaktiga uppgifter i samma svar;
- svagt matematiskt och logiskt resonemang;
- opålitlig grammatisk analys trots relativt flytande ytspråk;
- upprepningar och översättningsliknande formuleringar;
- instabil efterlevnad av exakta antal, listor och format;
- kan avsluta svar mitt i en mening när generationsgränsen nås.

## Rekommenderad användning

Modellen är lämplig för experiment, forskning och fortsatt post-training. För svensk
produktionsanvändning bör svar faktakontrolleras, och modellen bör inte ensam användas för medicinsk,
juridisk, ekonomisk eller annan högkonsekvensinformation.

Nästa rättvisa jämförelse bör köra exakt samma svenska promptar med deterministisk avkodning på både
basmodellen och SFT-checkpointen. Flera seeds eller upprepningar behövs för att uppskatta variationen.
En bredare svensk benchmark bör dessutom inkludera etablerade svenska uppgifter och mänsklig
blindbedömning av språk, korrekthet, instruktionsefterlevnad och hjälpsamhet.

## Reproducerbarhet

- Modell: [`openeurollm/oellm-9b-256k-sft`](https://huggingface.co/openeurollm/oellm-9b-256k-sft)
- Basmodell: [`openeurollm/oellm-9b-256k-theta64m-prelude`](https://huggingface.co/openeurollm/oellm-9b-256k-theta64m-prelude)
- Prompter och evaluator: [`BirgerMoell/qwen35-posttrain`](https://github.com/BirgerMoell/qwen35-posttrain)
- Råa svenska svar: [`eval/swedish-rubric.json`](./swedish-rubric.json)
- EU-holdout-resultat: [`eval/eu-holdouts-dev.json`](./eu-holdouts-dev.json)

Språkdetektering och automatiska holdout-poäng ska inte tolkas som en fullständig kvalitetsbedömning.
Kommentarerna ovan är en manuell kvalitativ granskning av de publicerade modellutdata, inte en
blindad expertstudie.
