

INFORMAZIONI RILEVANTI DATASET

MODALITÀ
Il dataset include immagini multimodali ottenute con diverse tecniche di risonanza magnetica:
- T1 nativo (T1) -> mostra bene la struttura anatomica.
- T1 pesato con mezzo di contrasto (T1Gd) -> evidenzia le aree con permeabilità vascolare, come il tumore attivo
- T2 pesato (T2) -> evidenzia le aree di edema e liquido *
- T2 Fluid Attenuated Inversion Recovery (FLAIR) -> sopprime il liquido cerebrospinale per mettere in risalto alterazioni del tessuto cerebrale

Queste immagini provengono da molteplici scanner e protocolli clinici di diverse istituzioni.
Per garantire una coerenza spaziale e comparabilità tra le modalità e tra i diversi pazienti, i dati di riferimento sono stati sottoposti a un preprocessing che include:
- Co-registrazione di tutte le immagini sullo stesso template anatomico, per assicurare l’allineamento spaziale tra le modalità e i soggetti.
- Interpolazione delle immagini alla stessa risoluzione isotropica di 1 mm³, per uniformare la scala spaziale e semplificare l’analisi.
- Skull stripping (rimozione del cranio e dei tessuti non cerebrali), per ridurre il rumore e focalizzarsi solo sul tessuto cerebrale rilevante.

Questo preprocessing è fondamentale per migliorare l’efficacia dei metodi di segmentazione multimodale e ridurre le variabilità dovute a differenti scanner o protocolli. 

* immagini di risonanza magnetica (MRI) acquisite con parametri specifici che enfatizzano il tempo di rilassamento T2 dei tessuti. Ogni tessuto nel corpo ha caratteristiche diverse di come si “rilassano” i protoni dopo essere stati eccitati dal campo magnetico della macchina MRI. Il tempo di rilassamento T2 è il tempo con cui il segnale del tessuto decade a causa della perdita di coerenza tra i protoni, facendo apparire alcune strutture brillanti (come il liquido cerebrospinale o l’edema, che hanno tempi T2 lunghi) e altre più scure (tessuti con tempi T2 brevi).


ETICHETTE
- ET descrive le regioni di tumore attivo così come le aree nodulari di potenziamento.
- NETC denota necrosi e cisti all’interno del tumore(core del tumore, la parte centrale che tipicamente viene rimossa chirurgicamente. Si tratta della zona in cui il tumore cresce così velocemente, da soffocare il suo stesso apporto di sangue e ossigeno;quindi il tessuto centrale muore, diventando necrotico, mentre il tumore continua a crescere attivamente intorno.)

- SNFH include edema(accumulo anomalo di liquido nei tessuti, spesso causato da infiammazione, trauma, infezione o crescita tumorale), tumore infiltrante e cambiamenti post-trattamento.
- RC consiste in cavità di resezione recenti e croniche e tipicamente contiene fluido, sangue, aria e/o materiali proteici.
- Il core tumorale (ET più NETC) descrive ciò che viene tipicamente rimosso durante un intervento chirurgico.
- L’intero tumore (ET più SNFH più NETC) definisce l’estensione totale del tumore, includendo il core tumorale, il tumore infiltrante, l’edema peritumorale e i cambiamenti correlati al trattamento.

METRICHE DI VALUTAZIONE
- Dice Similarity Coefficient (DSC) a livello di lesione, che misura la sovrapposizione voxel-wise tra segmentazioni predette e segmentazioni di riferimento, ignorando i voxel veri negativi (cioè le regioni non tumorali identificate correttamente come tali), così la metrica non è falsamente elevata nei dataset con tanto background sano.
- Normalized Surface Distance (NSD), che misura la sovrapposizione dei confini tra le segmentazioni predette e quelle di riferimento.

La metrica DSC a livello di lesione è stata progettata per valutare le performance del modello al livello delle singole lesioni piuttosto che sull’intera immagine. Questa metodologia evita che la valutazione favorisca i modelli che rilevano solo le lesioni più grandi, una limitazione spesso riscontrata nel DSC standard. Valutando i modelli lesione per lesione, possiamo comprendere meglio la loro capacità di segmentare malattie multifocali(tumori con più focolai distinti) e multicentriche.


