
veloce e poco ordinato commento al codice

Facciamo logging, per evitare print e mappiamo le modalità, così da passare dal nome logico (t1) al file vero (*-t1n.nii.gz), senza troppe complicazioni. 

def create_catalog_csv
Creiamo un catalogo !Se già presente saltiamo. Le colonne del catalogo sono: ['case', 'subject_id', 'timepoint', 'treatment_status', 't1ce', 't1', 'flair', 't2', 'seg']. Processiamo poi ogni cartella,dividendo il nome della cartella con -. Se non ha almeno 4 parti salta. Estraimo poi le informazioni di base: subject_id: terza parte del nome cartella, timepoint: quarta parte, treatment_status: “pre” se timepoint inizia con “0”, altrimenti “post”. Costruiamo la riga con le informazioni di base. Aggiungiamo i percorsi file. Salviamo la riga con tutte le informazioni nel CSV.

def check_missing_files
Controlliamo i valori mancanti. Per ogni riga se mancano delle moadalità obbligatorie, aggiungiamo casella vuota.
Per ogni modalità prendiamo il percorso del file dal CSV  e controlliamo se è vuoto, se è NaN (campo mancante), se il file non esiste sul disco. Se manca aggiungiamo modalità a case_missing. Ci resituisce un dizionario con le modalità mancanti.

def normalize_minmax
Le maschere di validità (valid_mask) indicano quali immagini mancano in quel campione e quali no.Se un’immagine esiste → True; se manca → False.  La maschera è True dove i voxel > 0. Se la maschera ha almeno un voxel: creiamo un nuovo array vuoto (img_norm) con stessi shape e dtype=float32 e calcoliamo min e max dei voxel “validi” (quelli mask=True) e normalizziamo. Se la maschera è vuota (nessun voxel “valido”), converte l’array in float32 e lo restituisce senza normalizzare. 

def _check_and_filter_missing_files
Verifichiamo se per un caso mancano file obbligatori (required_modalities). Se drop_missing=True, rimuoviamo dal dataset tutti i casi che hanno file obbligatori mancanti. Per le modalità opzionali, possiamo decidere di usare un placeholder (array di zeri) per sostituire i dati mancanti. Se per esempio manca il file per t2 o seg in un caso, quel caso viene rimosso, perché mancano dati fondamentali.

def __getitem__(self, idx):
Vogliamo capire la forma (dimensioni) delle immagini per creare placeholder (array di zeri) in caso di file mancanti. Il placeholder viene usato per i dati opzionali mancanti, ma per quelli obbligatori si scarta proprio il campione per evitare dati incompleti nel training. 

Scorriamo quindi i casi e le modalità finché trova il primo file valido esistente. Leggiamo l’immagine con SimpleITK (sitk.ReadImage) e la convertiamo in array NumPy (sitk.GetArrayFromImage). Se non è una segmentazione ('seg'), normalizza l’immagine con la funzione norm_fn. Se necessario aggiungiamo una dimensione canale (np.expand_dims) — perché PyTorch lavora con tensori 4D [canale, z, y, x]. Salviamo la forma di questo array in self.placeholder_shape. Se non trova alcuna immagine valida (caso molto raro o dati assenti), imposta uno shape di default (1, 128, 128, 128) per i placeholder. 
Il loader prende il placeholder di forma (1, 182, 218, 182), lo usa se manca un’immagine.
Quando poi crea un batch con batch_size=8, questi placeholder (o i dati veri, se presenti) vengono “stackati” lungo la prima dimensione (batch), generando (8, 1, 182, 218, 182).
- Le dimensioni (1, 182, 218, 182) sono per la singola immagine
- Le dimensioni (8, 1, 182, 218, 182) sono per un batch di 8 immagini


def custom_collate_fn(batch):
Prendiamo un batch di campioni (ognuno un dizionario) e lo trasformiamo in un dizionario dove: - le immagini per ogni modalità sono un unico tensore (batch stacked), le maschere di validità per ogni modalità sono tensori booleani, gli altri dati sono liste con i valori di ogni campione.


Questioni importanti da valutare:

- Gestione valori mancanti:
      - trasformazione in tensori nulli?
      - eliminiamo i campioni che non hanno tutte le modalità obbligatorie? 
               - questa opzione scelta ma con gestione delle modalità opzionali, attraverso sostituzione dei file opzionali mancanti con array di zeri.

       
- Organizzazione del dataset:
      - Ogni paziente può presentare più di una cartella con scanner presi in momenti diversi. Come lo gestiamo?
          - ! fare in modo che nel batch non finiscano pazienti con stesso ID


Obiettivo del progetto: 

- segmentazione -> La “mappa” 3D delle label è un volume in cui ogni voxel ha un numero che indica a quale classe appartiene (0,1,2,3,4). 
Nel training diamo al modello le immagini 3D e il modello deve imparare a predire per ogni voxel l’etichetta corretta, cioè la label del file seg.
Il dataset di validazione non ha ground truth labels.andranno generati nuovi file, con la segmentazione ultimata.File che nel dataset di training vengono forniti. Questi saranno testati dai giudici. 

dovremo quindi generare nuovi file immagine a partire dai dati di validazione senza groud truth che seguono regole specifiche:

! N.B.!
"The submission system will expect your segmentation files to use the following format/naming conventions:

       - All individual segmentations must be NIfTI and use the .nii.gz file extension

       - Submitted data files must precisely match the spatial characteristics of their corresponding images. Specifically, the array dimensions, voxel spacing, image origin, and spatial orientation of the submitted files must be identical with those of the corresponding source images. You may use CaPTk to verify and/or visualize this.

       - This spatial consistency is especially crucial for the Meningioma Radiotherapy (RT) and Metastasis tasks. Unlike other datasets that are registered to an atlas like MNI152 or SRI24, the data for these two tasks are provided in their native space. This native representation inherently includes variability in array size and spatial resolution

       - Discrepancies in array size and/or image spacing may lead to scoring issues and invalidation of submissions.

       - Filenames must end with the 5-digit case ID, followed by a dash, then by the 3-digit timepoint -- the case IDs and timepoints are provided by the input filenames. For the Meningioma Radiotherapy Challenge, the filename format is a 4-digit case ID, followed by a dash, then the 1-digit timepoint." 



File path, per runnare da terminale
python "C:/Users/SANTE ACER/OneDrive/Documenti/Training_data_loader.py" --data_dir "G:\Il mio Drive\BraTS-GLI-PRE-TrainingData" --output_csv "G:\Il mio Drive\catalog_BraTS-GLI-PRE.csv" --batch_size 8

python "C:/Users/SANTE ACER/OneDrive/Documenti/Training_data_loader.py" --data_dir "G:\Il mio Drive\BraTS-GLI-POST-TrainingData\post_training_data" --output_csv "G:\Il mio Drive\catalog_BraTS-GLI-POST.csv" --batch_size 8
    
