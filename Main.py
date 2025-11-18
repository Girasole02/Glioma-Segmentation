import os, logging, pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from Config import Config
from Utils import pesi_classi, catalogo_csv 

from Dataset import BraTS3DDataset
from Dataset import BraTS3DPatchDataset

from Model import UNet3D

from  Training import train_e_validazione
from Ris import plot_risultati

from Check import check_dimensioni_uguali, check_file_mancanti
from Stats import statistica_presenza_labels, analisi_voxel_segmentazioni
from Visualizzazioni import visualizza_modalita_e_segmentazione, visualizzazione_segmentazione, visualizzazione_sample, visualizza_tre_assi

if __name__ == "__main__":

    os.makedirs(Config.save_dir, exist_ok=True)
    log_file = os.path.join(Config.save_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("\n===== INIZIO SCRIPT =====")
    logging.info(f"{datetime.now():%Y-%m-%d %H:%M:%S}- Inizio Script")
    logging.info(f"Cartella log: {log_file}")
    logging.info(f"Device: {Config.device}")

    print("\n===== INIZIALIZZAZIONE =====")
    logging.info(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Inizio Iniziallizzazione")

    catalogo_csv(Config.data_dir, Config.modalities, Config.csv_file)
    df = pd.read_csv(Config.csv_file, sep=';')
    logging.info(f"Campioni totali nel CSV: {len(df)}")
    logging.info(f"Campioni validi dopo check: {len(df)}")

    Config.subset_size = len(df) if Config.subset_size is None else Config.subset_size
    df_subset = df.head(Config.subset_size).copy()
    logging.info(f"Utilizzo {'Dataset Intero' if Config.subset_size == len(df) else 'Subset'}: {Config.subset_size} campioni")

    
    print("\n===== CREAZIONE DATASET =====")
    logging.info(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Inizio Creazione Dataset")

    patient_ids = df_subset['subject_id'].unique()
    print(f"Pazienti unici nel subset: {len(patient_ids)}")
    
    train_patients, val_patients = train_test_split(
        patient_ids, 
        test_size=0.2, 
        random_state=42
    )

    train_df = df_subset[df_subset['subject_id'].isin(train_patients)]
    val_df = df_subset[df_subset['subject_id'].isin(val_patients)]
    
    patients_csv = "patients_subset.csv"
    train_csv = "train_subset.csv"
    val_csv = "val_subset.csv"
    df_subset.to_csv(patients_csv, sep=';', index=False)
    train_df.to_csv(train_csv, sep=';', index=False)
    val_df.to_csv(val_csv, sep=';', index=False)
    

    dataset = BraTS3DDataset(patients_csv, Config.modalities)
    train_dataset = BraTS3DPatchDataset(train_csv, Config.augment, Config.modalities)
    val_dataset = BraTS3DDataset(val_csv, Config.modalities)
    
    logging.info(f"Training dataset: {len(train_dataset)} campioni (patches augmentate)")
    logging.info(f"Validation dataset: {len(val_dataset)} campioni (volumi interi)")

    """
    sample = dataset[0]

    print("\n===== ANALISI STATISTICHE =====")

    check_dimensioni_uguali(dataset)
    statistica_presenza_labels(dataset, Config.label_names)
    analisi_voxel_segmentazioni(dataset, Config.label_names, Config.save_dir)
    
    print("\n===== VISUALIZZAZIONI =====")
    if len(dataset) > 0:
        visualizza_modalita_e_segmentazione(Config.save_dir, sample)
        visualizzazione_sample(Config.save_dir, sample)
        visualizzazione_segmentazione(Config.save_dir, sample)
        visualizza_tre_assi(Config.save_dir, Config.modalities, sample, modality='t1ce')
    """
    print("\n===== PREPARAZIONE TRAINING =====")
    logging.info(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Inizio Preparazione al Training")

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    logging.info(f"Train loader: {len(train_loader)} batch")
    logging.info(f"Val loader: {len(val_loader)} campioni")
    
    print("\n===== PESI CLASSI =====")
    logging.info(f"{datetime.now():%Y-%m-%d %H:%M:%S}- Inizio Calcolo Pesi")
    print(f"{'Classe':<10} {'Nome':<10} {'Conteggio':<12} {'Frequenza':<12} {'Peso':<10}")
    print("-" * 60)
    class_weights = pesi_classi(train_dataset, Config.label_names)
    logging.info(f"Pesi finali: {class_weights}")

    print("\n===== INIZIO TRAINING =====")
    logging.info(f"{datetime.now():%Y-%m-%d %H:%M:%S}- Inizio Training")

    model = UNet3D(Config.in_channels, Config.out_channels, Config.dropout_rate).to(Config.device)
    logging.info(f"Modello: {sum(p.numel() for p in model.parameters()):,} parametri")
    logging.info(f"Epoche: {Config.num_epochs}")
    logging.info(f"Early stopping patience: {Config.patience}")
    
    train_losses, val_losses, dice_metrics, hd95_metrics = train_e_validazione(  
         model, train_loader, val_loader, Config.device, Config.num_epochs, Config.patience, class_weights
    )

    print("\n===== RISULTATI FINALI =====")
    logging.info(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Inizio Calcolo e Plot Risultati")
    plot_risultati(Config.save_dir, train_losses, val_losses, dice_metrics, hd95_metrics)  
    
    if dice_metrics['ET']:
        logging.info(f"Dice ET: {dice_metrics['ET'][-1]:.4f}")
        logging.info(f"Dice WT: {dice_metrics['WT'][-1]:.4f}")
        logging.info(f"Dice TC: {dice_metrics['TC'][-1]:.4f}")
        
    if hd95_metrics['ET']:  
        logging.info(f"HD95 ET: {hd95_metrics['ET'][-1]:.4f}")
        logging.info(f"HD95 WT: {hd95_metrics['WT'][-1]:.4f}")
        logging.info(f"HD95 TC: {hd95_metrics['TC'][-1]:.4f}")
    
    try:
        os.remove(train_csv)
        os.remove(val_csv)
        print(f"File temporanei rimossi: {train_csv}, {val_csv}")
    except:
        pass
    logging.info(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Fine Training")
    logging.info("\n===== TRAINING COMPLETATO =====")