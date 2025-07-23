import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import SimpleITK as sitk
import numpy as np
import pandas as pd
import logging
import plotly.graph_objects as go
import webbrowser
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

modalities = {
    'seg': 'seg',
    't1ce': 't1c',
    't1': 't1n',
    'flair': 't2f',
    't2': 't2w'
}


def create_catalog_csv(data_path, modalities, csv_output_path):
    cases = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if os.path.exists(csv_output_path):
        logging.info(f"Catalogo CSV già presente: {csv_output_path}")
        return

    with open(csv_output_path, mode='w', newline='') as csv_file:
        fieldnames = ['case', 'subject_id', 'timepoint', 'treatment_status'] + list(modalities.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases:
            parts = case.split('-')
            if len(parts) < 4:
                logging.warning(f"Formato nome cartella inatteso: {case}, salto")
                continue

            subject_id = parts[2]
            timepoint = parts[3]
            treatment_status = 'pre' if timepoint.startswith('0') else 'post'

            row = {
                'case': case,
                'subject_id': subject_id,
                'timepoint': timepoint,
                'treatment_status': treatment_status
            }

            case_path = os.path.join(data_path, case)
            for modality, suffix in modalities.items():
                filename = f"{case}-{suffix}.nii.gz"
                filepath = os.path.join(case_path, filename)
                row[modality] = filepath if os.path.exists(filepath) else ""

            writer.writerow(row)

    logging.info(f"Catalogo CSV creato in: {csv_output_path}")

def check_missing_files(csv_path, modalities):
    df = pd.read_csv(csv_path)
    missing_files = {}

    for idx, row in df.iterrows():
        case_missing = []
        for modality in modalities:
            file_path = row.get(modality, "")
            if not file_path or pd.isna(file_path) or not os.path.exists(file_path):
                case_missing.append(modality)
        if case_missing:
            missing_files[row['case']] = case_missing
    return missing_files

def normalize_minmax(img_array, mask_zero=True):
    mask = img_array > 0 if mask_zero else np.ones_like(img_array, dtype=bool)
    if np.any(mask):
        img_norm = np.zeros_like(img_array, dtype=np.float32)
        img_norm[mask] = (img_array[mask] - img_array[mask].min()) / (img_array[mask].max() - img_array[mask].min())
        return img_norm
    return img_array.astype(np.float32)

class BraTSDataset(Dataset):
    def __init__(self, csv_path, modalities=['t1ce', 't1', 'flair', 't2', 'seg'], 
                 required_modalities=None, transform=None, drop_missing=True, 
                 norm_fn=normalize_minmax, add_channel_dim=True):
        self.df = pd.read_csv(csv_path)
        self.modalities = modalities
        self.required_modalities = required_modalities if required_modalities is not None else modalities
        self.transform = transform
        self.drop_missing = drop_missing
        self.norm_fn = norm_fn
        self.add_channel_dim = add_channel_dim
        self._check_and_filter_missing_files()

        self.placeholder_shape = None
        for idx, row in self.df.iterrows():
            for mod in self.modalities:
                filepath = row.get(mod, "")
                if filepath and not pd.isna(filepath) and os.path.exists(filepath):
                    img = sitk.ReadImage(filepath)
                    arr = sitk.GetArrayFromImage(img)
                    if mod != 'seg':
                        arr = self.norm_fn(arr)
                    if self.add_channel_dim:
                        arr = np.expand_dims(arr, axis=0)
                    self.placeholder_shape = arr.shape
                    break
            if self.placeholder_shape is not None:
                break
        if self.placeholder_shape is None:
            self.placeholder_shape = (1, 128, 128, 128)
        logging.info(f"Placeholder shape per dati mancanti: {self.placeholder_shape}")

    def _check_and_filter_missing_files(self):
        missing_count = {mod: 0 for mod in self.modalities if mod in self.df.columns}
        rows_to_drop = []

        for idx, row in self.df.iterrows():
            missing_required = False
            for mod in self.modalities:
                if mod not in self.df.columns:
                    continue
                if not row[mod] or pd.isna(row[mod]):
                    missing_count[mod] += 1
                    if mod in self.required_modalities:
                        missing_required = True
            if missing_required:
                rows_to_drop.append(idx)

        logging.info(f"File mancanti per modalità: {missing_count}")
        if self.drop_missing and rows_to_drop:
            logging.warning(f"Rimuovo {len(rows_to_drop)} campioni con file mancanti nelle modalità obbligatorie")
            self.df = self.df.drop(rows_to_drop).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = {}
        valid_mask = {}

        for modality in self.modalities:
            filepath = row.get(modality, "")
            if not filepath or pd.isna(filepath) or not os.path.exists(filepath):
                images[modality] = np.zeros(self.placeholder_shape, dtype=np.float32)
                valid_mask[modality] = False
                continue

            img = sitk.ReadImage(filepath)
            img_array = sitk.GetArrayFromImage(img)
            if modality != 'seg':
                img_array = self.norm_fn(img_array)
            else:
                img_array = img_array.astype(np.uint8)

            if self.add_channel_dim:
                img_array = np.expand_dims(img_array, axis=0)

            images[modality] = img_array
            valid_mask[modality] = True

        sample = {
            'subject_id': row['subject_id'],
            'timepoint': row['timepoint'],
            'treatment_status': row['treatment_status'],
            'images': images,
            'valid_mask': valid_mask
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

def custom_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        if key == 'images':
            images = {mod: [] for mod in batch[0]['images'].keys()}
            for sample in batch:
                for mod, img in sample['images'].items():
                    images[mod].append(torch.from_numpy(img))
            collated['images'] = {mod: torch.stack(tensors) for mod, tensors in images.items()}
        elif key == 'valid_mask':
            masks = {mod: [] for mod in batch[0]['valid_mask'].keys()}
            for sample in batch:
                for mod, valid in sample['valid_mask'].items():
                    masks[mod].append(valid)
            collated['valid_mask'] = {mod: torch.tensor(vals, dtype=torch.bool) for mod, vals in masks.items()}
        else:
            collated[key] = [sample[key] for sample in batch]
    return collated

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory dati (es. G:\\Il mio Drive\\BraTS-GLI-PRE-TrainingData)')
    parser.add_argument('--output_csv', type=str, default='catalog.csv', help='Output CSV path')
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    create_catalog_csv(args.data_dir, modalities, args.output_csv)

    mods = ['t1ce', 't1', 'flair', 't2', 'seg']
    missing = check_missing_files(args.output_csv, mods)

    if missing:
        logging.warning("Sono stati trovati i seguenti file mancanti:")
        for case, mods_missing in missing.items():
            logging.warning(f" - Caso {case}: modalità mancanti: {', '.join(mods_missing)}")
    else:
        logging.info("Tutti i file richiesti sono presenti.")
        
    required = mods

    dataset = BraTSDataset(args.output_csv, modalities=mods, required_modalities=required)
    logging.info(f"Numero totale campioni: {len(dataset)}")

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    logging.info(f"Campioni train: {len(train_dataset)}")
    logging.info(f"Campioni val: {len(val_dataset)}")



    for batch in train_loader:
        logging.info(f"Batch train subject_id: {batch['subject_id'][0]}")
        break