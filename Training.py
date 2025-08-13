# IMPORT LIBRERIE
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from collections import defaultdict
import logging
import webbrowser
import SimpleITK as sitk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# MAPPATURA MODALITA'
modalities = {
    'seg': 'seg',
    't1ce': 't1c',
    't1': 't1n',
    'flair': 't2f',
    't2': 't2w'
}

# CREAZIONE DEL CATALOGO
def create_catalog_csv(data_path, modalities, csv_output_path):
    cases = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if os.path.exists(csv_output_path):
        logging.info(f"Catalogo CSV già presente: {csv_output_path}")
        return
    with open(csv_output_path, mode='w', newline='') as csv_file:
        fieldnames = ['subject_id', 'timepoint', 'treatment_status'] + list(modalities.keys())
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_MINIMAL,
            delimiter=';'
        )
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
                'subject_id': subject_id,
                'timepoint': timepoint,
                'treatment_status': treatment_status
            }

            case_path = os.path.join(data_path, case)
            for modality, suffix in modalities.items():
                filename = f"{case}-{suffix}.nii.gz"
                filepath = os.path.join(case_path, filename)
                row[modality] = filepath if os.path.exists(filepath) else ""

            print(row)
            writer.writerow(row)

    logging.info(f"Catalogo CSV creato in: {csv_output_path}")

# CHECK FILE MANCANTI
def check_missing_files(csv_path, modalities):
    df = pd.read_csv(csv_path, sep=';')
    print("Colonne trovate nel CSV:", df.columns.tolist())
    missing_files = {}
    for idx, row in df.iterrows():
        case_missing = []
        for modality in modalities:
            file_path = row.get(modality, "")
            if not file_path or pd.isna(file_path) or not os.path.exists(file_path):
                case_missing.append(modality)
        if case_missing:
            missing_files[f"{row['subject_id']}_{row['timepoint']}"] = case_missing
    return missing_files

# ANALISI DELLE INTENSITA' 
def analizza_intensità_e_plot(dataset, modalità='t1', max_cases=5, bins=100):
    plt.figure(figsize=(15, 4))

    for i in range(min(max_cases, len(dataset))):
        img = dataset[i]['images'][modalità]
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img = img[img > 0]  

        min_val, max_val = img.min(), img.max()
        mean_val, std_val = img.mean(), img.std()
        print(f"[{i}] {modalità.upper()} → min: {min_val:.2f}, max: {max_val:.2f}, mean: {mean_val:.2f}, std: {std_val:.2f}")

        plt.subplot(1, max_cases, i + 1)
        plt.hist(img.flatten(), bins=bins, color='skyblue', edgecolor='black')
        plt.title(f"Sample {i}")
        plt.xlabel("Intensità")
        plt.ylabel("Voxel")
        plt.grid(True)

    plt.suptitle(f"Istogrammi intensità - Modalità: {modalità}", fontsize=14)
    plt.tight_layout()
    plt.show()

# NORMALIZZAZIONE MINMAX
def normalize_minmax(img_array, mask_zero=True):
    mask = img_array > 0 if mask_zero else np.ones_like(img_array, dtype=bool)
    if np.any(mask):
        img_norm = np.zeros_like(img_array, dtype=np.float32)
        img_norm[mask] = (img_array[mask] - img_array[mask].min()) / (img_array[mask].max() - img_array[mask].min())
        return img_norm
    return img_array.astype(np.float32)

# DEFINIZIONE DATASET
class BraTSDataset(Dataset):
    def __init__(self, csv_path, modalities=None, required_modalities=None, drop_missing=True,
                 norm_fn=normalize_minmax, add_channel_dim=True, placeholder_shape=(182, 218, 182),
                 transform=None):
        self.csv_path = csv_path
        self.modalities = modalities or ['t1', 't1ce', 't2', 'flair', 'seg']
        self.required_modalities = required_modalities or self.modalities
        self.drop_missing = drop_missing
        self.norm_fn = norm_fn
        self.add_channel_dim = add_channel_dim
        self.placeholder_shape = placeholder_shape
        self.transform = transform
        self.df = pd.read_csv(csv_path, sep=';')
        self._check_and_filter_missing_files()

    # ANALISI E FILTRAGGIO FILE MANCANTI

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

            if modality != 'seg' and self.norm_fn is not None:
                img_array = self.norm_fn(img_array)
            elif modality == 'seg':
                img_array = img_array.astype(np.uint8)

            if self.add_channel_dim:
                img_array = np.expand_dims(img_array, axis=0)

            images[modality] = img_array
            valid_mask[modality] = True

        sample = {
            'subject_id': row.get('subject_id'),
            'timepoint': row.get('timepoint'),
            'treatment_status': row.get('treatment_status'),
            'images': images,
            'valid_mask': valid_mask
        }

        if self.transform:
            sample = self.transform(sample)
        return sample
    
# COSTUME COLLATE PER PYTORCH
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

# APERTURA IMMAGINI
def apri_file_immagine(path):
    path = os.path.abspath(path)
    try:
        if os.name == 'nt':
            os.startfile(path)
        else:
            webbrowser.open(f'file://{path}')
    except Exception as e:
        logging.error(f"Errore apertura immagine: {e}")

# VISUALIZZAZIONE SEGMENTAZIONE
def visualize_segmentation_only(seg_tensor, slice_idx=None):
    if isinstance(seg_tensor, torch.Tensor):
        seg_tensor = seg_tensor.numpy()

    if seg_tensor.ndim == 4:
        seg_tensor = seg_tensor[0]

    if slice_idx is None:
        slice_idx = seg_tensor.shape[0] // 2 

    seg_slice = seg_tensor[slice_idx, :, :]

    plt.figure(figsize=(6, 6))
    cmap = plt.get_cmap('tab10', 5)
    plt.imshow(seg_slice, cmap=cmap, vmin=0, vmax=4)
    cbar = plt.colorbar(ticks=range(5))
    cbar.ax.set_yticklabels(['Background', 'NETC', 'SNFH', 'ET', 'RC'])
    plt.title(f"Segmentazione (slice {slice_idx})")
    plt.axis('off')
    plt.show()

# VISUALIZZAZIONE IMMAGINE SINGOLO SAMPLE
def visualize_sample(input_tensor, seg_tensor, slice_idx=None, modality_idx=1):
    assert input_tensor.ndim == 4 and input_tensor.shape[0] == 4, f"Input tensor non valido: {input_tensor.shape}"
    assert seg_tensor.ndim in [3, 4], f"Segmentazione ha forma non valida: {seg_tensor.shape}"

    input_tensor = input_tensor.numpy()
    seg_tensor = seg_tensor.numpy()

    if slice_idx is None:
        slice_idx = input_tensor.shape[1] // 2

    image_slice = input_tensor[modality_idx, slice_idx, :, :]
    seg_slice = seg_tensor[0, slice_idx, :, :]

    cmap_seg = plt.get_cmap('jet')

    label_names = {1: 'NETC', 2: 'SNFH', 3: 'ET', 4: 'RC'}
    colors = ['yellow', 'green', 'red', 'blue']

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_slice, cmap='gray')
    plt.title(f"Modality index {modality_idx} (slice {slice_idx})")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_slice, cmap='gray')
    plt.imshow(seg_slice, cmap=cmap_seg, alpha=0.5)
    plt.title("Overlay con segmentazione")
    plt.axis('off')

    legend_patches = [Patch(color=color, label=label_names[i]) for i, color in zip(label_names.keys(), colors)]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.show()

# ANALISI STATISTICA LABELS
def statistica_presenza_labels(dataset):
    label_names = {1: 'NETC', 2: 'SNFH', 3: 'ET', 4: 'RC'}
    presenza_label = {label: 0 for label in label_names}

    for sample in dataset:
        seg = sample['images']['seg']
        if isinstance(seg, torch.Tensor):
            seg = seg.numpy()
        if seg.ndim == 4:
            seg = seg[0]
        labels_presenti = np.unique(seg)
        for label in label_names:
            if label in labels_presenti:
                presenza_label[label] += 1

    n_samples = len(dataset)
    stats = []

    print(f"\n{'Label':<6} {'Nome':<6} {'Presenza':>9} {'Percentuale':>15}")
    print("-" * 40)
    for label, count in presenza_label.items():
        percentuale = 100 * count / n_samples
        nome = label_names[label]
        stats.append((nome, count, percentuale))
        print(f"{label:<6} {nome:<6} {count:>9} {percentuale:>14.4f}%")

    labels = [s[0] for s in stats]
    counts = [s[1] for s in stats]
    percentages = [s[2] for s in stats]
    colors = ['skyblue']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, counts, color=colors)
    for bar, pct in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5, f"{pct:.1f}%", ha='center')

    plt.title("Presenza delle etichette tumorali nei campioni")
    plt.ylabel("Numero di campioni")
    plt.ylim(0, max(counts) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ANALISI VOXEL SEGMENTAZIONE
def analisi_voxel_segmentazioni(dataset):
    label_names = {1: 'NETC', 2: 'SNFH', 3: 'ET', 4: 'RC'}

    total_voxels = 0
    total_positive = defaultdict(int)

    voxel_stats = defaultdict(list)
    for sample in dataset:
        seg = sample['images']['seg']
        if total_voxels == 0:
            total_voxels = np.prod(seg.shape)

        labels, counts = np.unique(seg, return_counts=True)
        counts_dict = dict(zip(labels, counts))

        for label in label_names:
            count_label = counts_dict.get(label, 0)
            total_positive[label] += count_label
            voxel_stats[label_names[label]].append(count_label)

    print("\n--- Statistica Globale ---")
    for label in label_names:
        percent = (total_positive[label] / (total_voxels * len(dataset)) * 100) if total_voxels > 0 else 0
        logging.info(f"Label {label} voxel positivi: {total_positive[label]}, Percentuale sul totale: {percent:.4f}%")
        print(f"Label {label} ({label_names[label]}): {total_positive[label]} voxel positivi, {percent:.4f}% sul totale")

    print("\n--- Statistiche per campione ---")
    labels = list(voxel_stats.keys())
    data = [voxel_stats[label] for label in labels]

    for label in labels:
        valori = voxel_stats[label]
        print(f"{label}: media={np.mean(valori):.2f}, max={np.max(valori)}, min={np.min(valori)}, std={np.std(valori):.2f}")

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=labels)
    plt.title("Distribuzione voxel per etichetta tumorale (campione per campione)")
    plt.ylabel("Numero di voxel")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# SET-UP PER L'ADDESTRAMENTO: MODELLO 2D (UNET)
def cat_with_resize(tensor1, tensor2):
    """
    Concateniamo tensor2 con tensor1 lungo la dimensione dei canali,
    ridimensionando tensor2 se necessario per far combaciare altezza e larghezza.
    """
    if tensor1.shape[2:] != tensor2.shape[2:]:
        tensor2 = F.interpolate(tensor2, size=tensor1.shape[2:], mode='bilinear', align_corners=False)
    return torch.cat((tensor1, tensor2), dim=1)

# DEFINIZIONE DATASET 2D: ESTRAZIONE SLICES
class BraTS2DDataset(Dataset):
    def __init__(self, brats_3d_dataset, slice_axis=0):
        self.original_dataset = brats_3d_dataset
        self.slice_axis = slice_axis
        self.samples = []

        for vol_idx in range(len(brats_3d_dataset)):
            seg = brats_3d_dataset[vol_idx]['images']['seg']
            for slice_idx in range(seg.shape[self.slice_axis+1]):
                if np.any(seg[0].take(indices=slice_idx, axis=self.slice_axis) > 0):
                    self.samples.append((vol_idx, slice_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.samples[idx]
        sample = self.original_dataset[vol_idx]
        
        images = {}
        for mod, img in sample['images'].items():
            if img.ndim == 4:
                slice_2d = img[0].take(indices=slice_idx, axis=self.slice_axis)
            else:
                slice_2d = img.take(indices=slice_idx, axis=self.slice_axis)
            images[mod] = np.expand_dims(slice_2d, axis=0)
        
        return {
            'subject_id': sample['subject_id'],
            'slice_idx': slice_idx,
            'images': images,
            'valid_mask': sample['valid_mask']
        }

# DEFINZIONE MODELLO UNET2D
class UNet2D(nn.Module):
    def __init__(self, in_channels=4, out_channels=5):
        super().__init__()
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.bottleneck = self._block(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))

        up3 = self.up3(bottleneck)
        dec3 = self.dec3(cat_with_resize(up3, enc3))

        up2 = self.up2(dec3)
        dec2 = self.dec2(cat_with_resize(up2, enc2))

        up1 = self.up1(dec2)
        dec1 = self.dec1(cat_with_resize(up1, enc1))

        return self.final(dec1)

# RAPIDO CHECK PRE-ADDESTRAMENTO SU SHAPES E CANALI
def check_shapes_and_channels(dataset, modalities, sample_size=10):
    shape_report = {mod: set() for mod in modalities}
    channels_report = {mod: set() for mod in modalities}
    
    for i in range(min(sample_size, len(dataset))):
        sample = dataset[i]
        for mod in modalities:
            tensor = sample['images'][mod]
            shape_report[mod].add(tensor.shape)
            if tensor.ndim == 4:
                channels_report[mod].add(tensor.shape[0])
            elif tensor.ndim == 3:
                channels_report[mod].add(1)

    for mod in modalities:
        if len(shape_report[mod]) == 1:
            logging.info(f"[Check] Modalità '{mod}' ha shape uniforme: {next(iter(shape_report[mod]))}")
        else:
            logging.warning(f"[Check] Modalità '{mod}' ha shape variabile: {shape_report[mod]}")

        if len(channels_report[mod]) == 1:
            logging.info(f"[Check] Modalità '{mod}' ha numero canali uniforme: {next(iter(channels_report[mod]))}")
        else:
            logging.warning(f"[Check] Modalità '{mod}' ha numero canali variabile: {channels_report[mod]}")

# METRICA DI VALUTAZIONE: DICE SCORE 
def dice_score_binary(pred, target):
    intersection = np.logical_and(pred, target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)

def evaluate_dice_brats(model, dataloader, device):
    model.eval()
    et_scores, wt_scores, tc_scores = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = torch.cat([batch['images'][mod] for mod in ['t1', 't1ce', 't2', 'flair']], dim=1).to(device)
            targets = batch['images']['seg'].squeeze(1).long().cpu().numpy()

            outputs = model(inputs)
            outputs = F.interpolate(outputs, size=targets.shape[1:], mode='bilinear', align_corners=False)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for p, t in zip(preds, targets):
                # ET = label 3
                et_scores.append(dice_score_binary(p == 3, t == 3))
                # WT = labels {1,2,3,4}
                wt_scores.append(dice_score_binary(np.isin(p, [1,2,3,4]), np.isin(t, [1,2,3,4])))
                # TC = labels {1,3,4}
                tc_scores.append(dice_score_binary(np.isin(p, [1,3,4]), np.isin(t, [1,3,4])))

    return {
        'ET': np.mean(et_scores) if et_scores else 0,
        'WT': np.mean(wt_scores) if wt_scores else 0,
        'TC': np.mean(tc_scores) if tc_scores else 0
    }

# MAIN
if __name__ == "__main__":
    data_dir = r"G:\Il mio Drive\BraTS-GLI-PRE-TrainingData"
    csv_file = "catalog_brats_pre.csv"

    create_catalog_csv(data_dir, modalities, csv_file)

    missing = check_missing_files(csv_file, modalities)
    if missing:
        logging.warning(f"Campioni con file mancanti: {missing}")

    dataset = BraTSDataset(csv_file, modalities=list(modalities.keys()), drop_missing=False)

    check_shapes_and_channels(dataset, modalities=list(modalities.keys()), sample_size=5)

    subset_size = 5
    subset_indices = list(range(min(subset_size, len(dataset))))
    subset = Subset(dataset, subset_indices)

    logging.info(f"Dataset caricato con {len(dataset)} campioni totali")
    logging.info(f"Lavoriamo su un subset di {len(subset)} campioni")

    sample = subset[0]
    print(f"Sample: subject_id={sample['subject_id']}, timepoint={sample['timepoint']}, treatment={sample['treatment_status']}")
    imgs = [sample['images'][m] if sample['images'][m].ndim == 4 else np.expand_dims(sample['images'][m], axis=0)
        for m in ['t1', 't1ce', 't2', 'flair']]
    input_tensor = torch.tensor(np.vstack(imgs))

    analizza_intensità_e_plot(dataset, modalità='t1ce', max_cases=3)
    visualize_sample(input_tensor, torch.tensor(sample['images']['seg']))
    visualize_segmentation_only(torch.tensor(sample['images']['seg']))
    statistica_presenza_labels(subset)
    analisi_voxel_segmentazioni(subset)

    print("\n" + "="*50)
    print("AVVIO TRAINING 2D".center(50))
    print("="*50)

    brats_2d = BraTS2DDataset(subset, slice_axis=0)
    logging.info(f"Estratte {len(brats_2d)} slice 2D con tessuto tumorale")

    # SPLIT PER PAZIENTE !!!
    patient_ids = list({vol_idx for vol_idx, _ in brats_2d.samples})
    train_patients, val_patients = train_test_split(patient_ids, test_size=0.2, random_state=42)

    train_indices = [i for i, (vol_idx, _) in enumerate(brats_2d.samples) if vol_idx in train_patients]
    val_indices = [i for i, (vol_idx, _) in enumerate(brats_2d.samples) if vol_idx in val_patients]

    train_dataset = Subset(brats_2d, train_indices)
    val_dataset = Subset(brats_2d, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                               collate_fn=custom_collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                             collate_fn=custom_collate_fn, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet2D(in_channels=4, out_channels=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 3
    train_losses, val_losses = [], []
    dice_et, dice_wt, dice_tc = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = torch.cat([batch['images'][mod] for mod in ['t1', 't1ce', 't2', 'flair']], dim=1).to(device)
            targets = batch['images']['seg'].squeeze(1).long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = F.interpolate(outputs, size=targets.shape[1:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = torch.cat([batch['images'][mod] for mod in ['t1', 't1ce', 't2', 'flair']], dim=1).to(device)
                targets = batch['images']['seg'].squeeze(1).long().to(device)

                outputs = model(inputs)
                outputs = F.interpolate(outputs, size=targets.shape[1:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        metrics = evaluate_dice_brats(model, val_loader, device)
        dice_et.append(metrics['ET'])
        dice_wt.append(metrics['WT'])
        dice_tc.append(metrics['TC'])

        logging.info(f"Epoch [{epoch+1}/{num_epochs}] "
                     f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                     f"ET: {metrics['ET']:.4f} | WT: {metrics['WT']:.4f} | TC: {metrics['TC']:.4f}")
        
    print("Training completato!")

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Andamento Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(dice_et, label='ET')
    plt.plot(dice_wt, label='WT')
    plt.plot(dice_tc, label='TC')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Dice Score per classe BraTS')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
