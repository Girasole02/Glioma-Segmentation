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
matplotlib.use('TkAgg')  # oppure 'Qt5Agg' se PyQt5 installato
import matplotlib.pyplot as plt

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
        fieldnames = ['subject_id', 'timepoint', 'treatment_status'] + list(modalities.keys())
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_MINIMAL,  # << usa virgolette solo se servono
            delimiter=';'                # << usa ; come separatore di colonna
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

def normalize_minmax(img_array, mask_zero=True):
    mask = img_array > 0 if mask_zero else np.ones_like(img_array, dtype=bool)
    if np.any(mask):
        img_norm = np.zeros_like(img_array, dtype=np.float32)
        img_norm[mask] = (img_array[mask] - img_array[mask].min()) / (img_array[mask].max() - img_array[mask].min())
        return img_norm
    return img_array.astype(np.float32)

class BraTSDataset(Dataset):
    def __init__(self, csv_path, modalities=None, required_modalities=None, drop_missing=True,
                 norm_fn=normalize_minmax, add_channel_dim=True, placeholder_shape=(155, 240, 240),
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
            'subject_id': row.get('subject_id'),
            'timepoint': row.get('timepoint'),
            'treatment_status': row.get('treatment_status'),
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

def apri_file_immagine(path):
    path = os.path.abspath(path) # Converte il percorso relativo in un percorso assoluto
    try:
        if os.name == 'nt':  # Windows
            os.startfile(path) # Apre il file con l'applicazione predefinita (es: Visualizzatore Foto)
        else:
            webbrowser.open(f'file://{path}')
    except Exception as e:
        logging.error(f"Errore apertura immagine: {e}")

def visualize_segmentation_only(seg_tensor, slice_idx=None):
    """
    Visualizza una slice 2D della segmentazione con colori distinti per ogni label.

    Args:
        seg_tensor (torch.Tensor o np.array): tensor o array numpy della segmentazione,
                                              shape [1, D, H, W] o [D, H, W].
        slice_idx (int): indice della slice lungo l’asse D (se None usa la slice centrale).
    """
    if isinstance(seg_tensor, torch.Tensor):
        seg_tensor = seg_tensor.numpy()

    if seg_tensor.ndim == 4:
        seg_tensor = seg_tensor[0]

    if slice_idx is None:
        slice_idx = seg_tensor.shape[0] // 2  # slice centrale

    seg_slice = seg_tensor[slice_idx, :, :]

    plt.figure(figsize=(6, 6))
    cmap = plt.get_cmap('tab10', 5)  # Colormap con 5 colori per le 5 label (0-4)
    plt.imshow(seg_slice, cmap=cmap, vmin=0, vmax=4)
    cbar = plt.colorbar(ticks=range(5))
    cbar.ax.set_yticklabels(['Background', 'NETC', 'SNFH', 'ET', 'RC'])
    plt.title(f"Segmentazione (slice {slice_idx})")
    plt.axis('off')
    plt.show()


def visualize_sample(input_tensor, seg_tensor, slice_idx=None, modality_idx=1):
    """
    Visualizza una slice 2D di un volume 3D con overlay della segmentazione.

    Args:
        input_tensor (torch.Tensor): shape [4, D, H, W]
        seg_tensor (torch.Tensor): shape [1, D, H, W]
        slice_idx (int): indice della slice lungo l'asse D
        modality_idx (int): indice della modalità da usare per lo sfondo (0=t1, 1=t1ce, 2=t2, 3=flair)
    """
    input_tensor = input_tensor.numpy()
    seg_tensor = seg_tensor.numpy()

    if slice_idx is None:
        slice_idx = input_tensor.shape[1] // 2  # asse D


    image_slice = input_tensor[modality_idx, slice_idx, :, :]
    seg_slice = seg_tensor[0, slice_idx, :, :]

    cmap_seg = plt.get_cmap('jet')

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

    plt.tight_layout()
    plt.show()

"""
def statistica_presenza_labels(dataset):
    label_names = {1: 'NETC', 2: 'SNFH', 3: 'ET', 4: 'RC'}
    count_dict = defaultdict(int)

    for i in range(len(dataset)):
        sample = dataset[i]
        seg = sample['images']['seg']
        labels_present = set(np.unique(seg))
        for lbl in labels_present:
            if lbl in label_names:
                count_dict[label_names[lbl]] += 1

    for label, count in count_dict.items():
        logging.info(f"Campioni con label {label}: {count}")
"""
def statistica_presenza_labels(dataset):
    label_names = {1: 'NETC', 2: 'SNFH', 3: 'ET', 4: 'RC'}
    presenza_label = {label: 0 for label in label_names}

    for sample in dataset:
        seg = sample['images']['seg']
        if isinstance(seg, torch.Tensor):
            seg = seg.numpy()
        if seg.ndim == 4:
            seg = seg[0]  # [D, H, W]
        labels_presenti = np.unique(seg)
        for label in label_names:
            if label in labels_presenti:
                presenza_label[label] += 1

    n_samples = len(dataset)
    stats = []

    print(f"\n{'Label':<5} {'Nome':<6} {'Presenza':>9} {'Percentuale':>15}")
    print("-" * 40)
    for label, count in presenza_label.items():
        percentuale = 100 * count / n_samples
        nome = label_names[label]
        stats.append((nome, count, percentuale))
        print(f"{label:<5} {nome:<6} {count:>9} {percentuale:>14.1f}%")

    labels = [s[0] for s in stats]
    counts = [s[1] for s in stats]
    percentages = [s[2] for s in stats]
    colors = ['yellow', 'green', 'red', 'blue']

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
""""
def analisi_voxel_segmentazioni(dataset):
    total_voxels = defaultdict(int)
    total_positive = defaultdict(int)

    for i in range(len(dataset)):
        sample = dataset[i]
        seg = sample['images']['seg']

        for label in [1, 2, 3, 4]:
            voxels_label = (seg == label).sum()
            total_positive[label] += voxels_label
            total_voxels[label] += np.prod(seg.shape)

    for label in [1, 2, 3, 4]:
        percent = (total_positive[label] / total_voxels[label] * 100) if total_voxels[label] > 0 else 0
        logging.info(f"Label {label} voxel positivi: {total_positive[label]}, Percentuale sul totale: {percent:.4f}%")
"""
def analisi_voxel_segmentazioni(dataset):
    label_names = {1: 'NETC', 2: 'SNFH', 3: 'ET', 4: 'RC'}
    voxel_stats = defaultdict(list)
    for sample in dataset:
        seg = sample['images']['seg'][0]  # [D,H,W]
        labels, counts = np.unique(seg, return_counts=True)
        for label, count in zip(labels, counts):
            if label in label_names:
                voxel_stats[label_names[label]].append(count)
    plt.figure(figsize=(10, 6))
    labels = list(voxel_stats.keys())
    data = [voxel_stats[label] for label in labels]
    plt.boxplot(data, labels=labels)
    plt.title("Distribuzione voxel per etichetta tumorale")
    plt.ylabel("Numero di voxel")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n STATISTICHE SEGMENTAZIONE")
    for label in labels:
        valori = voxel_stats[label]
        print(f"{label}: media={np.mean(valori):.2f}, max={np.max(valori)}, min={np.min(valori)}, std={np.std(valori):.2f}")
"""
# ---- MODELLO UNET 3D ----
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear=True):
        super(Up, self).__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=5):
        super(UNet3D, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# Funzione per calcolare il Dice score medio per ogni classe
def dice_score(preds, targets, num_classes=5, epsilon=1e-6):
    dices = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        intersection = (pred_cls * target_cls).sum(dim=(1, 2, 3))
        union = pred_cls.sum(dim=(1, 2, 3)) + target_cls.sum(dim=(1, 2, 3))
        dice = (2 * intersection + epsilon) / (union + epsilon)
        dices.append(dice.mean().item())
    return dices
"""
if __name__ == "__main__":
    data_dir = r"G:\Il mio Drive\BraTS-GLI-POST-TrainingData\post_training_data"
    csv_file = "catalog_brats_pre.csv"

    create_catalog_csv(data_dir, modalities, csv_file)

    missing = check_missing_files(csv_file, modalities)
    if missing:
        logging.warning(f"Campioni con file mancanti: {missing}")

    dataset = BraTSDataset(csv_file, modalities=list(modalities.keys()), drop_missing=True)

    # Subset per visualizzazioni/statistiche
    subset_size = 1
    subset_indices = list(range(min(subset_size, len(dataset))))
    subset = Subset(dataset, subset_indices)

    logging.info(f"Dataset caricato con {len(dataset)} campioni totali")
    logging.info(f"Lavoriamo su un subset di {len(subset)} campioni")

    sample = subset[0]
    print(f"Sample: subject_id={sample['subject_id']}, timepoint={sample['timepoint']}, treatment={sample['treatment_status']}")
    imgs = [sample['images'][m] if sample['images'][m].ndim == 4 else np.expand_dims(sample['images'][m], axis=0)
        for m in ['t1', 't1ce', 't2', 'flair']]
    input_tensor = torch.tensor(np.vstack(imgs))
    visualize_sample(input_tensor, torch.tensor(sample['images']['seg']))


    visualize_segmentation_only(torch.tensor(sample['images']['seg']))

    statistica_presenza_labels(subset)
    analisi_voxel_segmentazioni(subset)
"""
    # ------- TRAINING E VALIDAZIONE -------

    # Split 80% train, 20% val
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    logging.info(f"Uso device: {device}")

    model = UNet3D(in_channels=4, out_channels=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 5
    logging.info("Inizio training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = torch.cat([batch['images'][mod].to(device) for mod in ['t1', 't1ce', 't2', 'flair']], dim=1)
            target = batch['images']['seg'].squeeze(1).to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = torch.cat([batch['images'][mod].to(device) for mod in ['t1', 't1ce', 't2', 'flair']], dim=1)
                target = batch['images']['seg'].squeeze(1).to(device).long()
                
                outputs = model(inputs)  # <-- Calcolo il forward qui
                loss = criterion(outputs, target)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())

 
        avg_val_loss = val_loss / len(val_loader)

        # Concatenamento di tutte le predizioni e target per calcolare il Dice
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        dice_per_class = dice_score(all_preds_tensor, all_targets_tensor, num_classes=5)

        # Logging
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        logging.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        for cls, dice_val in enumerate(dice_per_class):
            logging.info(f" - Dice Class {cls}: {dice_val:.4f}")
"""

