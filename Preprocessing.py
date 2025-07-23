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
from matplotlib.patches import Patch

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
import numpy as np
import matplotlib.pyplot as plt

def analizza_intensità_e_plot(dataset, modalità='t1', max_cases=5, bins=100):
    """
    Analizza e plottare le intensità voxel (>0) per una modalità su più soggetti.

    Args:
        dataset: dataset BraTS con chiave 'images' per ogni campione
        modalità: nome della modalità da analizzare ('t1', 'flair', ecc.)
        max_cases: numero massimo di soggetti da analizzare
        bins: numero di bin per l'istogramma
    """
    plt.figure(figsize=(15, 4))

    for i in range(min(max_cases, len(dataset))):
        img = dataset[i]['images'][modalità]
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        img = img[img > 0]  # Ignora background

        # ── STATISTICHE ──
        min_val, max_val = img.min(), img.max()
        mean_val, std_val = img.mean(), img.std()
        print(f"[{i}] {modalità.upper()} → min: {min_val:.2f}, max: {max_val:.2f}, mean: {mean_val:.2f}, std: {std_val:.2f}")

        # ── PLOT ISTOGRAMMA ──
        plt.subplot(1, max_cases, i + 1)
        plt.hist(img.flatten(), bins=bins, color='skyblue', edgecolor='black')
        plt.title(f"Sample {i}")
        plt.xlabel("Intensità")
        plt.ylabel("Voxel")
        plt.grid(True)

    plt.suptitle(f"Istogrammi intensità - Modalità: {modalità}", fontsize=14)
    plt.tight_layout()
    plt.show()

def normalize_minmax(img_array, mask_zero=True):
    mask = img_array > 0 if mask_zero else np.ones_like(img_array, dtype=bool)
    if np.any(mask):
        img_norm = np.zeros_like(img_array, dtype=np.float32)
        img_norm[mask] = (img_array[mask] - img_array[mask].min()) / (img_array[mask].max() - img_array[mask].min())
        return img_norm
    return img_array.astype(np.float32)


class BraTSDataset(Dataset):
    def __init__(self, csv_path, modalities=None, required_modalities=None, drop_missing=True,
                 norm_fn=normalize_minmax, add_channel_dim=True, placeholder_shape=(182, 218, 182),
                 transform=None):
        self.csv_path = csv_path
        self.modalities = modalities or ['t1', 't1ce', 't2', 'flair', 'seg']
        self.required_modalities = required_modalities or self.modalities
        self.drop_missing = drop_missing
        self.norm_fn = norm_fn  # funzione di normalizzazione, può essere None
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

            # Normalizzo solo se norm_fn è settata e modality non è 'seg'
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
                channels_report[mod].add(1)  # Assumiamo canale mancante = 1

    for mod in modalities:
        if len(shape_report[mod]) == 1:
            logging.info(f"[Check] Modalità '{mod}' ha shape uniforme: {next(iter(shape_report[mod]))}")
        else:
            logging.warning(f"[Check] Modalità '{mod}' ha shape variabile: {shape_report[mod]}")

        if len(channels_report[mod]) == 1:
            logging.info(f"[Check] Modalità '{mod}' ha numero canali uniforme: {next(iter(channels_report[mod]))}")
        else:
            logging.warning(f"[Check] Modalità '{mod}' ha numero canali variabile: {channels_report[mod]}")


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


def statistica_presenza_labels(dataset):
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    label_names = {1: 'NETC', 2: 'SNFH', 3: 'ET', 4: 'RC'}
    presenza_label = {label: 0 for label in label_names}

    for sample in dataset:
        seg = sample['images']['seg']
        if isinstance(seg, torch.Tensor):
            seg = seg.numpy()
        if seg.ndim == 4:
            seg = seg[0]  # [1, D, H, W] → [D, H, W]
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

if __name__ == "__main__":
    data_dir = r"G:\Il mio Drive\BraTS-GLI-PRE-TrainingData"
    csv_file = "catalog_brats_pre.csv"

    create_catalog_csv(data_dir, modalities, csv_file)

    missing = check_missing_files(csv_file, modalities)
    if missing:
        logging.warning(f"Campioni con file mancanti: {missing}")

    dataset = BraTSDataset(csv_file, modalities=list(modalities.keys()), drop_missing=True, norm_fn=None)

    check_shapes_and_channels(dataset, modalities=list(modalities.keys()), sample_size=5)

    subset_size = 3
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
