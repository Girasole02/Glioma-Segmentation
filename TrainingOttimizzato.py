# IMPORT LIBRERIE
import os, csv, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, logging, webbrowser, SimpleITK as sitk, matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from collections import defaultdict
from sklearn.model_selection import train_test_split
from config.definitions import ROOT_DIR
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# CONFIGURAZIONE
class Config:
    data_dir = ROOT_DIR / "data/BraTS-GLI-PRE-TrainingData" # r"G:\Il mio Drive\BraTS-GLI-PRE-TrainingData"
    csv_file = "catalog_brats_pre.csv"
    modalities = {'seg': 'seg', 't1ce': 't1c', 't1': 't1n', 'flair': 't2f', 't2': 't2w'}
    required_mods = ['t1', 't1ce', 't2', 'flair', 'seg']
    placeholder_shape = (182, 218, 182)
    model_type = '2.5d'  # '2d', '2.5d', '3d'
    batch_size = 5
    num_epochs = 5
    subset_size = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_names = {1: 'NETC', 2: 'SNFH', 3: 'ET', 4: 'RC'}

# FUNZIONI UTILITY
def create_catalog_csv(data_path, modalities, csv_output_path):
    if os.path.exists(csv_output_path): 
        logging.info(f"Catalogo già presente: {csv_output_path}")
        return
        
    cases = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id', 'timepoint', 'treatment_status'] + list(modalities.keys()), delimiter=';')
        writer.writeheader()
        
        for case in cases:
            parts = case.split('-')
            if len(parts) < 4: continue
                
            row = {
                'subject_id': parts[2], 'timepoint': parts[3],
                'treatment_status': 'pre' if parts[3].startswith('0') else 'post'
            }
            
            case_path = os.path.join(data_path, case)
            for mod, suffix in modalities.items():
                filepath = os.path.join(case_path, f"{case}-{suffix}.nii.gz")
                row[mod] = filepath if os.path.exists(filepath) else ""
                
            writer.writerow(row)
    
    logging.info(f"Catalogo creato: {csv_output_path}")

def check_missing_files(csv_path, modalities):
    df = pd.read_csv(csv_path, sep=';')
    missing_files = {}
    for idx, row in df.iterrows():
        case_missing = [mod for mod in modalities if not row.get(mod, "") or not os.path.exists(row[mod])]
        if case_missing:
            missing_files[f"{row['subject_id']}_{row['timepoint']}"] = case_missing
    return missing_files

def normalize_minmax(img_array, mask_zero=True):
    mask = img_array > 0 if mask_zero else np.ones_like(img_array, dtype=bool)
    if np.any(mask):
        img_norm = np.zeros_like(img_array, dtype=np.float32)
        vals = img_array[mask].astype(np.float32)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin: img_norm[mask] = (vals - vmin) / (vmax - vmin)
        return img_norm
    return img_array.astype(np.float32)

def _to_tensor(x): return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)

def custom_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        if key == 'images':
            images = {mod: [] for mod in batch[0]['images'].keys()}
            for sample in batch:
                for mod, img in sample['images'].items():
                    images[mod].append(_to_tensor(img))
            collated['images'] = {mod: torch.stack(tensors, dim=0) for mod, tensors in images.items()}
        elif key == 'valid_mask':
            masks = {mod: [] for mod in batch[0]['valid_mask'].keys()}
            for sample in batch:
                for mod, valid in sample['valid_mask'].items():
                    masks[mod].append(bool(valid))
            collated['valid_mask'] = {mod: torch.tensor(vals, dtype=torch.bool) for mod, vals in masks.items()}
        else:
            collated[key] = [sample[key] for sample in batch]
    return collated

# DATASET PRINCIPALE
class BraTSDataset(Dataset):
    def __init__(self, csv_path, modalities=None, required_modalities=None, drop_missing=True,
                 norm_fn=normalize_minmax, add_channel_dim=True, placeholder_shape=(182, 218, 182), transform=None):
        self.df = pd.read_csv(csv_path, sep=';')
        self.modalities = modalities or ['t1', 't1ce', 't2', 'flair', 'seg']
        self.required_modalities = required_modalities or self.modalities
        self.drop_missing = drop_missing
        self.norm_fn = norm_fn
        self.add_channel_dim = add_channel_dim
        self.placeholder_shape = placeholder_shape
        self.transform = transform
        self._check_and_filter_missing_files()

    def _check_and_filter_missing_files(self):
        missing_count = {mod: 0 for mod in self.modalities if mod in self.df.columns}
        rows_to_drop = []
        for idx, row in self.df.iterrows():
            missing_required = False
            for mod in self.modalities:
                if mod not in self.df.columns: continue
                val = row[mod]
                if not isinstance(val, str) or not val or pd.isna(val) or not os.path.exists(val):
                    missing_count[mod] += 1
                    if mod in self.required_modalities: missing_required = True
            if self.drop_missing and missing_required: rows_to_drop.append(idx)
        
        logging.info(f"File mancanti: {missing_count}")
        if rows_to_drop: 
            logging.warning(f"Rimuovo {len(rows_to_drop)} campioni con file mancanti")
            self.df = self.df.drop(rows_to_drop).reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images, valid_mask = {}, {}
        
        for modality in self.modalities:
            filepath = row.get(modality, "")
            if not isinstance(filepath, str) or not filepath or pd.isna(filepath) or not os.path.exists(filepath):
                images[modality] = np.zeros((1,) + self.placeholder_shape, dtype=np.float32)
                valid_mask[modality] = False
                continue
                
            img = sitk.ReadImage(filepath)
            img_array = sitk.GetArrayFromImage(img)
            
            if modality != 'seg' and self.norm_fn: img_array = self.norm_fn(img_array)
            elif modality == 'seg': img_array = img_array.astype(np.uint8)
            
            if self.add_channel_dim: img_array = np.expand_dims(img_array, axis=0)
            else: img_array = img_array.astype(np.float32)
            
            images[modality] = img_array
            valid_mask[modality] = True

        sample = {
            'subject_id': row.get('subject_id'), 'timepoint': row.get('timepoint'), 
            'treatment_status': row.get('treatment_status'), 'images': images, 'valid_mask': valid_mask
        }
        
        return self.transform(sample) if self.transform else sample

# DATASET SPECIALIZZATI
class BraTS2DDataset(Dataset):
    def __init__(self, brats_3d_dataset, slice_axis=0):
        self.original_dataset, self.slice_axis = brats_3d_dataset, slice_axis
        self.samples = []
        for vol_idx in range(len(brats_3d_dataset)):
            seg = brats_3d_dataset[vol_idx]['images']['seg']
            D = seg.shape[self.slice_axis + 1]
            for slice_idx in range(D):
                seg_slice = seg[0].take(indices=slice_idx, axis=self.slice_axis)
                if np.any(seg_slice > 0): self.samples.append((vol_idx, slice_idx))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.samples[idx]
        sample = self.original_dataset[vol_idx]
        images = {}
        
        for mod, img in sample['images'].items():
            slice_2d = img[0].take(indices=slice_idx, axis=self.slice_axis)
            images[mod] = np.expand_dims(slice_2d.astype(np.float32), axis=0)
            
        return {'subject_id': sample['subject_id'], 'slice_idx': slice_idx, 
                'images': images, 'valid_mask': sample['valid_mask']}

class BraTS25DDataset(Dataset):
    def __init__(self, brats_3d_dataset, context_slices=3, slice_axis=0):
        assert context_slices % 2 == 1, "context_slices deve essere dispari"
        self.original_dataset, self.context_slices, self.slice_axis = brats_3d_dataset, context_slices, slice_axis
        self.half_context, self.samples = context_slices // 2, []
        
        for vol_idx in range(len(brats_3d_dataset)):
            seg = brats_3d_dataset[vol_idx]['images']['seg']
            D = seg.shape[self.slice_axis + 1]
            for center_slice in range(self.half_context, D - self.half_context):
                center_seg = seg[0].take(indices=center_slice, axis=self.slice_axis)
                if np.any(center_seg > 0): self.samples.append((vol_idx, center_slice))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        vol_idx, center_slice = self.samples[idx]
        sample = self.original_dataset[vol_idx]
        start_slice, end_slice = center_slice - self.half_context, center_slice + self.half_context + 1
        images = {}
        
        for mod in ['t1', 't1ce', 't2', 'flair']:
            img = sample['images'][mod]
            slices = [img[0].take(indices=i, axis=self.slice_axis).astype(np.float32) for i in range(start_slice, end_slice)]
            images[mod] = np.expand_dims(np.stack(slices), axis=0)
            
        seg = sample['images']['seg']
        images['seg'] = np.expand_dims(seg[0].take(indices=center_slice, axis=self.slice_axis), axis=0)
        
        return {'subject_id': sample['subject_id'], 'slice_idx': center_slice, 
                'images': images, 'valid_mask': sample['valid_mask']}

class BraTS3DDataset(Dataset):
    def __init__(self, brats_3d_dataset):
        self.original_dataset = brats_3d_dataset
        self.samples = [vol_idx for vol_idx in range(len(brats_3d_dataset)) 
                       if np.any(brats_3d_dataset[vol_idx]['images']['seg'] > 0)]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.original_dataset[self.samples[idx]]

# MODELLI
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
        
        if up3.shape[2:] != enc3.shape[2:]:
            up3 = F.interpolate(up3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
        
        up2 = self.up2(dec3)
        if up2.shape[2:] != enc2.shape[2:]:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        
        up1 = self.up1(dec2)
        if up1.shape[2:] != enc1.shape[2:]:
            up1 = F.interpolate(up1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        
        return self.final(dec1)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=5):
        super().__init__()
        self.enc1 = self._block_3d(in_channels, 32)
        self.enc2 = self._block_3d(32, 64)
        self.enc3 = self._block_3d(64, 128)
        self.bottleneck = self._block_3d(128, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._block_3d(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._block_3d(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._block_3d(64, 32)
        self.final = nn.Conv3d(32, out_channels, kernel_size=1)

    def _block_3d(self, in_channels, features):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1), nn.BatchNorm3d(features), nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1), nn.BatchNorm3d(features), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool3d(enc1, 2))
        enc3 = self.enc3(F.max_pool3d(enc2, 2))
        bottleneck = self.bottleneck(F.max_pool3d(enc3, 2))
        
        up3 = self.up3(bottleneck)
        if up3.shape[2:] != enc3.shape[2:]: enc3 = F.interpolate(enc3, size=up3.shape[2:], mode='trilinear', align_corners=False)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
        
        up2 = self.up2(dec3)
        if up2.shape[2:] != enc2.shape[2:]: enc2 = F.interpolate(enc2, size=up2.shape[2:], mode='trilinear', align_corners=False)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        
        up1 = self.up1(dec2)
        if up1.shape[2:] != enc1.shape[2:]: enc1 = F.interpolate(enc1, size=up1.shape[2:], mode='trilinear', align_corners=False)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        
        return self.final(dec1)

# WRAPPER E FACTORY
def get_model(model_type, base_model_class, **kwargs):
    if model_type == '2d': 
        return base_model_class(**kwargs)
    elif model_type == '2.5d': 
        class Wrapper25D(nn.Module):
            def __init__(self, context_slices=3, **inner_kwargs):
                super().__init__()
                # Filtra in_channels dagli inner_kwargs per evitare duplicati
                filtered_kwargs = {k: v for k, v in inner_kwargs.items() if k != 'in_channels'}
                self.model = base_model_class(in_channels=4 * context_slices, **filtered_kwargs)
            def forward(self, x): return self.model(x)
        return Wrapper25D(**kwargs)
    elif model_type == '3d': 
        return base_model_class(**kwargs)
    else: 
        raise ValueError(f"Tipo modello non supportato: {model_type}")

def get_dataset(dataset_type, original_dataset, **kwargs):
    if dataset_type == '2d': return BraTS2DDataset(original_dataset, **kwargs)
    elif dataset_type == '2.5d': return BraTS25DDataset(original_dataset, **kwargs)
    elif dataset_type == '3d': return BraTS3DDataset(original_dataset, **kwargs)
    else: raise ValueError(f"Tipo dataset non supportato: {dataset_type}")

# VISUALIZZAZIONI E ANALISI
def analizza_intensità_e_plot(dataset, modalità='t1', max_cases=5, bins=100):
    plt.figure(figsize=(15, 4))
    for i in range(min(max_cases, len(dataset))):
        img = dataset[i]['images'][modalità]
        if isinstance(img, torch.Tensor): img = img.numpy()
        if img.ndim == 4: img = img[0]
        img = img[img > 0]
        if img.size == 0: continue
        
        min_val, max_val, mean_val, std_val = img.min(), img.max(), img.mean(), img.std()
        print(f"[{i}] {modalità.upper()} → min: {min_val:.2f}, max: {max_val:.2f}, mean: {mean_val:.2f}, std: {std_val:.2f}")
        
        plt.subplot(1, max_cases, i + 1)
        plt.hist(img.flatten(), bins=bins, edgecolor='black')
        plt.title(f"Sample {i}"), plt.xlabel("Intensità"), plt.ylabel("Voxel"), plt.grid(True)
    
    plt.suptitle(f"Istogrammi intensità - Modalità: {modalità}", fontsize=14)
    plt.tight_layout(), plt.show()

def visualize_segmentation_only(seg_tensor, slice_idx=None):
    if isinstance(seg_tensor, torch.Tensor): seg_tensor = seg_tensor.numpy()
    if seg_tensor.ndim == 4: seg_tensor = seg_tensor[0]
    if slice_idx is None: slice_idx = seg_tensor.shape[0] // 2
    seg_slice = seg_tensor[slice_idx, :, :]
    
    plt.figure(figsize=(6, 6))
    cmap = plt.get_cmap('tab10', 5)
    plt.imshow(seg_slice, cmap=cmap, vmin=0, vmax=4)
    cbar = plt.colorbar(ticks=range(5))
    cbar.ax.set_yticklabels(['Background'] + list(Config.label_names.values()))
    plt.title(f"Segmentazione (slice {slice_idx})"), plt.axis('off'), plt.show()

def visualize_sample(input_tensor, seg_tensor, slice_idx=None, modality_idx=1):
    assert input_tensor.ndim == 4 and input_tensor.shape[0] == 4, f"Input tensor non valido: {input_tensor.shape}"
    if isinstance(input_tensor, torch.Tensor): input_tensor = input_tensor.numpy()
    if isinstance(seg_tensor, torch.Tensor): seg_tensor = seg_tensor.numpy()
    if seg_tensor.ndim == 4: seg_tensor = seg_tensor[0]
    if slice_idx is None: slice_idx = input_tensor.shape[1] // 2
    
    image_slice, seg_slice = input_tensor[modality_idx, slice_idx, :, :], seg_tensor[slice_idx, :, :]
    cmap_seg = plt.get_cmap('jet')
    colors = ['yellow', 'green', 'red', 'blue']
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_slice, cmap='gray')
    plt.title(f"Modality index {modality_idx} (slice {slice_idx})"), plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image_slice, cmap='gray')
    plt.imshow(seg_slice, cmap=cmap_seg, alpha=0.5)
    plt.title("Overlay con segmentazione"), plt.axis('off')
    
    legend_patches = [Patch(color=color, label=Config.label_names[i]) for i, color in zip(Config.label_names.keys(), colors)]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(), plt.show()

def statistica_presenza_labels(dataset):
    presenza_label = {label: 0 for label in Config.label_names}
    for sample in dataset:
        seg = sample['images']['seg']
        if isinstance(seg, torch.Tensor): seg = seg.numpy()
        if seg.ndim == 4: seg = seg[0]
        labels_presenti = np.unique(seg)
        for label in Config.label_names:
            if label in labels_presenti: presenza_label[label] += 1
    
    n_samples, stats = len(dataset), []
    print(f"\n{'Label':<6} {'Nome':<6} {'Presenza':>9} {'Percentuale':>15}\n" + "-" * 40)
    for label, count in presenza_label.items():
        percentuale = 100 * count / n_samples if n_samples > 0 else 0.0
        nome = Config.label_names[label]
        stats.append((nome, count, percentuale))
        print(f"{label:<6} {nome:<6} {count:>9} {percentuale:>14.4f}%")
    
    labels, counts, percentages = [s[0] for s in stats], [s[1] for s in stats], [s[2] for s in stats]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, counts)
    for bar, pct in zip(bars, percentages): plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5, f"{pct:.1f}%", ha='center')
    plt.title("Presenza delle etichette tumorali nei campioni"), plt.ylabel("Numero di campioni")
    plt.ylim(0, max(counts) * 1.2 if counts else 1), plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout(), plt.show()

def analisi_voxel_segmentazioni(dataset):
    total_voxels, total_positive = 0, defaultdict(int)
    voxel_stats = defaultdict(list)
    
    for sample in dataset:
        seg = sample['images']['seg']
        if isinstance(seg, torch.Tensor): seg = seg.numpy()
        if seg.ndim == 4: seg = seg[0]
        if total_voxels == 0: total_voxels = int(np.prod(seg.shape))
        
        labels, counts = np.unique(seg, return_counts=True)
        counts_dict = dict(zip(labels.tolist(), counts.tolist()))
        
        for label in Config.label_names:
            count_label = int(counts_dict.get(label, 0))
            total_positive[label] += count_label
            voxel_stats[Config.label_names[label]].append(count_label)
    
    print("\n--- Statistica Globale ---")
    for label in Config.label_names:
        denom = total_voxels * len(dataset) if total_voxels > 0 and len(dataset) > 0 else 1
        percent = (total_positive[label] / denom * 100)
        print(f"Label {label} ({Config.label_names[label]}): {total_positive[label]} voxel positivi, {percent:.4f}% sul totale")
    
    print("\n--- Statistiche per campione ---")
    for label_name in Config.label_names.values():
        valori = np.array(voxel_stats[label_name], dtype=np.float32)
        if len(valori) > 0: print(f"{label_name}: media={np.mean(valori):.2f}, max={np.max(valori)}, min={np.min(valori)}, std={np.std(valori):.2f}")
    
    if voxel_stats:
        plt.figure(figsize=(10, 6))
        plt.boxplot([voxel_stats[label] for label in Config.label_names.values()], tick_labels=list(Config.label_names.values()))
        plt.title("Distribuzione voxel per etichetta tumorale"), plt.ylabel("Numero di voxel"), plt.grid(True)
        plt.tight_layout(), plt.show()

# METRICHE
def dice_score_binary(pred, target):
    intersection = np.logical_and(pred, target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)

def evaluate_dice_brats(model, dataloader, device, model_type='2d'):
    model.eval()
    et_scores, wt_scores, tc_scores = [], [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if model_type == '3d':
                inputs = torch.cat([batch['images'][mod] for mod in ['t1', 't1ce', 't2', 'flair']], dim=1).to(device)
                targets = batch['images']['seg'].squeeze(1).long().to(device)
                outputs = model(inputs)
                if outputs.shape[2:] != targets.shape[1:]: outputs = F.interpolate(outputs, size=targets.shape[1:], mode='trilinear', align_corners=False)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                targets_np = targets.cpu().numpy()
            else:
                if model_type == '2.5d':
                    modalità_inputs = [batch['images'][mod].squeeze(1) for mod in ['t1', 't1ce', 't2', 'flair']]
                    inputs = torch.cat(modalità_inputs, dim=1).to(device)
                    targets = batch['images']['seg'].squeeze(1).long().to(device)
                else:
                    inputs = torch.cat([batch['images'][mod] for mod in ['t1', 't1ce', 't2', 'flair']], dim=1).to(device)
                    targets = batch['images']['seg'].squeeze(1).long().to(device)
                
                outputs = model(inputs)
                if outputs.shape[2:] != targets.shape[1:]: outputs = F.interpolate(outputs, size=targets.shape[1:], mode='bilinear', align_corners=False)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                targets_np = targets.cpu().numpy()
            
            for i in range(preds.shape[0]):
                p, t = preds[i], targets_np[i]
                et_scores.append(dice_score_binary(p == 3, t == 3))
                wt_scores.append(dice_score_binary(np.isin(p, [1, 2, 3, 4]), np.isin(t, [1, 2, 3, 4])))
                tc_scores.append(dice_score_binary(np.isin(p, [1, 3, 4]), np.isin(t, [1, 3, 4])))
    
    return {
        'ET': float(np.mean(et_scores)) if et_scores else 0.0,
        'WT': float(np.mean(wt_scores)) if wt_scores else 0.0,
        'TC': float(np.mean(tc_scores)) if tc_scores else 0.0
    }

# MAIN
if __name__ == "__main__":
    cfg = Config()
    
    # Setup dati
    create_catalog_csv(cfg.data_dir, cfg.modalities, cfg.csv_file)
    missing = check_missing_files(cfg.csv_file, cfg.modalities)
    if missing: logging.warning(f"Campioni con file mancanti: {missing}")
    
    dataset = BraTSDataset(cfg.csv_file, modalities=list(cfg.modalities.keys()), drop_missing=False)
    subset_size = Config.subset_size
    subset = Subset(dataset, list(range(min(subset_size, len(dataset)))))
    logging.info(f"Dataset caricato con {len(dataset)} campioni, subset di {len(subset)}")
    
    # Visualizzazioni
    sample = subset[0]
    print(f"Sample: subject_id={sample['subject_id']}, timepoint={sample['timepoint']}, treatment={sample['treatment_status']}")
    imgs = [sample['images'][m] if sample['images'][m].ndim == 4 else np.expand_dims(sample['images'][m], axis=0) for m in ['t1', 't1ce', 't2', 'flair']]
    input_tensor = torch.tensor(np.vstack(imgs), dtype=torch.float32)
    
    analizza_intensità_e_plot(dataset, modalità='t1ce', max_cases=3)
    visualize_sample(input_tensor, torch.tensor(sample['images']['seg']))
    visualize_segmentation_only(torch.tensor(sample['images']['seg']))
    statistica_presenza_labels(subset)
    analisi_voxel_segmentazioni(subset)
    
    print("\n" + "="*50 + "\nAVVIO TRAINING".center(50) + "\n" + "="*50)
    
    # Training setup
    BASE_MODEL_CLASS = UNet2D if cfg.model_type in ['2d', '2.5d'] else UNet3D
    model_kwargs = {'context_slices': 3} if cfg.model_type == '2.5d' else {}
    dataset_kwargs = {'context_slices': 3, 'slice_axis': 0} if cfg.model_type == '2.5d' else {}
    
    brats_dataset = get_dataset(cfg.model_type, subset, **dataset_kwargs)
    logging.info(f"Dataset {cfg.model_type.upper()} creato con {len(brats_dataset)} campioni")
    
    # Split - MODIFICATO
    patient_ids = list({brats_dataset[i]['subject_id'] for i in range(len(brats_dataset))})
    train_patients, val_patients = train_test_split(patient_ids, test_size=0.2, random_state=42)
    
    train_indices = [i for i in range(len(brats_dataset)) if brats_dataset[i]['subject_id'] in train_patients]
    val_indices = [i for i in range(len(brats_dataset)) if brats_dataset[i]['subject_id'] in val_patients]
    
    train_dataset, val_dataset = Subset(brats_dataset, train_indices), Subset(brats_dataset, val_indices)
    collate_fn = custom_collate_fn  # MODIFICATO: rimosso custom_collate_fn_3d
    
    train_loader = DataLoader(train_dataset, batch_size=2 if cfg.model_type == '3d' else 4, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2 if cfg.model_type == '3d' else 4, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    # Model e training
    model = get_model(cfg.model_type, BASE_MODEL_CLASS, in_channels=4, out_channels=5, **model_kwargs).to(cfg.device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=cfg.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, val_losses, dice_et, dice_wt, dice_tc = [], [], [], [], []
    
    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            if cfg.model_type == '3d':
                inputs = torch.cat([batch['images'][mod] for mod in ['t1', 't1ce', 't2', 'flair']], dim=1).to(cfg.device)
                targets = batch['images']['seg'].squeeze(1).long().to(cfg.device)
            elif cfg.model_type == '2.5d':
                modalità_inputs = [batch['images'][mod].squeeze(1) for mod in ['t1', 't1ce', 't2', 'flair']]
                inputs = torch.cat(modalità_inputs, dim=1).to(cfg.device)
                targets = batch['images']['seg'].squeeze(1).long().to(cfg.device)
            else:
                inputs = torch.cat([batch['images'][mod] for mod in ['t1', 't1ce', 't2', 'flair']], dim=1).to(cfg.device)
                targets = batch['images']['seg'].squeeze(1).long().to(cfg.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if cfg.model_type == '3d':
                if outputs.shape[2:] != targets.shape[1:]: outputs = F.interpolate(outputs, size=targets.shape[1:], mode='trilinear', align_corners=False)
            else:
                if outputs.shape[2:] != targets.shape[-2:]: outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        
        avg_train_loss = running_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if cfg.model_type == '3d':
                    inputs = torch.cat([batch['images'][mod] for mod in ['t1', 't1ce', 't2', 'flair']], dim=1).to(cfg.device)
                    targets = batch['images']['seg'].squeeze(1).long().to(cfg.device)
                    outputs = model(inputs)
                    if outputs.shape[2:] != targets.shape[1:]: outputs = F.interpolate(outputs, size=targets.shape[1:], mode='trilinear', align_corners=False)
                elif cfg.model_type == '2.5d':
                    modalità_inputs = [batch['images'][mod].squeeze(1) for mod in ['t1', 't1ce', 't2', 'flair']]
                    inputs = torch.cat(modalità_inputs, dim=1).to(cfg.device)
                    targets = batch['images']['seg'].squeeze(1).long().to(cfg.device)
                    outputs = model(inputs)
                    if outputs.shape[2:] != targets.shape[-2:]: outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    inputs = torch.cat([batch['images'][mod] for mod in ['t1', 't1ce', 't2', 'flair']], dim=1).to(cfg.device)
                    targets = batch['images']['seg'].squeeze(1).long().to(cfg.device)
                    outputs = model(inputs)
                    if outputs.shape[2:] != targets.shape[-2:]: outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)
                
                loss = criterion(outputs, targets)
                val_running_loss += float(loss.item())
        
        avg_val_loss = val_running_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        
        metrics = evaluate_dice_brats(model, val_loader, cfg.device, cfg.model_type)
        dice_et.append(metrics['ET']), dice_wt.append(metrics['WT']), dice_tc.append(metrics['TC'])
        
        logging.info(f"Epoch [{epoch+1}/{cfg.num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                     f"ET: {metrics['ET']:.4f} | WT: {metrics['WT']:.4f} | TC: {metrics['TC']:.4f}")
    
    print("Training completato!")
    
    # Plot finali
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss'), plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title('Andamento Loss'), plt.legend(), plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(dice_et, label='ET'), plt.plot(dice_wt, label='WT'), plt.plot(dice_tc, label='TC')
    plt.xlabel('Epoch'), plt.ylabel('Dice Score'), plt.title('Dice Score per classe BraTS'), plt.legend(), plt.grid(True)
    
    plt.tight_layout(), plt.show()