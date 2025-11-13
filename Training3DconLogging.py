import os, csv, torch, torch.nn as nn
import numpy as np, pandas as pd,logging, SimpleITK as sitk, torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
from monai.transforms import Compose, RandFlipd, RandRotate90d, RandGaussianNoised, RandScaleIntensityd, Rand3DElasticd
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.metrics import HausdorffDistanceMetric


class Config: 
    data_dir = r"G:\Il mio Drive\BraTS-GLI-PRE-TainingData"
    save_dir = r"G:\Il mio Drive\Logging"
    
    csv_file = "catalog_brats_pre.csv"
    modalities = {'t1': 't1', 't1ce': 't1ce', 't2': 't2', 'flair': 'flair', 'seg': 'seg'}
    batch_size = 1
    num_epochs = 100
    subset_size = None
    in_channels = 4          
    out_channels = 5         
    dropout_rate = 0.3      
    patience = 5
    augment = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_names = {0: 'Background', 1: 'NETC',2: 'SNFH', 3: 'ET', 4:'RC'}

def catalogo_csv(data_path, modalities, csv_output_path):
    if os.path.exists(csv_output_path): 
        print(f"Catalogo già presente: {csv_output_path}")
        return
    cases = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id', 'timepoint', 'treatment_status'] + list(modalities.keys()), delimiter=';')
        writer.writeheader()
        for case in cases:
            parts = case.split('-')
            if len(parts) < 4: 
                continue 
            row = { 'subject_id': parts[2], 'timepoint': parts[3],'treatment_status': 'pre' if parts[3].startswith('0') else 'post'}
            case_path = os.path.join(data_path, case)
            for mod, suffix in modalities.items():
                filepath = os.path.join(case_path, f"{case}-{suffix}.nii.gz")
                row[mod] = filepath if os.path.exists(filepath) else ""
            writer.writerow(row)
    logging.info(f"Catalogo creato: {csv_output_path}")


def check_dimensioni_uguali(dataset):
    dimensioni_trovate = set()
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        input_img = sample['input']  
        seg_img = sample['seg']      
        for j, mod in enumerate(['t1', 't1ce', 't2', 'flair']):
            mod_slice = input_img[j:j+1]  
            shape = mod_slice.shape
            dim_str = f"{mod}:{shape[1]}x{shape[2]}x{shape[3]}"
            dimensioni_trovate.add(dim_str)
        seg_shape = seg_img.shape
        dim_str_seg = f"seg:{seg_shape[1]}x{seg_shape[2]}x{seg_shape[3]}"
        dimensioni_trovate.add(dim_str_seg)
    print(f"Dimensioni trovate: {dimensioni_trovate}")
    return len(dimensioni_trovate) == 5

def check_file_mancanti(df, modalities=Config.modalities):
    missing = {mod: 0 for mod in modalities}
    valid_indices =[]
    for idx, row in df.iterrows():
        valid = all(os.path.exists(row[mod]) for mod in modalities)
        if valid:
            valid_indices.append(idx)
        else:
            for mod in modalities:
                if not os.path.exists(row[mod]):
                    missing[mod]+=1
    
    print( f"File mancanti: {missing}")
    if len(valid_indices) < len(df):
        print(f"Rimuovo{len(df)  - len(valid_indices)}")
    return df.iloc[valid_indices].reset_index(drop=True)

def statistica_presenza_labels(dataset, label_names=Config.label_names):
    presenza_label = {label: 0 for label in label_names}
    for sample in dataset:
        seg = sample['seg']
        if isinstance(seg, torch.Tensor):
            seg = seg.numpy()
        if seg.ndim==4:
            seg = seg[0]
        labels_presenti = np.unique(seg)
        for label in label_names:
            if label in labels_presenti:
                presenza_label[label] += 1
    n_samples, stats = len(dataset), []
    print(f"\n{'Label':<6}{'Nome':<6}{'Presenza':>9}{'Percentuale':>15}")
    print("-"*40)
    for label, count in presenza_label.items():
        percentuale = 100 * count / n_samples if n_samples > 0 else 0.0
        nome = label_names[label]
        stats.append((nome, count, percentuale))
        print(f"{label:<6}{nome:<6}{count:>9}{percentuale:>14.4f}%")

def analisi_voxel_segmentazioni(dataset, label_names = Config.label_names, save_dir=Config.save_dir):
    total_voxels, total_positive = 0, defaultdict(int)
    voxel_stats = defaultdict(list)
    for sample in dataset:
        seg = sample['seg']
        if isinstance(seg, torch.Tensor): seg = seg.numpy()
        if seg.ndim == 4: seg = seg[0]
        if total_voxels == 0: total_voxels = int(np.prod(seg.shape))
        labels, counts = np.unique(seg, return_counts=True)
        counts_dict = dict(zip(labels.tolist(), counts.tolist()))
        for label in label_names:
            count_label = int(counts_dict.get(label, 0))
            total_positive[label] += count_label
            voxel_stats[label_names[label]].append(count_label)
    
    print("\n--- Statistica Globale ---")
    for label in label_names:
        denom = total_voxels * len(dataset) if total_voxels > 0 and len(dataset) > 0 else 1
        percent = (total_positive[label] / denom * 100)
        logging.info(f"Label {label} ({label_names[label]}): {total_positive[label]} voxel positivi, {percent:.4f}% sul totale")
    
    print("\n--- Statistiche per campione ---")
    for label_name in label_names.values():
        valori = np.array(voxel_stats[label_name], dtype=np.float32)
        if len(valori) > 0: print(f"{label_name}: media={np.mean(valori):.2f}, max={np.max(valori)}, min={np.min(valori)}, std={np.std(valori):.2f}")

    if voxel_stats:
        plt.figure(figsize=(10, 6))
        plt.boxplot([voxel_stats[label] for label in label_names.values()], tick_labels=list(label_names.values()))
        plt.title("Distribuzione voxel per etichetta tumorale"), plt.ylabel("Numero di voxel"), plt.grid(True)
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)  
            save_path = os.path.join(save_dir, f"modalita_{sample['subject_id']}_{sample['timepoint']}.png")
            plt.savefig(save_path)
        plt.show()
        plt.close()


def visualizzazione_segmentazione(seg_tensor, slice_idx=None, save_dir=Config.save_dir):
    if isinstance(seg_tensor, torch.Tensor): seg_tensor = seg_tensor.numpy()
    if seg_tensor.ndim == 4: seg_tensor = seg_tensor[0]
    if slice_idx is None: slice_idx = seg_tensor.shape[0] // 2
    seg_slice = seg_tensor[slice_idx, :, :]
    
    plt.figure(figsize=(6, 6))
    plt.imshow(seg_slice, cmap='tab10', alpha=0.6, vmin=0, vmax=4)
    plt.title(f"Seg slice {slice_idx}"), plt.axis('off')

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True) 
        save_path = os.path.join(save_dir, f"segmentazione_{sample['subject_id']}_{sample['timepoint']}.png")
        plt.savefig(save_path)
    plt.show()
    plt.close()

def visualizzazione_sample(input_tensor, seg_tensor, slice_idx=None, modality_idx=1, save_dir=Config.save_dir):
    assert input_tensor.ndim == 4 and input_tensor.shape[0] == 4, f"Input tensor non valido: {input_tensor.shape}"
    if isinstance(input_tensor, torch.Tensor): input_tensor = input_tensor.numpy()
    if isinstance(seg_tensor, torch.Tensor): seg_tensor = seg_tensor.numpy()
    if seg_tensor.ndim == 4: seg_tensor = seg_tensor[0]
    if slice_idx is None: slice_idx = input_tensor.shape[1] // 2
    image_slice, seg_slice = input_tensor[modality_idx, slice_idx, :, :], seg_tensor[slice_idx, :, :]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_slice, cmap='gray')
    plt.title(f"Modality index {modality_idx} slice {slice_idx}"), plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image_slice, cmap='gray')
    plt.imshow(seg_slice, cmap='tab10', alpha=0.6, vmin=0, vmax=4)
    plt.title("Overlay con Segmentazione"), plt.axis('off')
    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)  
        save_path = os.path.join(save_dir, f"sample_{sample['subject_id']}_{sample['timepoint']}.png")
        plt.savefig(save_path)
    plt.show()

def visualizza_modalita_e_segmentazione(sample, pred_seg=None, slice_idx=None, save_dir= Config.save_dir):
    input_tensor = sample['input']  
    seg = sample['seg']             
    mods = ['t1', 't1ce', 't2', 'flair']
    imgs = []
    for i in range(4):
        imgs.append(input_tensor[i:i+1])
    imgs = [img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img for img in imgs]
    seg = seg.detach().cpu().numpy() if isinstance(seg, torch.Tensor) else seg
    imgs = [np.squeeze(img) for img in imgs]
    seg = np.squeeze(seg)
    if slice_idx is None:
        slice_idx = seg.shape[0] // 2
    slice_idx = min(slice_idx, seg.shape[0] - 1)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, (mod, img) in enumerate(zip(mods, imgs)):
        axes[i].imshow(img[slice_idx], cmap='gray')
        axes[i].set_title(mod.upper())
        axes[i].axis('off')

    overlay = pred_seg if pred_seg is not None else seg
    if isinstance(overlay, torch.Tensor):
        overlay = overlay.detach().cpu().numpy()
    overlay = np.squeeze(overlay)
    axes[4].imshow(imgs[1][slice_idx], cmap='gray')
    axes[4].imshow(overlay[slice_idx], cmap='tab10', alpha=0.6, vmin=0, vmax=4)
    axes[4].set_title('Segmentazione')
    axes[4].axis('off')
    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)  
        save_path = os.path.join(save_dir, f"modalita_{sample['subject_id']}_{sample['timepoint']}.png")
        plt.savefig(save_path)
    plt.show()

def visualizza_tre_assi(sample, modality='t1ce', slice_indices=None, save_dir=Config.save_dir):
    modality_idx = list(Config.modalities.keys()).index(modality)
    img_tensor = sample['input'][modality_idx] 

    if slice_indices is None:
        slice_indices = (img_tensor.shape[0]//2, img_tensor.shape[1]//2, img_tensor.shape[2]//2)
    axial_idx, coronal_idx, sagittal_idx = slice_indices
    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    # Assiale
    axes[0].imshow(img_tensor[axial_idx], cmap='gray')
    axes[0].set_title(f'Assiale (slice {axial_idx})')
    axes[0].axis('off')

    # Coronale
    axes[1].imshow(img_tensor[:, coronal_idx, :], cmap='gray')
    axes[1].set_title(f'Coronale (slice {coronal_idx})')
    axes[1].axis('off')

    # Sagittale
    axes[2].imshow(img_tensor[:, :, sagittal_idx], cmap='gray')
    axes[2].set_title(f'Sagittale (slice {sagittal_idx})')
    axes[2].axis('off')

    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)  
        save_path = os.path.join(save_dir, f"assi_{sample['subject_id']}_{sample['timepoint']}.png")
        plt.savefig(save_path)
    plt.show()
    plt.close()

def normalizzazione_zscore(img_array):
    mask = img_array > 0 
    if np.any(mask):
        vals = img_array[mask].astype(np.float32)
        mu = vals.mean() 
        sigma = vals.std()
        if sigma > 1e-6:
            img_norm = np.zeros_like(img_array, dtype=np.float32)
            img_norm[mask] = (vals - mu) / sigma
            return img_norm
    return img_array.astype(np.float32)

def pesi_classi(dataset, n_classi = 5, label_names=Config.label_names):
    conteggi = np.zeros(n_classi, dtype=np.float64)
    for sample in dataset:
        seg = sample ['seg']
        if isinstance(seg, torch.Tensor):
            seg = seg.numpy()
        if seg.ndim == 4:
            seg = seg[0]
        vals, counts = np.unique(seg, return_counts = True)
        for v, c in zip(vals, counts):
            if v<n_classi:
                conteggi[int(v)] += c
    
    valid_mask = conteggi > 0
    frequenze = np.zeros_like(conteggi)
    frequenze[valid_mask] = conteggi[valid_mask] / conteggi[valid_mask].sum()

    pesi = np.ones_like(conteggi, dtype=np.float32)
    pesi[valid_mask] = 1.0 / (frequenze[valid_mask] + 1e-6)
    pesi = pesi / pesi[valid_mask].sum() 
    pesi = pesi * valid_mask.sum()

    for i in range(n_classi):
        if i == 0:
            nome = 'Background'
        else:
            nome = label_names.get(i, 'Sconosciuta')
        conteggio = int(conteggi[i])
        frequenza = frequenze[i] if i < len(frequenze) else 0.0
        peso = pesi[i] if i < len(pesi) else 1.0
        print(f"{i:<10} {nome:<10} {conteggio:<12} {frequenza:<12.6f} {peso:<10.4f}")
    print("=" * 60)
    return torch.tensor(pesi, dtype=torch.float32)


class BraTS3DDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, sep=';')
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = {}
        
        for mod in Config.modalities:
            img = sitk.ReadImage(row[mod])
            img_array = sitk.GetArrayFromImage(img)
            img_array = np.expand_dims(img_array, axis=0)
            if mod != 'seg': 
                img_array = normalizzazione_zscore(img_array)

            else: 
                img_array = img_array.astype(np.uint8)
            images[mod] = img_array
        
        input_tensor = torch.cat([
            torch.from_numpy(images['t1']).float(),
            torch.from_numpy(images['t1ce']).float(),
            torch.from_numpy(images['t2']).float(),
            torch.from_numpy(images['flair']).float()
        ], dim=0)
        
        seg_tensor = torch.from_numpy(images['seg']).long()
        
        return {
            'input': input_tensor,
            'seg': seg_tensor,
            'subject_id': row['subject_id'],
            'timepoint': row['timepoint'],
            'treatment_status': row['treatment_status']
        }

class BraTS3DPatchDataset(BraTS3DDataset):
    def __init__(self, csv_path, patch_size=(128, 128, 128), augment=Config.augment, modalities=Config.modalities):
        super().__init__(csv_path)
        self.patch_size = patch_size  
        self.augment = augment  
        self.modalities = modalities

        if self.augment:
            self.transforms = Compose([
                RandFlipd(keys=["input", "seg"], prob=0.5, spatial_axis=[0, 1, 2]),
                RandRotate90d(keys=["input", "seg"], prob=0.3, max_k=3, spatial_axes=[1, 2]),
                Rand3DElasticd(keys=["input", "seg"], sigma_range=(5,7), magnitude_range=(100,200), prob=0.3),
                RandGaussianNoised(keys=["input"], prob=0.2, mean=0.0, std=0.05),
                RandScaleIntensityd(keys=["input"], factors=0.1, prob=0.2)
            ])

    def __getitem__(self, idx):
        images, row = self._load_volume(idx)

        if random.random() < 0.7:
            coords = self._sample_patch_coords(images['seg'])
        else:
            coords = self._sample_patch_coords(np.zeros_like(images['seg']))

        input_patches = []
        for mod in ['t1', 't1ce', 't2', 'flair']:
            patch = self._extract_patch(images[mod], coords)
            input_patches.append(torch.from_numpy(patch).float())

        seg_patch = self._extract_patch(images['seg'], coords)
        seg_patch = torch.from_numpy(seg_patch).long()

        if self.augment:
            data_dict = {
                "input": torch.cat(input_patches, dim=0),  
                "seg": seg_patch
            }
            augmented = self.transforms(data_dict)
            input_tensor = augmented["input"]
            seg_tensor = augmented["seg"]
        else:
            input_tensor = torch.cat(input_patches, dim=0)
            seg_tensor = seg_patch

        return {
            "input": input_tensor,
            "seg": seg_tensor,
            "subject_id": row['subject_id'],
            "timepoint": row['timepoint'],
            "treatment_status": row['treatment_status']
        }

    def _load_volume(self, idx):
        row = self.df.iloc[idx]
        images = {}
        for mod in self.modalities:
            img = sitk.ReadImage(row[mod])
            img_array = sitk.GetArrayFromImage(img)
            img_array = np.expand_dims(img_array, axis=0) 
            if mod != 'seg':
                img_array = normalizzazione_zscore(img_array)   
            else:
                img_array = img_array.astype(np.uint8)
            images[mod] = img_array
        return images, row

    def _sample_patch_coords(self, seg):
        seg_np = seg[0] if seg.ndim == 4 else seg
        depth, height, width = seg_np.shape
        pd, ph, pw = self.patch_size

        tumor_voxels = np.where(seg_np > 0)
        if len(tumor_voxels[0]) > 0:
            idx = random.randint(0, len(tumor_voxels[0]) - 1)
            d, h, w = tumor_voxels[0][idx], tumor_voxels[1][idx], tumor_voxels[2][idx]
            d = max(0, min(d - pd // 2, depth - pd))
            h = max(0, min(h - ph // 2, height - ph))
            w = max(0, min(w - pw // 2, width - pw))
        else:
            d = random.randint(0, max(0, depth - pd))
            h = random.randint(0, max(0, height - ph))
            w = random.randint(0, max(0, width - pw))
        return (d, h, w)

    def _extract_patch(self, img, coords):
        d, h, w = coords
        pd, ph, pw = self.patch_size
        if img.ndim == 4:  # (C, D, H, W)
            return img[:, d:d+pd, h:h+ph, w:w+pw]
        else:  
            return img[d:d+pd, h:h+ph, w:w+pw]


class UNet3D(nn.Module):
    def __init__(self, in_channels=Config.in_channels, out_channels=Config.out_channels, dropout_rate=Config.dropout_rate):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        self.enc1 = self._block_3d(self.in_channels, 32, use_dropout=False)
        self.enc2 = self._block_3d(32, 64, use_dropout=False)
        self.enc3 = self._block_3d(64, 128, use_dropout=False)

        self.bottleneck = self._block_3d(128, 256, use_dropout=True)

        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._block_3d(256, 128, use_dropout=True)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._block_3d(128, 64, use_dropout=True)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._block_3d(64, 32, use_dropout=True)

        self.final = nn.Conv3d(32, self.out_channels, kernel_size=1)

    def _block_3d(self, in_channels, features, use_dropout=False):
        """Blocco conv 3D con BatchNorm e ReLU. Dropout3d opzionale."""
        layers = [
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout3d(p=self.dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool3d(enc1, 2))
        enc3 = self.enc3(F.max_pool3d(enc2, 2))
        bottleneck = self.bottleneck(F.max_pool3d(enc3, 2))

        up3 = self.up3(bottleneck)
        if up3.shape[2:] != enc3.shape[2:]:
            up3 = F.interpolate(up3, size=enc3.shape[2:], mode='trilinear', align_corners=False)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))

        up2 = self.up2(dec3)
        if up2.shape[2:] != enc2.shape[2:]:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='trilinear', align_corners=False)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))

        up1 = self.up1(dec2)
        if up1.shape[2:] != enc1.shape[2:]:
            up1 = F.interpolate(up1, size=enc1.shape[2:], mode='trilinear', align_corners=False)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))

        return self.final(dec1)
    
class EarlyStopping:
    def __init__(self, model, save_path="best_model.pth", patience = Config.patience, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.model = model
        self.save_path = save_path
        self.best_model_state = None
        self.epoch_count = 0
        
    def __call__(self, val_loss, epoch):
        self.epoch_count += 1
        improvement = self.best_loss - val_loss
        
        if improvement > self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = self.model.state_dict().copy()
            self.counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.best_model_state,
                'best_loss': self.best_loss,
            }, self.save_path)
            
            print(f"Nuovo best model salvato! Loss: {val_loss:.4f} (Epoca {epoch})")
            return False
        else:
            self.counter += 1
            print(f"Early stopping: {self.counter}/{self.patience} (Loss: {val_loss:.4f}, Best: {self.best_loss:.4f})")
            
            if self.counter >= self.patience:
                print(f"Early stopping attivato! Nessun miglioramento per {self.patience} epoche")
                if self.best_model_state is not None:
                    self.model.load_state_dict(self.best_model_state)
                return True
        return False
    
    def load_best_model(self):
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Best model ricaricato (loss: {self.best_loss:.4f})")
        elif os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_loss = checkpoint['best_loss']
            print(f"Best model caricato da file (loss: {self.best_loss:.4f})")

def train_e_validazione(model, train_loader, val_loader, device, num_epochs, class_weights=None):

    if torch.cuda.is_available():
        logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceCELoss(
        to_onehot_y=True,        
        softmax=True,            
        include_background=False,
        lambda_dice=0.5,         
        lambda_ce=0.5,           
        weight=class_weights  
    )
    early_stopping = EarlyStopping(
        model=model, 
        save_path="best_model_brats.pth",
        min_delta=0.001
    )
    
    train_losses = []
    val_losses = []
    dice_metrics = {'ET': [], 'WT': [], 'TC': []}
    hd95_metrics = {'ET': [], 'WT': [], 'TC': []}  
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0

        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=True) as train_pbar:
            for batch in train_pbar:
                inputs = batch['input'].to(device)
                segs = batch['seg'].to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                segs_target = segs.long()
                
                loss = criterion(outputs, segs_target)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                train_batches += 1
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{epoch_train_loss/train_batches:.4f}'
                })

        avg_train_loss = epoch_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=True) as val_pbar:
                for batch in val_pbar:
                    inputs = batch['input'].to(device)
                    segs = batch['seg'].to(device)
                    
                    outputs = sliding_window_inference(
                        inputs, 
                        roi_size=(128, 128, 128),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.5
                    )
                    
                    target_shape = segs.shape[2:]
                    outputs = F.interpolate(outputs, size=target_shape, mode='trilinear', align_corners=False)
                    segs_target = segs.long()
                    
                    loss = criterion(outputs, segs_target)
                    epoch_val_loss += loss.item()
                    val_batches += 1
                    
                    val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})

        avg_val_loss = epoch_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        dice_et, dice_wt, dice_tc = calculate_dice_metrics(model, val_loader, device)
        hd95_et, hd95_wt, hd95_tc = calculate_hd95_metrics(model, val_loader, device)  
        
        dice_metrics['ET'].append(dice_et)
        dice_metrics['WT'].append(dice_wt)
        dice_metrics['TC'].append(dice_tc)
        
        hd95_metrics['ET'].append(hd95_et)  
        hd95_metrics['WT'].append(hd95_wt)  
        hd95_metrics['TC'].append(hd95_tc)  
    
        if early_stopping(avg_val_loss, epoch + 1):
            break
            
        print(f'Epoch {epoch+1:3d} | '
              f'Train: {avg_train_loss:.4f} | '
              f'Val: {avg_val_loss:.4f} | '
              f'Dice - ET: {dice_et:.4f}, WT: {dice_wt:.4f}, TC: {dice_tc:.4f} | '
              f'HD95 - ET: {hd95_et:.4f}, WT: {hd95_wt:.4f}, TC: {hd95_tc:.4f}')  
    
    early_stopping.load_best_model()
    
    return train_losses, val_losses, dice_metrics, hd95_metrics  


def dice_score(pred, target, label):
    if isinstance(label, list):
        pred_mask = np.isin(pred, label)
        target_mask = np.isin(target, label)
    else:
        pred_mask = (pred == label)
        target_mask = (target == label)
    
    intersection = np.sum(pred_mask & target_mask)
    union = np.sum(pred_mask) + np.sum(target_mask)
    return (2.0 * intersection) / (union + 1e-6)

def calculate_dice_metrics(model, val_loader, device):
    model.eval()
    dice_et, dice_wt, dice_tc = 0.0, 0.0, 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            segs = batch['seg'].to(device)
            outputs = sliding_window_inference(
                inputs,
                roi_size=(128, 128, 128),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5
            )
                
            target_shape = segs.shape[2:]
            if outputs.shape[2:] != target_shape:
                outputs = F.interpolate(
                    outputs, 
                    size=target_shape, 
                    mode='trilinear', 
                    align_corners=False
                )

            preds = torch.argmax(outputs, dim=1)
            pred_np = preds.cpu().numpy()[0]
            seg_np = segs.cpu().numpy()[0][0]
                
            dice_et += dice_score(pred_np, seg_np, label=3)
            dice_wt += dice_score(pred_np, seg_np, label=[1, 2, 3])
            dice_tc += dice_score(pred_np, seg_np, label=[1, 3])
                
            n_samples += 1
                
    return dice_et / n_samples, dice_wt / n_samples, dice_tc / n_samples

def calculate_hd95_metrics(model, val_loader, device):
    model.eval()
    
    hd_metric_et = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    hd_metric_wt = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    hd_metric_tc = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    
    n_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            segs = batch['seg'].to(device)
            
            outputs = sliding_window_inference(
                inputs,
                roi_size=(128, 128, 128),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5
            )
            
            target_shape = segs.shape[2:]
            if outputs.shape[2:] != target_shape:
                outputs = F.interpolate(
                    outputs, 
                    size=target_shape, 
                    mode='trilinear', 
                    align_corners=False
                )

            preds = torch.argmax(outputs, dim=1, keepdim=True)
            segs_target = segs.long()
            

            pred_et = (preds == 3).float()
            target_et = (segs_target == 3).float()
            
            pred_wt = torch.isin(preds, torch.tensor([1, 2, 3], device=device)).float()
            target_wt = torch.isin(segs_target, torch.tensor([1, 2, 3], device=device)).float()
        
            pred_tc = torch.isin(preds, torch.tensor([1, 3], device=device)).float()
            target_tc = torch.isin(segs_target, torch.tensor([1, 3], device=device)).float()
            
    
            hd_metric_et(y_pred=pred_et, y=target_et)
            hd_metric_wt(y_pred=pred_wt, y=target_wt)
            hd_metric_tc(y_pred=pred_tc, y=target_tc)
            
            n_samples += 1
    
    hd95_et = hd_metric_et.aggregate().item() if n_samples > 0 else float('inf')
    hd95_wt = hd_metric_wt.aggregate().item() if n_samples > 0 else float('inf')
    hd95_tc = hd_metric_tc.aggregate().item() if n_samples > 0 else float('inf')
    
    return hd95_et, hd95_wt, hd95_tc


def plot_risultati(train_losses, val_losses, dice_metrics, hd95_metrics):  
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 5, 1)
    plt.plot(train_losses, label='Train Loss', color='orange')
    plt.plot(val_losses, label='Val Loss', color='purple')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 5, 2)
    plt.plot(dice_metrics['ET'], label='ET', color='red')
    plt.plot(dice_metrics['WT'], label='WT', color='blue') 
    plt.plot(dice_metrics['TC'], label='TC', color='green')
    plt.title('Dice Metrics per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    

    plt.subplot(1, 5, 3)
    plt.plot(hd95_metrics['ET'], label='ET', color='red')
    plt.plot(hd95_metrics['WT'], label='WT', color='blue') 
    plt.plot(hd95_metrics['TC'], label='TC', color='green')
    plt.title('Hausdorff Distance (95° Percentile)')
    plt.xlabel('Epoch')
    plt.ylabel('HD95')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 5, 4)
    final_dice = [dice_metrics['ET'][-1], dice_metrics['WT'][-1], dice_metrics['TC'][-1]]
    labels_dice = ['ET', 'WT', 'TC']
    colors_dice = ['red', 'blue', 'green']
    
    plt.bar(labels_dice, final_dice, color=colors_dice)
    plt.title('Final Dice Metrics')
    plt.ylabel('Dice Score')
    plt.ylim(0, 1)
    
    plt.subplot(1, 5, 5)
    final_hd95 = [hd95_metrics['ET'][-1], hd95_metrics['WT'][-1], hd95_metrics['TC'][-1]]
    labels_hd95 = ['ET', 'WT', 'TC']
    colors_hd95 = ['red', 'blue', 'green']
    
    plt.bar(labels_hd95, final_hd95, color=colors_hd95)
    plt.title('Final HD95 Metrics')
    plt.ylabel('HD95')
    
    plt.tight_layout()
    plt.show()


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
    
    df_clean = check_file_mancanti(df, Config.modalities)
    logging.info(f"Campioni validi dopo check: {len(df_clean)}")

    if Config.subset_size is None:
        Config.subset_size = len(df_clean)
        logging.info(f"Utilizzo Dataset Intero: {Config.subset_size} campioni")
    else:
        logging.info(f"Utilizzo Subset: {Config.subset_size} campioni")

    if len(df_clean) > Config.subset_size:
        df_subset = df_clean.head(Config.subset_size).copy()
    else:
        df_subset = df_clean.copy()
        Config.subset_size = len(df_clean)
    
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
    

    dataset = BraTS3DDataset(patients_csv)
    train_dataset = BraTS3DPatchDataset(train_csv)
    val_dataset = BraTS3DDataset(val_csv)
    
    logging.info(f"Training dataset: {len(train_dataset)} campioni (patches augmentate)")
    logging.info(f"Validation dataset: {len(val_dataset)} campioni (volumi interi)")
    
    print("\n===== ANALISI STATISTICHE =====")
    logging.info(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Inizio Analisi statistiche")

    check_dimensioni_uguali(dataset)
    statistica_presenza_labels(dataset)
    analisi_voxel_segmentazioni(dataset)
    

    print("\n===== VISUALIZZAZIONI =====")
    logging.info(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Inizio Visualizzazioni")
    if len(dataset) > 0:
        sample = dataset[0]
        input_tensor = sample['input']
        seg_tensor = sample['seg']
        
        visualizza_modalita_e_segmentazione(sample)
        visualizzazione_sample(input_tensor, seg_tensor)
        visualizzazione_segmentazione(seg_tensor)
        visualizza_tre_assi(sample, modality='t1ce')
    
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
    class_weights = pesi_classi(train_dataset)
    logging.info(f"Pesi finali: {class_weights}")

    print("\n===== INIZIO TRAINING =====")
    logging.info(f"{datetime.now():%Y-%m-%d %H:%M:%S}- Inizio Training")

    model = UNet3D().to(Config.device)
    logging.info(f"Modello: {sum(p.numel() for p in model.parameters()):,} parametri")
    logging.info(f"Epoche: {Config.num_epochs}")
    logging.info(f"Early stopping patience: {Config.patience}")
    
    train_losses, val_losses, dice_metrics, hd95_metrics = train_e_validazione(  
        model, train_loader, val_loader, Config.device, Config.num_epochs, class_weights
    )

    print("\n===== RISULTATI FINALI =====")
    logging.info(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Inizio Calcolo e Plot Risultati")
    plot_risultati(train_losses, val_losses, dice_metrics, hd95_metrics)  
    
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