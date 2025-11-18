import torch, numpy as np, pandas as pd,SimpleITK as sitk,random
from torch.utils.data import Dataset
from monai.transforms import Compose, RandFlipd, RandRotate90d, RandGaussianNoised, RandScaleIntensityd, Rand3DElasticd
from Utils import normalizzazione_zscore

class BraTS3DDataset(Dataset):
    def __init__(self, csv_path, modalities):
        self.df = pd.read_csv(csv_path, sep=';')
        self.modalities = modalities
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
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
    def __init__(self, csv_path, augment, modalities, patch_size=(128, 128, 128)):
        super().__init__(csv_path,modalities)
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
        if img.ndim == 4: 
            return img[:, d:d+pd, h:h+ph, w:w+pw]
        else:  
            return img[d:d+pd, h:h+ph, w:w+pw]
