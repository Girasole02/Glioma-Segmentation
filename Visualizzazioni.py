import os, torch
import numpy as np
import matplotlib.pyplot as plt

def visualizzazione_segmentazione(save_dir, sample, slice_idx=None):
    seg_tensor = sample['seg']
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

def visualizzazione_sample(save_dir, sample, slice_idx=None, modality_idx=1):
    input_tensor = sample['input']
    seg_tensor = sample['seg']

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

def visualizza_modalita_e_segmentazione(save_dir, sample, pred_seg=None, slice_idx=None):
    input_tensor = sample['input']  
    seg = sample['seg']             
    mods = ['t1', 't1ce', 't2', 'flair']
    imgs = []

    for i in range(4):
        imgs.append(input_tensor[i:i+1])
    imgs = [img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img for img in imgs]
    seg = seg.detach().cpu().numpy() if isinstance(seg, torch.Tensor) else seg
    imgs = [np.squeeze(img) for img in imgs]
    seg = np.squeeze(seg) if seg.ndim == 4 else seg
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


def visualizza_tre_assi(save_dir, modalities,  sample, modality='t1ce', slice_indices=None):
    modality_idx = list(modalities.keys()).index(modality)
    input_tensor = sample['input'][modality_idx] 

    if slice_indices is None:
        slice_indices = (input_tensor.shape[0]//2, input_tensor.shape[1]//2, input_tensor.shape[2]//2)
    axial_idx, coronal_idx, sagittal_idx = slice_indices
    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    axes[0].imshow(input_tensor[axial_idx], cmap='gray')
    axes[0].set_title(f'Assiale (slice {axial_idx})')
    axes[0].axis('off')

    axes[1].imshow(input_tensor[:, coronal_idx, :], cmap='gray')
    axes[1].set_title(f'Coronale (slice {coronal_idx})')
    axes[1].axis('off')

    axes[2].imshow(input_tensor[:, :, sagittal_idx], cmap='gray')
    axes[2].set_title(f'Sagittale (slice {sagittal_idx})')
    axes[2].axis('off')

    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)  
        save_path = os.path.join(save_dir, f"assi_{sample['subject_id']}_{sample['timepoint']}.png")
        plt.savefig(save_path)
