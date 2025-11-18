import numpy as np, logging, os, torch, matplotlib.pyplot as plt
from collections import defaultdict

def statistica_presenza_labels(dataset, label_names):
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
        logging.info(f"{label:<6}{nome:<6}{count:>9}{percentuale:>14.4f}%")

def analisi_voxel_segmentazioni(dataset, label_names, save_dir):
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