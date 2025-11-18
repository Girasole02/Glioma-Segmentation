import numpy as np, torch, os, logging, csv, copy

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

def pesi_classi(dataset, label_names, n_classi = 4):
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


class EarlyStopping:
    def __init__(self, model, patience, save_path="best_model.pth", min_delta=0.0):
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
            self.best_model_state = copy.deepcopy(self.model.state_dict())
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