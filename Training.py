import torch, torch.nn.functional as F, gc
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.metrics import HausdorffDistanceMetric, DiceMetric
from monai.inferers import sliding_window_inference
from Utils import EarlyStopping

def train_e_validazione(model, train_loader, val_loader, device, num_epochs, patience, class_weights):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    criterion = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=0.5,
        lambda_ce=0.5,
        weight=class_weights.to(device) if class_weights is not None else None
    )

    early_stopping = EarlyStopping(model=model, patience=patience, save_path="best_model_brats.pth", min_delta=0.001)

    dice_et = DiceMetric(include_background=False, reduction="mean")
    dice_wt = DiceMetric(include_background=False, reduction="mean")
    dice_tc = DiceMetric(include_background=False, reduction="mean")

    hd_et = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    hd_wt = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    hd_tc = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    train_losses, val_losses = [], []
    dice_metrics = {'ET': [], 'WT': [], 'TC': []}
    hd95_metrics = {'ET': [], 'WT': [], 'TC': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as train_pbar:
            for batch in train_pbar:
                inputs = batch['input'].to(device)
                labels = batch['seg'].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_epoch += loss.item()
                train_pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Avg": f"{train_loss_epoch/(train_pbar.n+1):.4f}"
                })

                del inputs, labels, outputs, loss
                torch.cuda.empty_cache()
                gc.collect()

        avg_train_loss = train_loss_epoch / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_epoch = 0.0

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") as val_pbar:
                for batch in val_pbar:
                    inputs = batch['input'].to(device)
                    labels = batch['seg'].to(device)

                    outputs = sliding_window_inference(
                        inputs,
                        roi_size=(128, 128, 128),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.5
                    )

                    outputs = F.interpolate(outputs, size=labels.shape[2:], mode='trilinear', align_corners=False)

                    loss = criterion(outputs, labels.long())
                    val_loss_epoch += loss.item()

                    preds = torch.argmax(outputs, dim=1, keepdim=True)

                    pred_et = (preds == 3).float(); target_et = (labels == 3).float()
                    pred_wt = torch.isin(preds, torch.tensor([1,2,3], device=device)).float()
                    target_wt = torch.isin(labels, torch.tensor([1,2,3], device=device)).float()
                    pred_tc = torch.isin(preds, torch.tensor([1,3], device=device)).float()
                    target_tc = torch.isin(labels, torch.tensor([1,3], device=device)).float()

                    dice_et(y_pred=pred_et, y=target_et)
                    dice_wt(y_pred=pred_wt, y=target_wt)
                    dice_tc(y_pred=pred_tc, y=target_tc)

                    hd_et(y_pred=pred_et, y=target_et)
                    hd_wt(y_pred=pred_wt, y=target_wt)
                    hd_tc(y_pred=pred_tc, y=target_tc)

                    del inputs, labels, outputs, preds, pred_et, target_et, pred_wt, target_wt, pred_tc, target_tc, loss
                    torch.cuda.empty_cache()
                    gc.collect()

        avg_val_loss = val_loss_epoch / len(val_loader)
        val_losses.append(avg_val_loss)


        dice_metrics['ET'].append(torch.nan_to_num(dice_et.aggregate(), nan=0.0).item())
        dice_metrics['WT'].append(torch.nan_to_num(dice_wt.aggregate(), nan=0.0).item())
        dice_metrics['TC'].append(torch.nan_to_num(dice_tc.aggregate(), nan=0.0).item())

        hd95_metrics['ET'].append(torch.nan_to_num(hd_et.aggregate(), nan=0.0).item())
        hd95_metrics['WT'].append(torch.nan_to_num(hd_wt.aggregate(), nan=0.0).item())
        hd95_metrics['TC'].append(torch.nan_to_num(hd_tc.aggregate(), nan=0.0).item())

        dice_et.reset(); dice_wt.reset(); dice_tc.reset()
        hd_et.reset(); hd_wt.reset(); hd_tc.reset()

        scheduler.step(avg_val_loss)
        if early_stopping(avg_val_loss, epoch+1):
            break

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Dice ET/WT/TC: {dice_metrics['ET'][-1]:.4f}/{dice_metrics['WT'][-1]:.4f}/{dice_metrics['TC'][-1]:.4f} | "
              f"HD95 ET/WT/TC: {hd95_metrics['ET'][-1]:.4f}/{hd95_metrics['WT'][-1]:.4f}/{hd95_metrics['TC'][-1]:.4f}")

    early_stopping.load_best_model()
    return train_losses, val_losses, dice_metrics, hd95_metrics
