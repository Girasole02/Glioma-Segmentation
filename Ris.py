import os, matplotlib.pyplot as plt

def plot_risultati(save_dir, train_losses, val_losses, dice_metrics, hd95_metrics):  
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 5, 1)
    plt.plot(train_losses, label='Train Loss', color='orange')
    plt.plot(val_losses, label='Val Loss', color='purple')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right') 
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
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)  
        save_path = os.path.join(save_dir, f"metriche.png")
        plt.savefig(save_path)
