import torch 

class Config: 
    data_dir = r"G:\Il mio Drive\BraTS-GLI-PRE-TrainingData"
    save_dir = r"G:\Il mio Drive\Logging"
    csv_file = "catalog_brats_pre.csv"
    modalities = {'t1': 't1c', 't1ce': 't1n', 't2': 't2f', 'flair': 't2w', 'seg': 'seg'}
    batch_size = 1
    num_epochs = 1
    subset_size = 2
    in_channels = 4          
    out_channels = 4        
    dropout_rate = 0.3      
    patience = 5
    augment = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_names = {0: 'Background', 1: 'NETC',2: 'SNFH', 3: 'ET'}