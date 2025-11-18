import logging, os

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
    logging.info(f"Dimensioni trovate: {dimensioni_trovate}")
    return len(dimensioni_trovate) == 5

def check_file_mancanti(df, modalities):
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

