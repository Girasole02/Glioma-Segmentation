import os
import pytest
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Subset

from Preprocessing import (
    create_catalog_csv,
    check_missing_files,
    BraTSDataset,
    normalize_minmax,
    modalities
)

# CONFIGURAZIONE PATH TEST 
TEST_DATA_DIR = r"G:\Il mio Drive\BratsTest"
TEST_CSV_FILE = "test_catalog.csv"

# CREAZIONE CSV UNA SOLA VOLTA 
@pytest.fixture(scope="module")
def setup_csv():
    csv_path = os.path.join(TEST_DATA_DIR, TEST_CSV_FILE)
    create_catalog_csv(TEST_DATA_DIR, modalities, csv_path)
    assert os.path.exists(csv_path), "CSV non creato"
    yield csv_path
    # Cleanup opzionale
    os.remove(csv_path)

# TEST CSV 
def test_csv_creation(setup_csv):
    df = pd.read_csv(setup_csv, sep=';')
    assert not df.empty, "Il CSV è vuoto"
    assert 'subject_id' in df.columns, "'subject_id' mancante"
    assert all(col in df.columns for col in modalities.keys()), "Colonne delle modalità mancanti"

# TEST MOD MANCANTI
def test_check_missing_files(setup_csv):
    missing = check_missing_files(setup_csv, modalities)
    assert isinstance(missing, dict), "Output non è un dizionario"
    # Può essere vuoto, ma non deve causare errori
    for k, v in missing.items():
        assert isinstance(v, list)

# TEST COMPLETEZZA METADATI
def test_metadata_integrity(setup_csv):
    df = pd.read_csv(setup_csv, sep=';')
    for col in ['subject_id', 'timepoint', 'treatment_status']:
        assert df[col].notna().all(), f"Colonna {col} contiene NaN"

# TEST SULLE LABELS
def test_valid_segmentation_labels(setup_csv):
    dataset = BraTSDataset(setup_csv, modalities=['seg'])
    valid_labels = {0, 1, 2, 3, 4}
    for i in range(min(10, len(dataset))):  # solo i primi 10 per velocità
        seg = dataset[i]['images']['seg']
        unique_vals = np.unique(seg)
        assert set(unique_vals).issubset(valid_labels), f"Label non valida: {unique_vals}"

# TEST DATASET 
def test_brats_dataset_loading(setup_csv):
    dataset = BraTSDataset(setup_csv, modalities=list(modalities.keys()))
    assert len(dataset) > 0, "Dataset vuoto"
    sample = dataset[0]
    assert 'images' in sample, "'images' mancante"
    assert 'seg' in sample['images'], "Segmentazione mancante"

    # Controllo dimensioni attese
    expected_shape = (1,182,218,182)
    for mod in ['t1', 't1ce', 't2', 'flair']:
        assert mod in sample['images'], f"{mod} mancante"
        assert sample['images'][mod].shape == expected_shape, f"Shape errata per {mod}: {sample['images'][mod].shape}"

    # Segmentazione
    seg = sample['images']['seg']
    assert seg.shape == expected_shape, f"Shape segmentazione errata: {seg.shape}"

# TEST NORMALIZZAZIONE
def test_normalize_minmax():
    img = np.random.randint(1, 1000, size=(10, 10, 10)).astype(np.float32)
    norm = normalize_minmax(img)
    assert norm.min() >= 0 and norm.max() <= 1, "Valori fuori intervallo [0,1]"
    assert norm.shape == img.shape, "Shape alterata"
    
def test_image_value_range(setup_csv):
    dataset = BraTSDataset(setup_csv, modalities=['t1', 't2'])
    for i in range(5):
        sample = dataset[i]
        for mod in ['t1', 't2']:
            img = sample['images'][mod]
            assert np.isfinite(img).all(), f"Valori non finiti in {mod}"
            assert 0.0 <= img.min() <= 1.0, f"{mod} min fuori range: {img.min()}"
            assert 0.0 <= img.max() <= 1.0, f"{mod} max fuori range: {img.max()}"

