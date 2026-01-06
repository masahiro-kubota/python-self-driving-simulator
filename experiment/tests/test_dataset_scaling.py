import numpy as np
import pytest
from pathlib import Path
from torch.utils.data import DataLoader
from experiment.data.dataset import ScanControlDataset

@pytest.fixture
def mock_dataset_dir(tmp_path):
    """Create a temporary directory with dummy batched data."""
    data_dir = tmp_path / "mock_data"
    data_dir.mkdir()
    
    # Create 3 batches
    # Batch 0: 10 samples
    # Batch 1: 5 samples
    # Batch 2: 8 samples
    # Total: 23 samples
    
    batches = [
        (0, 10),
        (1, 5),
        (2, 8)
    ]
    
    for batch_id, n_samples in batches:
        # Scans: (N, 10) random float32
        scans = np.random.randn(n_samples, 10).astype(np.float32)
        # Steers: (N,) random float32
        steers = np.random.randn(n_samples).astype(np.float32)
        # Accels: (N,) random float32
        accels = np.random.randn(n_samples).astype(np.float32)
        
        batch_base = f"batch_{batch_id:04d}"
        np.save(data_dir / f"{batch_base}_scans.npy", scans)
        np.save(data_dir / f"{batch_base}_steers.npy", steers)
        np.save(data_dir / f"{batch_base}_accelerations.npy", accels)
        
    return data_dir

def test_lazy_loading(mock_dataset_dir):
    """Test that dataset loads correctly without concatenating (simulated by checking length)."""
    dataset = ScanControlDataset(mock_dataset_dir)
    
    # Check total length
    assert len(dataset) == 10 + 5 + 8
    
    # Check random access
    # Index 0 -> Batch 0, Index 0
    scan0, target0 = dataset[0]
    assert scan0.shape == (10,)
    
    # Index 10 -> Batch 1, Index 0
    scan10, target10 = dataset[10]
    assert scan10.shape == (10,)
    
    # Index 15 -> Batch 2, Index 0
    scan15, target15 = dataset[15]
    assert scan15.shape == (10,)
    
    # Index 22 -> Batch 2, Index 7 (Last element)
    scan22, target22 = dataset[22]
    assert scan22.shape == (10,)
    
    # Check boundary
    with pytest.raises(IndexError):
        _ = dataset[23]

def test_dataloader_compatibility(mock_dataset_dir):
    """Verify DataLoader works with the dataset."""
    dataset = ScanControlDataset(mock_dataset_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    batches = list(dataloader)
    total_samples = sum(b[0].shape[0] for b in batches)
    assert total_samples == 23

def test_legacy_loading(tmp_path):
    """Verify legacy single-file loading still works."""
    data_dir = tmp_path / "legacy_data"
    data_dir.mkdir()
    
    scans = np.random.randn(5, 10).astype(np.float32)
    steers = np.random.randn(5).astype(np.float32)
    accels = np.random.randn(5).astype(np.float32)
    
    np.save(data_dir / "scans.npy", scans)
    np.save(data_dir / "steers.npy", steers)
    np.save(data_dir / "accelerations.npy", accels)
    
    dataset = ScanControlDataset(data_dir)
    assert len(dataset) == 5
    assert dataset[0][0].shape == (10,)
