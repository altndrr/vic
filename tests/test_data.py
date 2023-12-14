from pathlib import Path

import pytest
import torch

from src.data import (
    DTD,
    SUN397,
    UCF101,
    Caltech101,
    EuroSAT,
    FGVCAircraft,
    Flowers102,
    Food101,
    ImageNet,
    OxfordPets,
    StanfordCars,
)


@pytest.mark.parametrize("batch_size", [32, 128])
def test_caltech101(batch_size):
    data_dir = "data/"

    dm = Caltech101(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "Caltech101").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 8242

    batch = next(iter(dm.train_dataloader()))
    x, y = batch["images_tensor"], batch["targets_one_hot"]
    assert len(x) == batch_size
    assert y.dim() == 2
    assert y.shape[0] == batch_size and y.shape[1] == 100
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32, 128])
def test_dtd(batch_size):
    data_dir = "data/"

    dm = DTD(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "DTD").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 5640

    batch = next(iter(dm.train_dataloader()))
    x, y = batch["images_tensor"], batch["targets_one_hot"]
    assert len(x) == batch_size
    assert y.dim() == 2
    assert y.shape[0] == batch_size and y.shape[1] == 47
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32, 128])
def test_eurosat(batch_size):
    data_dir = "data/"

    dm = EuroSAT(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "EuroSAT").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 27000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch["images_tensor"], batch["targets_one_hot"]
    assert len(x) == batch_size
    assert y.dim() == 2
    assert y.shape[0] == batch_size and y.shape[1] == 10
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32, 128])
def test_fgvc_aircraft(batch_size):
    data_dir = "data/"

    dm = FGVCAircraft(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "FGVCAircraft").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 10_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch["images_tensor"], batch["targets_one_hot"]
    assert len(x) == batch_size
    assert y.dim() == 2
    assert y.shape[0] == batch_size and y.shape[1] == 100
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32, 128])
def test_flowers102(batch_size):
    data_dir = "data/"

    dm = Flowers102(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "Flowers102").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 8189

    batch = next(iter(dm.train_dataloader()))
    x, y = batch["images_tensor"], batch["targets_one_hot"]
    assert len(x) == batch_size
    assert y.dim() == 2
    assert y.shape[0] == batch_size and y.shape[1] == 102
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32, 128])
def test_food101(batch_size):
    data_dir = "data/"

    dm = Food101(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "Food101").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 101_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch["images_tensor"], batch["targets_one_hot"]
    assert len(x) == batch_size
    assert y.dim() == 2
    assert y.shape[0] == batch_size and y.shape[1] == 101
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32, 128])
def test_imagenet(batch_size):
    data_dir = "data/"

    dm = ImageNet(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "ImageNet").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val)
    assert len(dm.data_val) == len(dm.data_test)
    assert num_datapoints == 1331167

    batch = next(iter(dm.test_dataloader()))
    x, y = batch["images_tensor"], batch["targets_one_hot"]
    assert len(x) == batch_size
    assert y.dim() == 2
    assert y.shape[0] == batch_size and y.shape[1] == 1000
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32, 128])
def test_oxford_pets(batch_size):
    data_dir = "data/"

    dm = OxfordPets(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "OxfordPets").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 7349

    batch = next(iter(dm.train_dataloader()))
    x, y = batch["images_tensor"], batch["targets_one_hot"]
    assert len(x) == batch_size
    assert y.dim() == 2
    assert y.shape[0] == batch_size and y.shape[1] == 37
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32, 128])
def test_stanford_cars(batch_size):
    data_dir = "data/"

    dm = StanfordCars(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "StanfordCars").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 16185

    batch = next(iter(dm.train_dataloader()))
    x, y = batch["images_tensor"], batch["targets_one_hot"]
    assert len(x) == batch_size
    assert y.dim() == 2
    assert y.shape[0] == batch_size and y.shape[1] == 196
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32, 128])
def test_sun397(batch_size):
    data_dir = "data/"

    dm = SUN397(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "SUN397").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 39_700

    batch = next(iter(dm.train_dataloader()))
    x, y = batch["images_tensor"], batch["targets_one_hot"]
    assert len(x) == batch_size
    assert y.dim() == 2
    assert y.shape[0] == batch_size and y.shape[1] == 397
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32, 128])
def test_ucf101(batch_size):
    data_dir = "data/"

    dm = UCF101(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "UCF101").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 13320

    batch = next(iter(dm.train_dataloader()))
    x, y = batch["images_tensor"], batch["targets_one_hot"]
    assert len(x) == batch_size
    assert y.dim() == 2
    assert y.shape[0] == batch_size and y.shape[1] == 101
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
