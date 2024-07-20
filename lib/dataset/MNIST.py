import pathlib

import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import v2


class DataSetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def imgTransform_train(img):
    t1 = ToTensor()(img)
    t2 = transformations(t1)
    t3 = torch.transpose(t2, dim0=-3, dim1=-2)
    t4 = torch.transpose(t3, dim0=-2, dim1=-1)
    return t4


def imgTransform_test(img):
    t1 = ToTensor()(img)
    t2 = torch.transpose(t1, dim0=-3, dim1=-2)
    t3 = torch.transpose(t2, dim0=-2, dim1=-1)
    return t3


def oneHotEncoding(label):
    t1 = torch.tensor(label)
    t2 = torch.nn.functional.one_hot(t1, num_classes=10)  # pylint: disable=not-callable
    return t2


transformations = v2.Compose(
    [
        v2.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.0), antialias=True),
    ]
)


def get_train_val_test(
    batch_size=64, train=55000, val=5000, test=10000, shuffle=True, dataset_dir=None
):
    # return the train val test dataloaders
    assert train + val <= 60000
    assert test <= 10000

    if not dataset_dir:
        dataset_dir = pathlib.PosixPath(__file__).parent / "downloads/MNIST"
    else:
        dataset_dir = pathlib.PosixPath(dataset_dir)

    if not dataset_dir.is_dir():
        dataset_dir.mkdir(parents=True)

    training_data_all = datasets.MNIST(
        root=(dataset_dir / "train"),
        train=True,
        download=True,
        transform=None,
        target_transform=oneHotEncoding,
    )

    test_data_all = datasets.MNIST(
        root=(dataset_dir / "train"),
        train=False,
        download=True,
        transform=imgTransform_test,
        target_transform=oneHotEncoding,
    )

    training_data = DataSetWrapper(
        torch.utils.data.Subset(training_data_all, range(0, train)),
        transform=imgTransform_train,
    )
    val_data = DataSetWrapper(
        torch.utils.data.Subset(training_data_all, range(train, train + val)),
        transform=imgTransform_test,
    )
    test_data = DataSetWrapper(
        torch.utils.data.Subset(test_data_all, range(0, test)),
        transform=imgTransform_test,
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

    print(
        f"training on {len(training_data)}\n"
        f"validating on {len(val_data)}\n"
        f"testing on {len(val_data)}\n"
    )

    return train_dataloader, val_dataloader, test_dataloader
