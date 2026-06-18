from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def get_dataloaders(batch_size=16):
    dataset = datasets.ImageFolder(
        "data/processed/semantic_faces",
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2   # fixed from 4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2   # fixed from 4
    )

    return train_loader, test_loader