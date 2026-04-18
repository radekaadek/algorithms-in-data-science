import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

device = torch.device("cpu")


class FashionMNISTNet(nn.Module):
    def __init__(self):
        super(FashionMNISTNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=3, padding=1, bias=False
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=3, padding=1, bias=False
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=84),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=84, out_features=10),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=10,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

model = FashionMNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

epochs = 3

print("[x] Rozpoczęcie trenowania...")
total_start_time = time.time()

for epoch in range(epochs):
    epoch_start_time = time.time()

    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    train_loop = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{epochs}: ",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
        colour="cyan",
    )
    for inputs, targets in train_loop:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)

    model.eval()

    eval_train_loop = tqdm(
        train_loader,
        desc="Eval Train:   ",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
        colour="green",
    )
    for inputs, targets in eval_train_loop:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()

    train_acc = 100 * correct_train / total_train

    correct_test = 0
    total_test = 0

    with torch.no_grad():
        eval_test_loop = tqdm(
            test_loader,
            desc="Eval Test:    ",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.2,
            colour="green",
        )
        for inputs, targets in eval_test_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += targets.size(0)
            correct_test += (predicted == targets).sum().item()

    test_acc = 100 * correct_test / total_test

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    print(
        f"↳ Koniec Epoki {epoch + 1} ({epoch_duration:.2f} s) - Średnia strata: {avg_loss:.4f} | Dokładność treningowa: {train_acc:.2f}% | Dokładność testowa: {test_acc:.2f}%"
    )

total_end_time = time.time()
total_duration = total_end_time - total_start_time
minutes, seconds = divmod(total_duration, 60)

print("\n[x] Trenowanie zakończone pomyślnie!")
print(f"Całkowity czas wykonania: {int(minutes)} min {seconds:.2f} s")
