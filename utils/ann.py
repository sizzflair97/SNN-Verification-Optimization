import torch
from torch.utils.data import DataLoader, TensorDataset
from mnist import MNIST
from tqdm.auto import tqdm
from pathlib import Path
from argparse import ArgumentParser
epochs = 100

class SimpleANN(torch.nn.Module):
    def __init__(self, n_hidden_neurons:int) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.hidden = torch.nn.Linear(28*28, n_hidden_neurons)
        self.output = torch.nn.Linear(n_hidden_neurons, 10)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.relu(x)
        return x

def train_ann(dataset:str, n_hidden_neurons:int, device:torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> None:
    model_path = Path("models") / "ann" / f"{dataset}_mlp_{n_hidden_neurons}.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # torch.use_deterministic_algorithms(True)
    torch.random.manual_seed(42)
    
    data, labels = MNIST("data/mnist/MNIST/raw/").load_training()
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(data, dtype=torch.float32).view(-1, 28, 28),
            torch.tensor(labels, dtype=torch.long)
        ),
        batch_size=32,
        shuffle=True
    )
    test_data, test_labels = MNIST("data/mnist/MNIST/raw/").load_testing()
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(test_data, dtype=torch.float32).view(-1, 28, 28),
            torch.tensor(test_labels, dtype=torch.long)
        ),
        batch_size=32,
        shuffle=False
    )

    model = SimpleANN(n_hidden_neurons).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)
    
    for _ in tqdm(range(epochs), desc="Epochs"):
        accumulated_loss = 0.0
        for batch in (pbar := tqdm(train_loader, leave=False)):
            images, targets = batch
            outputs = model(images.to(device))
            loss = torch.nn.functional.cross_entropy(outputs, targets.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            accumulated_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
        scheduler.step(accumulated_loss)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in (pbar := tqdm(test_loader, leave=False)):
                images, targets = batch
                outputs = model(images.to(device))
                loss = torch.nn.functional.cross_entropy(outputs, targets.to(device))
                correct += outputs.argmax(dim=1).eq(targets.to(device)).sum().item()
                total += len(targets)
            accuracy = correct / total
            print("acc:", accuracy)

    torch.save(model.state_dict(), model_path)
    return

def load_ann(model_path:Path=Path(), n_hidden_neurons:int=512, device:torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> SimpleANN:
    if model_path == Path():
        model_path /= Path("models") / "ann" / f"mnist_mlp_{n_hidden_neurons}.pth"
    model = SimpleANN(n_hidden_neurons).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def get_gradient(model:SimpleANN, x:torch.Tensor, target:torch.Tensor, device:torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> torch.Tensor:
    model.eval()
    x = x.to(device).requires_grad_(True)
    target = target.to(device)
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()
    if x.grad is None:
        raise ValueError("Gradient is None.")
    return x.grad.detach().cpu()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-hidden-neurons", dest="n_hidden_neurons", type=int, default=512)
    parser.add_argument("--dataset", dest="dataset", type=str, default="mnist")
    args = parser.parse_args()
    dataset = args.dataset
    n_hidden_neurons = args.n_hidden_neurons
    train_ann(dataset, n_hidden_neurons)