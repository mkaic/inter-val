from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torchvision import transforms as T
from vae import VAE
import torch
from pathlib import Path
import wandb
import copy

EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = Path("./logs")

if not LOG_DIR.exists():
    LOG_DIR.mkdir(parents=True)

print("loading datasets...")
input_size = 128
transforms = T.Compose(
    [T.RandomHorizontalFlip(), T.CenterCrop(148), T.Resize(input_size), T.ToTensor()]
)

download = not Path("./celeb_a").exists()

train_ds = CelebA(
    root="./celeb_a", split="train", download=download, transform=transforms
)
val_ds = CelebA(
    root="./celeb_a", split="valid", download=download, transform=transforms
)

print("train_ds length: ", len(train_ds))
print("val_ds length: ", len(val_ds))

batch_size = 512
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

model = VAE(
    input_size=input_size,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    train_dl = iter(train_dl)
    val_dl = iter(val_dl)
    for step in range(len(train_dl) // 2):
        x_0, _ = next(train_dl)
        x_1, _ = next(train_dl)

        x_0 = x_0.to(DEVICE)
        x_1 = x_1.to(DEVICE)

        y_hat_0, mu_0, log_var_0 = model.forward(x_0)

        loss_0 = model.loss_function(
            y_hat=y_hat_0,
            x=x_0,
            mu=mu_0,
            log_var=log_var_0,
        )

        model_snapshot = copy.deepcopy(model)

        optimizer.zero_grad()
        loss_0.backward()
        optimizer.step()

        with torch.no_grad():
            y_hat_1a, mu_1a, log_var_1a = model(x_1)
            loss_1a = model.loss_function(
                y_hat=y_hat_1a,
                x=x_1,
                mu=mu_1a,
                log_var=log_var_1a,
            )

            y_hat_1b, mu_1b, log_var_1b = model_snapshot(x_1)
            loss_1b = model_snapshot.loss_function(
                y_hat=y_hat_1b,
                x=x_1,
                mu=mu_1b,
                log_var=log_var_1b,
            )

            wandb.log(
                {
                    "TRAIN Initial Loss": loss_0.item(),
                    "TRAIN Altered Model Future Loss": loss_1a.item(),
                    "TRAIN Snapshot Model Future Loss": loss_1b.item(),
                },
                sync_dist=True,
            )

        for x, _ in val_dl:
            x = x.to(DEVICE)
            y_hat, mu, log_var = model(x)
            loss = model.loss_function(y_hat, x, mu, log_var)
            wandb.log({"VAL Loss": loss.item()}, sync_dist=True)

        reconstructions_path = Path(LOG_DIR) / "Reconstructions"
        reconstructions_path.mkdir(exist_ok=True, parents=True)
        vutils.save_image(
            y_hat.data,
            reconstructions_path
            / f"recons_epoch_{epoch}.png",
            normalize=True,
            nrow=12,
        )


torch.save(model.state_dict(), Path(LOG_DIR) / "final_weights.pt")
