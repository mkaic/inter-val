from torch.utils.data import DataLoader
import torchvision.utils as vutils
from celeb_a import CelebA
from torchvision import transforms as T
from vae import VAE
import torch
from pathlib import Path
import wandb
import copy
from tqdm import tqdm

wandb.init(project="basic_training")

EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = Path("./logs")
BATCH_SIZE = 32

if not LOG_DIR.exists():
    LOG_DIR.mkdir(parents=True)

transforms = T.Compose(
    [T.RandomHorizontalFlip(), T.Resize(128), T.CenterCrop(128), T.ToTensor()]
)

download = not Path("./celeb_a/celeba/img_align_celeba").exists()

train_ds = CelebA(
    root="./celeb_a", split="train", download=download, transform=transforms
)
val_ds = CelebA(
    root="./celeb_a", split="valid", download=download, transform=transforms
)

print("train_ds length: ", len(train_ds))
print("val_ds length: ", len(val_ds))

model = VAE(
    input_size=128,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

validation_loss = 0.0
train_loss = 0.0

for epoch in range(EPOCHS):
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    progress_bar = tqdm(train_dl, leave=False)
    for step, x in enumerate(progress_bar):
        progress_bar.set_description_str(
            f"Epoch {epoch} | Train Loss: {train_loss:.5f} | Val Loss: {validation_loss:.5f}"
        )

        x = x.to(DEVICE)

        y_hat, mu, log_var = model.forward(x)

        loss = model.loss_function(
            y_hat=y_hat,
            x=x,
            mu=mu,
            log_var=log_var,
        )

        if step % 25 == 0:
            wandb.log(
                {
                    "Train Loss": loss.item(),
                },
                sync_dist=True,
            )

        train_loss = train_loss * 0.99 + loss.item() * 0.01 if step > 0 else loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    all_val_losses = []
    for x in tqdm(val_dl):
        x = x.to(DEVICE)
        y_hat, mu, log_var = model(x)
        loss = model.loss_function(y_hat, x, mu, log_var)
        all_val_losses.append(loss.item())

    validation_loss = sum(all_val_losses) / len(all_val_losses)
    wandb.log({"VAL Loss": validation_loss}, sync_dist=True)

    run_path = (
        Path(LOG_DIR) / "most_recent"
        if not wandb.run.name
        else Path(LOG_DIR) / wandb.run.name
    )
    run_path.mkdir(exist_ok=True, parents=True)

    vutils.save_image(
        y_hat.data,
        run_path / f"recons_epoch_{epoch}.png",
        normalize=True,
        nrow=12,
    )


torch.save(model.state_dict(), Path(LOG_DIR) / "final_weights.pt")
