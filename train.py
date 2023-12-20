import copy
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import wandb

from celeb_a import CelebA
from vae import VAE


def copy_params_and_grad_from(model_a: nn.Module, model_b: nn.Module) -> nn.Module:
    with torch.no_grad():
        for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
            param_a = copy.deepcopy(param_b)


wandb.init(project="basic_training")

EPOCHS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = Path("./logs")
BATCH_SIZE = 64
DO_INTER_VAL = True

if not LOG_DIR.exists():
    LOG_DIR.mkdir(parents=True)

transforms = T.Compose(
    [T.RandomHorizontalFlip(), T.Resize(128), T.CenterCrop(128), T.ToTensor()]
)

download = not Path("./celeb_a/celeba/img_align_celeba").exists()

train_ds = CelebA(
    root="./celeb_a",
    split="train",
    download=download,
    transform=transforms,
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

    n_rejected = 0
    for step, x in enumerate(progress_bar):
        progress_bar.set_description_str(
            f"Epoch {epoch} | Train Loss: {train_loss:.5f} | Val Loss: {validation_loss:.5f} | Rejected: {n_rejected/(step+1):.2f}"
        )

        x = x.to(DEVICE)
        x_0, x_1 = x[: BATCH_SIZE // 2].detach(), x[BATCH_SIZE // 2:].detach()

        y_hat_0, mu_0, log_var_0 = model.forward(x_0)

        loss_0 = model.loss_function(
            y_hat=y_hat_0,
            x=x_0,
            mu=mu_0,
            log_var=log_var_0,
        )

        train_loss = (
            (0.99 * train_loss) + (0.01 * loss_0.item()) if step > 0 else loss_0.item()
        )

        if DO_INTER_VAL:
            model_snapshot = copy.deepcopy(model)

        optimizer.zero_grad()
        loss_0.backward()
        optim_state_dict = copy.deepcopy(optimizer.state_dict())
        optimizer.step()

        if DO_INTER_VAL:
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

            diff = loss_1a - loss_1b

            if diff > 0:  # if this update worsed generalizability, revert.
                n_rejected += 1
                optimizer.load_state_dict(optim_state_dict)
                copy_params_and_grad_from(model, model_snapshot)

        if step % 25 == 0:
            wandb.log(
                {
                    "TRAIN Initial Loss": loss_0.item(),
                    "TRAIN Altered Model Future Loss": loss_1a.item()
                    if DO_INTER_VAL
                    else 0.0,
                    "TRAIN Snapshot Model Future Loss": loss_1b.item()
                    if DO_INTER_VAL
                    else 0.0,
                },
            )

    all_val_losses = []
    for x in tqdm(val_dl, leave=False):
        x = x.to(DEVICE)
        y_hat, mu, log_var = model(x)
        loss = model.loss_function(y_hat, x, mu, log_var)
        all_val_losses.append(loss.item())

    validation_loss = sum(all_val_losses) / len(all_val_losses)
    wandb.log({"VAL Loss": validation_loss})

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
