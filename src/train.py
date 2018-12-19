import random
import warnings
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import Saver, Saver_Encoder
from model import SpeechEggEncoder, EGGEncoder, Discriminator
from loader import train_validate_test_loader

# INFO: Set random seeds
np.random.seed(42)
th.manual_seed(42)
th.cuda.manual_seed_all(42)
random.seed(42)


def train(
    model_G: nn.Module,
    model_D: nn.Module,
    encoder: nn.Module,
    optimizer_G: optim.Optimizer,
    optimizer_R: optim.Optimizer,
    optimizer_D: optim.Optimizer,
    train_data: DataLoader,
    use_cuda: bool = True,
    scheduler_G=None,
    scheduler_R=None,
    scheduler_D=None,
):
    device = "cuda" if th.cuda.is_available() and use_cuda else "cpu"

    model_G.train()
    model_D.train()
    encoder.eval()

    batches = len(train_data)
    D_loss_sum = 0
    D_real_prob = 0
    D_fake_prob = 0
    G_loss_sum = 0
    loss_sum = 0
    reconstruction_loss = 0

    model_G.to(device)
    model_D.to(device)
    encoder.to(device)

    for data, egg_data in train_data:
        if scheduler_G is not None:
            scheduler_G.step()
            scheduler_R.step()
            scheduler_D.step()

        data, egg_data = data.to(device), egg_data.to(device)

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        optimizer_R.zero_grad()

        batch_size = data.shape[0]
        ones_label = th.ones(batch_size, 1).to(device)
        zeros_label = th.zeros(batch_size, 1).to(device)

        # Optimize model_D
        true = encoder(egg_data)
        _, embeddings_ = model_G(data)
        D_real = model_D(true)
        D_fake = model_D(embeddings_)

        D_loss_real = F.binary_cross_entropy(D_real, ones_label)
        D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label)
        D_loss = D_loss_real + D_loss_fake

        D_real_prob += D_real.mean().item()
        D_fake_prob += D_fake.mean().item()

        # for i in model_G.parameters():
        #     i.requires_grad = False
        # for i in model_D.parameters():
        #     i.requires_grad = True
        D_loss.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        optimizer_R.zero_grad()

        # Optimize model_G
        # for i in model_G.parameters():
        #     i.requires_grad = True
        # for i in model_D.parameters():
        #     i.requires_grad = False
        _, embeddings_ = model_G(data)
        D_fake = model_D(embeddings_)

        G_loss = F.binary_cross_entropy(D_fake, ones_label)
        G_loss.backward()
        optimizer_G.step()
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        optimizer_R.zero_grad()

        reconstructions, _ = model_G(data)
        # loss_reconstruction = (egg_data * reconstructions).sum(dim=1) / (
        #     egg_data.norm(dim=1) * reconstructions.norm(dim=1)
        # )
        # loss_reconstruction = th.acos(loss_reconstruction) * 180 / np.pi
        # loss_reconstruction = loss_reconstruction.mean()

        loss_reconstruction = F.mse_loss(egg_data, reconstructions)
        loss_reconstruction.backward()
        optimizer_R.step()
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        optimizer_R.zero_grad()

        net_loss = loss_reconstruction + G_loss

        loss_sum += net_loss.item()
        reconstruction_loss += loss_reconstruction.item()

        D_loss_sum += D_loss
        G_loss_sum += G_loss

    del D_loss, G_loss
    th.cuda.empty_cache()

    return (
        loss_sum / batches,
        D_loss_sum / batches,
        G_loss_sum / batches,
        reconstruction_loss / batches,
        D_real_prob / batches,
        D_fake_prob / batches,
    )


def test(
    model_G: nn.Module,
    model_D: nn.Module,
    encoder: nn.Module,
    test_loader: DataLoader,
    use_cuda: bool = False,
):
    device = "cuda" if th.cuda.is_available() and use_cuda else "cpu"

    model_G.to(device)
    model_D.to(device)
    model_G.eval()
    model_D.eval()
    encoder.eval()

    batches = len(test_loader)
    D_loss_sum = 0
    G_loss_sum = 0
    loss_sum = 0
    D_real_prob = 0
    D_fake_prob = 0
    reconstruction_loss = 0

    with th.set_grad_enabled(False):
        for data, egg_data in test_loader:
            data, egg_data = data.to(device), egg_data.to(device)
            data.requires_grad_, egg_data.requires_grad_ = False, False

            batch_size = data.shape[0]
            ones_label = th.ones(batch_size, 1).to(device)
            zeros_label = th.zeros(batch_size, 1).to(device)

            # Test model_D
            true = encoder(egg_data)
            _, embeddings_ = model_G(data)
            D_real = model_D(true)
            D_fake = model_D(embeddings_)

            D_loss_real = F.binary_cross_entropy(D_real, ones_label)
            D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label)
            D_loss = D_loss_real + D_loss_fake

            D_real_prob += D_real.mean().item()
            D_fake_prob += D_fake.mean().item()

            # Test model_G
            reconstructions, embeddings_ = model_G(data)
            D_fake = model_D(embeddings_)

            # loss_reconstruction =             (egg_data * reconstructions).sum(dim=1) / (egg_data.norm(dim=1) * reconstructions.norm(dim=1))
            # loss_reconstruction = th.acos(
            #     loss_reconstruction
            # ) * 180 / np.pi +
            # F.mse_loss(
            #     egg_data, reconstructions, reduction="none"
            # ).sum(dim=1)
            # loss_reconstruction = loss_reconstruction.mean()

            loss_reconstruction = F.mse_loss(egg_data, reconstructions)

            G_loss = F.binary_cross_entropy(D_fake, ones_label)

            net_loss = loss_reconstruction + G_loss

            loss_sum += net_loss.item()
            reconstruction_loss += loss_reconstruction.item()

            D_loss_sum += D_loss
            G_loss_sum += G_loss

    del D_loss, G_loss
    th.cuda.empty_cache()

    print(
        "\nTest set: Test loss {:4.4} D_loss {:4.4} G_loss {:4.4} reconstruction loss {:4.4} Real D prob. {:4.4} Fake D prob. {:4.4}\n".format(
            loss_sum / batches,
            D_loss_sum / batches,
            G_loss_sum / batches,
            reconstruction_loss / batches,
            D_real_prob / batches,
            D_fake_prob / batches,
        )
    )


def main():
    train_data, test_data, _ = train_validate_test_loader(
        "../data/Childers/M/speech",
        "../data/Childers/M/egg",
        split={"train": 0.65, "validate": 0.15, "test": 0.2},
        batch_size=1,
        workers=2,
        stride={"train": 2, "validate": 20},
        pin_memory=False,
        # model_folder="data/childers_clean_data",
    )

    model_G = SpeechEggEncoder()
    model_D = Discriminator()
    save_model = Saver("checkpoints/vmodels/childers_clean_l2")

    encoder = EGGEncoder()
    save_encoder = Saver_Encoder("encoder")
    encoder, _, _ = save_encoder.load_checkpoint(encoder, file_name="epoch_65.pt")

    use_cuda = True
    epochs = 100

    optimizer_G = optim.Adam(list(model_G.parameters())[:12], lr=2e-3)
    optimizer_R = optim.Adam(model_G.parameters(), lr=2e-3)
    optimizer_D = optim.Adam(model_D.parameters(), lr=2e-3)
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, 10, 0.9)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, 10, 0.9)
    scheduler_R = optim.lr_scheduler.StepLR(optimizer_D, 10, 0.5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i in range(1, epochs + 1):

            net_loss, D_loss, G_loss, R_loss, D_real_prob, D_fake_prob = train(
                model_G,
                model_D,
                encoder,
                optimizer_G,
                optimizer_R,
                optimizer_D,
                train_data,
                use_cuda,
            )
            print(
                "Train loss {:4.4} D_loss {:4.4} G_loss {:4.4} reconstruction loss {:4.4} Real D prob. {:4.4} Fake D prob. {:4.4} @epoch {}".format(
                    net_loss, D_loss, G_loss, R_loss, D_real_prob, D_fake_prob, i
                )
            )
            if i % 5 == 0:
                checkpoint = save_model.create_checkpoint(
                    model_G,
                    model_D,
                    optimizer_G,
                    optimizer_R,
                    optimizer_D,
                    {"win": 100, "stride": 3},
                )

                save_model.save_checkpoint(
                    checkpoint, file_name="epoch_{}.pt".format(i), append_time=False
                )
                test(model_G, model_D, encoder, test_data, use_cuda)

            if scheduler_G is not None:
                scheduler_G.step()
                scheduler_D.step()
                scheduler_R.step()


if __name__ == "__main__":
    main()
