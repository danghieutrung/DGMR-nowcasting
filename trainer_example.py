from datetime import datetime
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MSELoss, L1Loss

from loss import *


X_train, y_train = torch.zeros((2, 4, 1, 256, 256)), torch.zeros((2, 18, 1, 256, 256))
X_test, y_test = torch.zeros((2, 4, 1, 256, 256)), torch.zeros((2, 18, 1, 256, 256))

# Create training and test datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoader instances for training and test datasets
batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

temporal_discriminator = TemporalDiscriminator()
spatial_discriminator = SpatialDiscriminator()
generator = Generator()

optimizerT = torch.optim.Adam(
    temporal_discriminator.parameters(), lr=0.0002, betas=(0, 0.999)
)
optimizerS = torch.optim.Adam(
    spatial_discriminator.parameters(), lr=0.0002, betas=(0, 0.999)
)
optimizerG = torch.optim.Adam(generator.parameters(), lr=0.00005, betas=(0, 0.999))

mse_loss, l1_loss = MSELoss(), L1Loss()


def train_one_epoch():
    running_gloss, running_tloss, running_sloss = 0.0, 0.0, 0.0

    for i, train_data in enumerate(train_loader):

        # Every data instance is an input + obs pair
        X_train, y_train = train_data
        pred = generator(X_train)
        pred_all, obs_all = torch.cat((X_train, pred.detach()), dim=1), torch.cat(
            (X_train, y_train), dim=1
        )

        # Train temporal discriminator
        temporal_discriminator.zero_grad()
        T_pred, T_obs = temporal_discriminator(pred_all), temporal_discriminator(
            obs_all
        )
        t_loss = T_loss(T_pred, T_obs)
        t_loss.backward()
        optimizerT.step()

        # Train spatial discriminator
        spatial_discriminator.zero_grad()
        S_pred, S_obs = spatial_discriminator(pred.detach()), spatial_discriminator(
            y_train
        )
        s_loss = S_loss(S_pred, S_obs)
        s_loss.backward()
        optimizerS.step()

        # Train generator
        generator.zero_grad()
        T_pred, S_pred = temporal_discriminator(pred_all), spatial_discriminator(
            pred_all
        )
        g_loss = G_loss(pred, y_train, T_pred, S_pred)
        g_loss.backward()
        optimizerG.step()

        # Gather data and report
        running_gloss += g_loss.item()
        running_tloss += t_loss.item()
        running_sloss += s_loss.item()

    avg_gloss = running_gloss / (i + 1)
    avg_tloss = running_tloss / (i + 1)
    avg_sloss = running_sloss / (i + 1)
    return avg_gloss, avg_tloss, avg_sloss


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
    epoch_number = 0

    EPOCHS = 60

    best_gloss = 1_000_000.0

    for epoch in range(EPOCHS):
        t = time.time()
        print("EPOCH {}:".format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        temporal_discriminator.train(True)
        spatial_discriminator.train(True)
        generator.train(True)

        avg_gloss_train, avg_tloss_train, avg_sloss_train = train_one_epoch()

        running_gloss_test, running_tloss_test, running_sloss_test = 0.0, 0.0, 0.0
        running_mse, running_l1 = 0.0, 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        temporal_discriminator.eval()
        spatial_discriminator.eval()
        generator.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                X_test, y_test = test_data

                pred = generator(X_test)
                pred_all, obs_all = torch.cat(
                    (X_test, pred.detach()), dim=1
                ), torch.cat((X_test, y_test), dim=1)

                T_pred, T_obs = temporal_discriminator(
                    pred_all
                ), temporal_discriminator(obs_all)
                S_pred, S_obs = spatial_discriminator(pred_all), spatial_discriminator(
                    obs_all
                )

                t_loss = T_loss(T_pred, T_obs)
                s_loss = S_loss(S_pred, S_obs)
                g_loss = G_loss(pred, y_test, T_pred, S_pred)
                l2 = mse_loss(pred, y_test)
                l1 = l1_loss(pred, y_test)

                running_tloss_test += t_loss
                running_sloss_test += s_loss
                running_gloss_test += g_loss
                running_mse += l2
                running_l1 += l1

        avg_gloss_test = running_gloss_test / (i + 1)
        avg_tloss_test = running_tloss_test / (i + 1)
        avg_sloss_test = running_sloss_test / (i + 1)
        avg_mse = running_mse / (i + 1)
        avg_l1 = running_l1 / (i + 1)

        print(
            "GEN LOSS train {:.4f} test {:.4f}".format(avg_gloss_train, avg_gloss_test)
        )
        print(
            "TMP LOSS train {:.4f} test {:.4f}".format(avg_tloss_train, avg_tloss_test)
        )
        print(
            "SPA LOSS train {:.4f} test {:.4f}".format(avg_sloss_train, avg_sloss_test)
        )
        print("MSE {:.4f} L1 {:.4f}".format(avg_mse, avg_l1))
        print("Epoch {}: {}s".format(epoch + 1, int(time.time() - t)))
        # break

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Training vs. Validation GEN Loss",
            {"Training": avg_gloss_train, "Validation": avg_gloss_test},
            epoch_number + 1,
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_gloss_test < best_gloss:
            best_gloss = avg_gloss_test
            generator_path = "saved_models/generator/model_{}_{}_{}".format(
                best_gloss, timestamp, epoch_number
            )
            torch.save(generator.state_dict(), generator_path)
            tdis_path = "saved_models/temporal_discriminator/model_{}_{}".format(
                timestamp, epoch_number
            )
            torch.save(temporal_discriminator.state_dict(), tdis_path)
            sdis_path = "saved_models/spatial_discriminator/model_{}_{}".format(
                timestamp, epoch_number
            )
            torch.save(spatial_discriminator.state_dict(), sdis_path)

        epoch_number += 1


if __name__ == "__main__":
    main()
