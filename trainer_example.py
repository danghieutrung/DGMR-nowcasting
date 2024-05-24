from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


import torch
from torch.utils.data import TensorDataset, DataLoader

from loss import *
from model import DGMRNowcasting


X_train, y_train = torch.zeros((2, 4, 1, 256, 256)), torch.zeros((2, 18, 1, 256, 256))
X_test, y_test = torch.zeros((2, 4, 1, 256, 256)), torch.zeros((2, 18, 1, 256, 256))

# Create training and test datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoader instances for training and test datasets
batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = DGMRNowcasting()
optimizer = torch.optim.Adam(model.parameters())


def train_one_epoch(epoch_index, tb_writer):

    running_gloss, running_tloss, running_sloss = 0.0, 0.0, 0.0
    last_gloss, last_tloss, last_sloss = 0.0, 0.0, 0.0

    for i, train_data in enumerate(train_loader):

        # Every data instance is an input + obs pair
        X_train, y_train = train_data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        pred, T_pred, S_pred = model(X_train)

        # Compute the losses and their gradients
        g_loss = G_loss(pred, y_train, T_pred, S_pred)
        print("GG", g_loss)
        g_loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        _, T_pred, _, T_obs, _ = model(X_train, y_train)
        t_loss = T_loss(T_pred, T_obs)
        print("TT", t_loss)
        t_loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        _, _, S_pred, _, S_obs = model(X_train, y_train)
        s_loss = S_loss(S_pred, S_obs)
        print("SS", s_loss)
        s_loss.backward()
        optimizer.step()

        # Gather data and report
        running_gloss += g_loss.item()
        running_tloss += t_loss.item()
        running_sloss += s_loss.item()

        if i % 1000 == 999:
            last_gloss = running_gloss / 1000  # loss per batch
            last_tloss = running_tloss / 1000
            last_sloss = running_sloss / 1000

            print("  batch {} GEN loss: {}".format(i + 1, last_gloss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_gloss, tb_x)
            running_gloss, running_tloss, running_sloss = 0.0, 0.0, 0.0

    return last_gloss, last_tloss, last_sloss


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
    epoch_number = 0

    EPOCHS = 1

    best_gloss = 1_000_000.0

    for epoch in range(EPOCHS):
        print("EPOCH {}:".format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        avg_gloss_train, avg_tloss_train, avg_sloss_train = train_one_epoch(
            epoch_number, writer
        )

        running_gloss_test, running_tloss_test, running_sloss_test = 0.0, 0.0, 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                X_test, y_test = test_data
                pred, T_pred, S_pred, T_obs, S_obs = model(X_test, y_test)
                g_loss_test = G_loss(pred, y_test, T_pred, S_pred)
                t_loss_test = T_loss(T_pred, T_obs)
                s_loss_test = S_loss(S_pred, S_obs)

                running_gloss_test += g_loss_test
                running_tloss_test += t_loss_test
                running_sloss_test += s_loss_test

        avg_gloss_test = running_gloss_test / (i + 1)
        avg_tloss_test = running_tloss_test / (i + 1)
        avg_sloss_test = running_sloss_test / (i + 1)

        print("GEN LOSS train {} test {}".format(avg_gloss_train, avg_gloss_test))
        print("TEMP LOSS train {} test {}".format(avg_tloss_train, avg_tloss_test))
        print("SPA LOSS train {} test {}".format(avg_sloss_train, avg_sloss_test))

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
            model_path = "model_{}_{}".format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


if __name__ == "__main__":
    main()
