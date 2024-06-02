from components.Generator import Generator
from components.TemporalDiscriminator import TemporalDiscriminator
from components.SpatialDiscriminator import SpatialDiscriminator

from loss import *
import time

import torch
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss


class Trainer():
		def __init__(self, cfg):
				super().__init()
				self.cfg = cfg

				self.forecast_steps = cfg.models.generator.forecast_steps
				self.epochs = cfg.optim.params.epochs

				self.generator = Generator(self.forecast_steps)
				self.spatial_discriminator = SpatialDiscriminator(
						n_frame=cfg.spatial_discriminator.n_frame
				)
				self.temporal_discriminator = TemporalDiscriminator(
						crop_size=cfg.models.temporal_discriminator.crop_size
				)

				self.optimizerG = Adam(
						self.generator.parameters(),
						lr=cfg.optim.generator.lr,
						betas=(cfg.optim.generator.b1, cfg.optim.generator.b2),
				)
				self.optimizerS = Adam(
						self.spatial_discriminator.parameters(),
						lr=cfg.optim.spatial_discriminator.lr,
						betas=(
								cfg.optim.spatial_discriminator.b1,
								cfg.optim.spatial_discriminator.b2,
						),
				)
				self.optimizerT = Adam(
						self.temporal_discriminator.parameters(),
						lr=cfg.optim.temporal_discriminator.lr,
						betas=(
								cfg.optim.temporal_discriminator.b1,
								cfg.optim.temporal_discriminator.b2,
						),
				)

		def fit(self, train_loader, val_loader, epochs):
				for epoch in range(epochs):
				
						start_epoch = time.time()

						avg_gloss_train, avg_tloss_train, avg_sloss_train, avg_mse_train, avg_l1_train = self._train(train_loader)

						avg_gloss_test, avg_tloss_test, avg_sloss_test, avg_mse_test, avg_l1_test = self._validate(val_loader)

						print(
						"GEN LOSS train {:.4f} test {:.4f}".format(avg_gloss_train, avg_gloss_test)
				)
						print(
								"TMP LOSS train {:.4f} test {:.4f}".format(avg_tloss_train, avg_tloss_test)
						)
						print(
								"SPA LOSS train {:.4f} test {:.4f}".format(avg_sloss_train, avg_sloss_test)
						)
						print("MSE LOSS train {:.4f} test {:.4f}".format(avg_mse_train, avg_mse_test))
						print("MAE LOSS train {:.4f} test {:.4f}".format(avg_l1_train, avg_l1_test))
						print("Epoch {}: {}s".format(epoch + 1, int(time.time() - start_epoch)))



		def _train(self, train_loader):
				self.temporal_discriminator.train(True)
				self.spatial_discriminator.train(True)
				self.generator.train(True)

				running_gloss, running_tloss, running_sloss = 0.0, 0.0, 0.0
				running_mse, running_l1 = 0.0, 0.0

				for i, train_data in enumerate(train_loader):

						# Every data instance is an input + obs pair
						X_train, y_train = train_data
						pred = self.generator(X_train)
						pred_all, obs_all = torch.cat((X_train, pred.detach()), dim=1), torch.cat(
								(X_train, y_train), dim=1
						)

						# Train temporal discriminator
						self.temporal_discriminator.zero_grad()
						T_pred, T_obs = self.temporal_discriminator(
								pred_all
						), self.temporal_discriminator(obs_all)
						t_loss = T_loss(T_pred, T_obs)
						t_loss.backward()
						self.optimizerT.step()

						# Train spatial discriminator
						self.spatial_discriminator.zero_grad()
						S_pred, S_obs = self.spatial_discriminator(
								pred.detach()
						), self.spatial_discriminator(y_train)
						s_loss = S_loss(S_pred, S_obs)
						s_loss.backward()
						self.optimizerS.step()

						# Train generator
						self.generator.zero_grad()
						T_pred, S_pred = self.temporal_discriminator(
								pred_all
						), self.spatial_discriminator(pred_all)
						g_loss = G_loss(pred, y_train, T_pred, S_pred)
						g_loss.backward()
						self.optimizerG.step()

						# Gather data and report
						l2 = MSELoss()(pred, y_train)
						l1 = L1Loss()(pred, y_train)

						running_gloss += g_loss.item()
						running_tloss += t_loss.item()
						running_sloss += s_loss.item()
						running_mse += l2.item()
						running_l1 += l1.item()

				avg_gloss = running_gloss / (i + 1)
				avg_tloss = running_tloss / (i + 1)
				avg_sloss = running_sloss / (i + 1)
				avg_mse = running_mse / (i + 1)
				avg_l1 = running_l1 / (i + 1)

				return avg_gloss, avg_tloss, avg_sloss, avg_mse, avg_l1

		def _validate(self, test_loader):
				self.temporal_discriminator.eval(True)
				self.spatial_discriminator.eval(True)
				self.generator.eval(True)

				running_gloss, running_tloss, running_sloss = 0.0, 0.0, 0.0
				running_mse, running_l1 = 0.0, 0.0

				with torch.no_grad():
						for i, test_data in enumerate(test_loader):
								X_test, y_test = test_data

								pred = self.generator(X_test)
								pred_all, obs_all = torch.cat(
										(X_test, pred.detach()), dim=1
								), torch.cat((X_test, y_test), dim=1)

								T_pred, T_obs = self.temporal_discriminator(
										pred_all
								), self.temporal_discriminator(obs_all)
								S_pred, S_obs = self.spatial_discriminator(
										pred_all
								), self.spatial_discriminator(obs_all)

								t_loss = T_loss(T_pred, T_obs)
								s_loss = S_loss(S_pred, S_obs)
								g_loss = G_loss(pred, y_test, T_pred, S_pred)
								l2 = MSELoss()(pred, y_test)
								l1 = L1Loss()(pred, y_test)

								running_tloss += t_loss.item()
								running_sloss += s_loss.item()
								running_gloss += g_loss.item()
								running_mse += l2.item()
								running_l1 += l1.item()

								avg_gloss = running_gloss / (i + 1)
								avg_tloss = running_tloss / (i + 1)
								avg_sloss = running_sloss / (i + 1)
								avg_mse = running_mse / (i + 1)
								avg_l1 = running_l1 / (i + 1)

				return avg_gloss, avg_tloss, avg_sloss, avg_mse, avg_l1
