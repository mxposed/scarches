import torch
from sklearn.cluster import KMeans
from .trainer import Trainer
import torch.nn as nn
from torch.nn import functional as F

class tranVAETrainer(Trainer):
    """ScArches Unsupervised Trainer class. This class contains the implementation of the unsupervised CVAE/TRVAE
       Trainer.

           Parameters
           ----------
           model: trVAE
                Number of input features (i.e. gene in case of scRNA-seq).
           adata: : `~anndata.AnnData`
                Annotated data matrix.
           condition_key: String
                column name of conditions in `adata.obs` data frame.
           cell_type_key: String
                column name of celltypes in `adata.obs` data frame.
           train_frac: Float
                Defines the fraction of data that is used for training and data that is used for validation.
           batch_size: Integer
                Defines the batch size that is used during each Iteration
           n_samples: Integer or None
                Defines how many samples are being used during each epoch. This should only be used if hardware resources
                are limited.
           clip_value: Float
                If the value is greater than 0, all gradients with an higher value will be clipped during training.
           weight decay: Float
                Defines the scaling factor for weight decay in the Adam optimizer.
           alpha_iter_anneal: Integer or None
                If not 'None', the KL Loss scaling factor will be annealed from 0 to 1 every iteration until the input
                integer is reached.
           alpha_epoch_anneal: Integer or None
                If not 'None', the KL Loss scaling factor will be annealed from 0 to 1 every epoch until the input
                integer is reached.
           use_early_stopping: Boolean
                If 'True' the EarlyStopping class is being used for training to prevent overfitting.
           early_stopping_kwargs: Dict
                Passes custom Earlystopping parameters.
           use_stratified_sampling: Boolean
                If 'True', the sampler tries to load equally distributed batches concerning the conditions in every
                iteration.
           use_stratified_split: Boolean
                If `True`, the train and validation data will be constructed in such a way that both have same distribution
                of conditions in the data.
           monitor: Boolean
                If `True', the progress of the training will be printed after each epoch.
           n_workers: Integer
                Passes the 'n_workers' parameter for the torch.utils.data.DataLoader class.
           seed: Integer
                Define a specific random seed to get reproducable results.
        """
    def __init__(
            self,
            model,
            adata,
            eta: float = 1,
            tau: float = 0.2,
            **kwargs
    ):
        super().__init__(model, adata, **kwargs)
        self.landmarks = None
        self.eta = eta
        self.tau = tau

    def before_loop(self):
        self.landmarks = self.initialize_labelled_landmarks()
        l, a = self.init_landmark_loss(self.landmarks)
        print("LOSS:", l)
        print("ACCURACY:", a)

    def loss(self, total_batch=None):
        latent, recon_loss, kl_loss, mmd_loss = self.model(**total_batch)
        loss = recon_loss + self.calc_alpha_coeff()*kl_loss + mmd_loss
        landmark_loss, landmark_accuracy = self.landmark_loss(
            latent,
            self.landmarks,
            total_batch["celltype"]
        )
        loss += self.eta * landmark_loss
        self.iter_logs["loss"].append(loss)
        self.iter_logs["unweighted_loss"].append(recon_loss + kl_loss + mmd_loss )
        self.iter_logs["recon_loss"].append(recon_loss)
        self.iter_logs["kl_loss"].append(kl_loss)
        self.iter_logs["landmark_loss"].append(landmark_loss)
        self.iter_logs["landmark_accuracy_loss"].append(landmark_accuracy)
        if self.model.use_mmd:
            self.iter_logs["mmd_loss"].append(mmd_loss)
        return loss

    def on_epoch_end(self):
        latents = []
        indices = torch.arange(self.train_data.data.size(0), device=self.device)
        subsampled_indices = indices.split(self.batch_size)
        for batch in subsampled_indices:
            latent = self.model.get_latent(
                self.train_data.data[batch, :].to(self.device),
                self.train_data.conditions[batch].to(self.device)
            )
            latents += [latent]
        latent = torch.cat(latents)
        self.landmarks = self.update_labelled_landmarks(
            latent,
            self.train_data.cell_types,
            self.landmarks,
            self.tau,
        )
        super().on_epoch_end()

    def initialize_labelled_landmarks(self):
        latents = []
        labels = []
        indices = torch.arange(self.train_data.data.size(0), device=self.device)
        subsampled_indices = indices.split(self.batch_size)
        for batch in subsampled_indices:
            with torch.no_grad():
                latent = self.model.get_latent(
                    self.train_data.data[batch, :].to(self.device),
                    self.train_data.conditions[batch].to(self.device)
                )
                label = self.train_data.cell_types[batch].to(self.device)
            latents += [latent]
            labels += [label]
        latent = torch.cat(latents)
        label = torch.cat(labels)
        '''
        k_means = KMeans(n_clusters=self.model.n_cell_types, random_state=0).fit(latent.cpu())

        landmarks = torch.zeros(
            size=(self.model.n_cell_types, self.model.latent_dim),
            device=self.device,
            requires_grad=True,
        )
        with torch.no_grad():
            landmarks.copy_(torch.tensor(k_means.cluster_centers_, device=self.device))
        '''
        landmarks = self.update_labelled_landmarks(latent, label, None, self.tau)
        return landmarks

    def update_labelled_landmarks(self, latent, labels, previous_landmarks, tau):
        with torch.no_grad():
            unique_labels = torch.unique(labels, sorted=True)
            landmarks_mean = None
            for value in unique_labels:
                indices = labels.eq(value).nonzero()
                landmark = latent[indices].mean(0)
                landmarks_mean = torch.cat([landmarks_mean, landmark]) if landmarks_mean is not None else landmark
            '''
            class_indices = list(map(lambda y: labels.eq(y).nonzero(), unique_labels))

            landmarks_mean = torch.stack([latent[class_index].mean(0) for class_index in class_indices]).squeeze()
            '''
            if previous_landmarks is None or tau == 0:
                return landmarks_mean

            previous_landmarks_sum = previous_landmarks.sum(0)
            n_landmarks = previous_landmarks.shape[0]
            landmarks_distance_partial = (tau / (n_landmarks - 1)) * torch.stack(
                [previous_landmarks_sum - landmark for landmark in previous_landmarks])
            landmarks = (1 / (1 - tau)) * (landmarks_mean - landmarks_distance_partial)

        return landmarks

    def euclidean_dist(self, x, y):
        """
        Compute euclidean distance between two tensors
        """
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

    def landmark_loss(self, latent, landmarks, labels):
        n_samples = latent.shape[0]
        unique_labels = torch.unique(labels, sorted=True)
        distances = self.euclidean_dist(latent, landmarks)
        loss = None
        for value in unique_labels:
            indices = labels.eq(value).nonzero()
            label_loss = distances[indices, value].sum(0)
            loss = torch.cat([loss, label_loss]) if loss is not None else label_loss
        loss = loss.sum() / n_samples

        _, y_pred = torch.max(-distances, dim=1)

        accuracy = y_pred.eq(labels.squeeze()).float().mean()

        return loss, accuracy

    def init_landmark_loss(self, landmarks):
        latents = []
        labels = []
        indices = torch.arange(self.train_data.data.size(0), device=self.device)
        subsampled_indices = indices.split(self.batch_size)
        for batch in subsampled_indices:
            with torch.no_grad():
                latent = self.model.get_latent(
                    self.train_data.data[batch, :].to(self.device),
                    self.train_data.conditions[batch].to(self.device)
                )
                label = self.train_data.cell_types[batch].to(self.device)
            latents += [latent]
            labels += [label]
        latent = torch.cat(latents)
        label = torch.cat(labels)
        n_samples = latent.shape[0]
        unique_labels = torch.unique(label, sorted=True)
        distances = self.euclidean_dist(latent, landmarks)
        loss = None
        for value in unique_labels:
            indices = label.eq(value).nonzero()
            label_loss = distances[indices, value].sum(0)
            loss = torch.cat([loss, label_loss]) if loss is not None else label_loss
        loss = loss.sum() / n_samples

        _, y_pred = torch.max(-distances, dim=1)

        accuracy = y_pred.eq(label.squeeze()).float().mean()

        return loss, accuracy