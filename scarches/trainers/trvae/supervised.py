import torch
from sklearn.cluster import KMeans
from .trainer import Trainer
from ._utils import make_dataset, euclidean_dist


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
            n_clusters: int = None,
            eta: float = 1,
            tau: float = 0,
            labeled_indices: list = None,
            **kwargs
    ):
        super().__init__(model, adata, **kwargs)
        self.landmarks_labeled = None
        self.landmarks_unlabeled = None
        self.eta = eta
        self.tau = tau
        if labeled_indices is None:
            self.labeled_indices = range(len(adata))
        else:
            self.labeled_indices = labeled_indices
        self.update_labeled_indices(self.labeled_indices)
        self.n_clusters = self.model.n_cell_types if n_clusters is None else n_clusters
        self.lndmk_optim = None

    def update_labeled_indices(self, labeled_indices):
        self.labeled_indices = labeled_indices
        self.train_data, self.valid_data = make_dataset(
            self.adata,
            train_frac=self.train_frac,
            use_stratified_split=self.use_stratified_split,
            condition_key=self.condition_key,
            cell_type_key=self.cell_type_key,
            condition_encoder=self.model.condition_encoder,
            cell_type_encoder=self.model.cell_type_encoder,
            use_normalized=self.use_normalized,
            labeled_indices=self.labeled_indices,
        )

    def before_loop(self, lr, eps):
        self.initialize_landmarks()
        if 0 in self.train_data.labeled_vector.unique().tolist():
            self.lndmk_optim = torch.optim.Adam(
                params=self.landmarks_unlabeled,
                lr=lr,
                eps=eps,
                weight_decay=self.weight_decay
            )

    def loss(self, total_batch=None):
        latent, recon_loss, kl_loss, mmd_loss = self.model(**total_batch)
        trvae_loss = recon_loss + self.calc_alpha_coeff()*kl_loss + mmd_loss

        # Calculate classifier loss for labeled/unlabeled data
        label_categories = total_batch["labeled"].unique().tolist()
        landmark_loss = None
        unlabeled_loss = torch.tensor(0, device=self.device, requires_grad=False)
        labeled_loss = torch.tensor(0, device=self.device, requires_grad=False)
        if 0 in label_categories:
            unlabeled_loss, _ = self.landmark_unlabeled_loss(
                latent[total_batch['labeled'] == 0],
                torch.stack(self.landmarks_unlabeled).squeeze(),
                self.tau,
            )
            landmark_loss = unlabeled_loss
        if 1 in label_categories:
            labeled_loss, labeled_accuracy = self.landmark_labeled_loss(
                latent[total_batch['labeled'] == 1],
                self.landmarks_labeled,
                total_batch["celltype"][total_batch['labeled'] == 1],
            )
            landmark_loss = labeled_loss if landmark_loss is None else landmark_loss + labeled_loss
        classifier_loss = self.eta * landmark_loss

        loss = trvae_loss + classifier_loss
        self.iter_logs["loss"].append(loss)
        self.iter_logs["unweighted_loss"].append(recon_loss + kl_loss + mmd_loss + landmark_loss)
        self.iter_logs["trvae_loss"].append(trvae_loss)
        self.iter_logs["classifier_loss"].append(classifier_loss)
        self.iter_logs["unlabeled_loss"].append(unlabeled_loss)
        self.iter_logs["labeled_loss"].append(labeled_loss)
        return loss

    def on_epoch_end(self):
        self.model.eval()
        # Update landmark positions
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
        label_categories = self.train_data.labeled_vector.unique().tolist()
        if 0 in label_categories:
            for landmk in self.landmarks_unlabeled:
                landmk.requires_grad = True
            self.lndmk_optim.zero_grad()
            update_loss, args_count = self.landmark_unlabeled_loss(
                latent[self.train_data.labeled_vector == 0],
                torch.stack(self.landmarks_unlabeled).squeeze(),
                self.tau,
            )
            update_loss.backward()
            self.lndmk_optim.step()
            for landmk in self.landmarks_unlabeled:
                landmk.requires_grad = False

        if 1 in label_categories:
            self.landmarks_labeled = self.update_labeled_landmarks(
                latent[self.train_data.labeled_vector == 1],
                self.train_data.cell_types[self.train_data.labeled_vector == 1],
                self.landmarks_labeled,
                self.tau,
            )
        self.model.train()

        super().on_epoch_end()

    def after_loop(self):
        self.model.landmarks_labeled = self.landmarks_labeled
        if 0 in self.train_data.labeled_vector.unique().tolist():
            self.model.landmarks_unlabeled = torch.stack(self.landmarks_unlabeled).squeeze()

    def initialize_landmarks(self):
        # Compute Latent of whole train data
        indices = torch.arange(self.train_data.data.size(0), device=self.device)
        subsampled_indices = indices.split(self.batch_size)
        latents = []
        for batch in subsampled_indices:
            with torch.no_grad():
                latent = self.model.get_latent(
                    self.train_data.data[batch, :].to(self.device),
                    self.train_data.conditions[batch].to(self.device)
                )
            latents += [latent]
        latent = torch.cat(latents)

        # Init labeled Landmarks if labeled data existent
        if 1 in self.train_data.labeled_vector.unique().tolist():
            self.landmarks_labeled = self.update_labeled_landmarks(
                latent[self.train_data.labeled_vector == 1],
                self.train_data.cell_types[self.train_data.labeled_vector == 1],
                None,
                self.tau
            )

        # Init unlabeled Landmarks if unlabeled data existent
        if 0 in self.train_data.labeled_vector.unique().tolist():
            self.landmarks_unlabeled = torch.zeros(
                size=(self.n_clusters, self.model.latent_dim),
                device=self.device,
                requires_grad=True,
            )
            self.landmarks_unlabeled = [
                torch.zeros(
                    size=(1, self.model.latent_dim),
                    requires_grad=True,
                    device=self.device)
                for _ in range(self.n_clusters)
            ]
            k_means = KMeans(n_clusters=self.n_clusters,
                             random_state=0).fit(latent[self.train_data.labeled_vector == 0].cpu())
            k_means_lndmk = torch.tensor(k_means.cluster_centers_, device=self.device)
            with torch.no_grad():
                [self.landmarks_unlabeled[i].copy_(k_means_lndmk[i, :]) for i in range(k_means_lndmk.shape[0])]

    def update_labeled_landmarks(self, latent, labels, previous_landmarks, tau):
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

    def landmark_labeled_loss(self, latent, landmarks, labels):
        n_samples = latent.shape[0]
        unique_labels = torch.unique(labels, sorted=True)
        distances = euclidean_dist(latent, landmarks)
        loss = None
        for value in unique_labels:
            indices = labels.eq(value).nonzero()
            label_loss = distances[indices, value].sum(0)
            loss = torch.cat([loss, label_loss]) if loss is not None else label_loss
        loss = loss.sum() / n_samples

        _, y_pred = torch.max(-distances, dim=1)

        accuracy = y_pred.eq(labels.squeeze()).float().mean()

        return loss, accuracy

    def unlabeled_loss_basic(self, latent, landmarks):
        dists = euclidean_dist(latent, landmarks)
        min_dist = torch.min(dists, 1)

        y_hat = min_dist[1]
        args_uniq = torch.unique(y_hat, sorted=True)
        args_count = torch.stack([(y_hat == x_u).sum() for x_u in args_uniq])

        min_dist = min_dist[0]  # get_distances

        loss_val = torch.stack([min_dist[y_hat == idx_class].mean(0) for idx_class in args_uniq]).mean()

        return loss_val, args_count

    def landmark_unlabeled_loss(self, latent, landmarks, tau):
        loss_val_test, args_count = self.unlabeled_loss_basic(latent, landmarks)
        if tau > 0:
            dists = euclidean_dist(landmarks, landmarks)
            nproto = landmarks.shape[0]
            loss_val2 = - torch.sum(dists) / (nproto * nproto - nproto)

            loss_val_test += tau * loss_val2

        return loss_val_test, args_count
