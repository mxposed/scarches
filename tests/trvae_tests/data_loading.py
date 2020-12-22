from scarches.trainers.trvae._utils import make_dataset, custom_collate
import os
import scanpy as sc
import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np

labeled_batches = ['Pancreas inDrop', 'Pancreas CelSeq2', 'Pancreas CelSeq', 'Pancreas Fluidigm C1', 'Pancreas SS2']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
adata_all = sc.read(os.path.expanduser(f'~/Documents/benchmarking_datasets/pancreas_normalized.h5ad'))
adata = adata_all.raw.to_adata()
n_samples = len(adata)
print("UNPROCESSED ANNDATA:")
print(adata)
print(adata.obs.study.unique().tolist())
print(adata.obs.cell_type.unique().tolist())
labeled = np.array([s in labeled_batches for s in adata.obs["study"]]).nonzero()[0].tolist()

train_data, valid_data = make_dataset(
            adata,
            train_frac=0.9,
            use_stratified_split=True,
            condition_key="study",
            cell_type_key="cell_type",
            condition_encoder=None,
            cell_type_encoder=None,
            use_normalized=False,
            labeled_indices=labeled
        )

if 1 in train_data.labeled_vector.unique().tolist():
    print('1')
if 0 in train_data.labeled_vector.unique().tolist():
    print('0')
exit()
print("\nANNDATA AFTER MAKE DATASET")
print(adata)
print("\nLABELED CONDITIONS IN ANNDATA")
print(adata.obs.study[adata.obs['trvae_labeled'] == 1].unique().tolist())
print("\nLABELED CONDITIONS IN TRAINDATA")
print(train_data.conditions[train_data.labeled_vector == 1].unique().tolist())
print(train_data.condition_encoder)
print("\nLABELED CONDITIONS IN VALIDDATA")
print(valid_data.conditions[valid_data.labeled_vector == 1].unique().tolist())
print(valid_data.condition_encoder)

stratifier_weights = torch.tensor(train_data.stratifier_weights, device=device)

sampler = WeightedRandomSampler(
    stratifier_weights,
    num_samples=n_samples,
    replacement=True)
dataloader_train = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=512,
    sampler=sampler,
    collate_fn=custom_collate,
    num_workers=0)

for iter, batch_data in enumerate(dataloader_train):
    for key, batch in batch_data.items():
        batch_data[key] = batch.to(device)

    c0 = len(torch.nonzero(batch_data['batch'] == 0))
    c1 = len(torch.nonzero(batch_data['batch'] == 1))
    c2 = len(torch.nonzero(batch_data['batch'] == 2))
    c3 = len(torch.nonzero(batch_data['batch'] == 3))
    c4 = len(torch.nonzero(batch_data['batch'] == 4))

    y0 = len(torch.nonzero(batch_data['celltype'] == 0))
    y1 = len(torch.nonzero(batch_data['celltype'] == 1))
    y2 = len(torch.nonzero(batch_data['celltype'] == 2))
    y3 = len(torch.nonzero(batch_data['celltype'] == 3))
    y4 = len(torch.nonzero(batch_data['celltype'] == 4))
    y5 = len(torch.nonzero(batch_data['celltype'] == 5))
    y6 = len(torch.nonzero(batch_data['celltype'] == 6))
    y7 = len(torch.nonzero(batch_data['celltype'] == 7))

    labeled_batches = batch_data['batch'][batch_data['labeled']==1].unique().tolist()
    print(f"ITER: {iter+1}, \tLABELED BATCHES: {labeled_batches} \tCONDITION COUNTS: {c0} {c1} {c2} {c3} {c4} "
          f"CELLTYPE COUNTS: {y0} {y1} {y2} {y3} {y4} {y5} {y6} {y7}")