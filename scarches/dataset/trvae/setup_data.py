import scanpy as sc
import numpy as np
from ._utils import label_encoder


def setup_trvae_data(
        adata,
        use_normalized,
        condition_key=None,
        cell_type_key=None,
        condition_encoder=None,
        cell_type_encoder=None,
):
    if use_normalized:
        sc.pp.normalize_total(adata,
                              exclude_highly_expressed=True,
                              target_sum=1e4,
                              key_added='trvae_size_factors')
    else:
        size_factors = np.log(adata.X.sum(1))
        if len(size_factors.shape) < 2:
            size_factors = np.expand_dims(size_factors, axis=1)
        adata.obs['trvae_size_factors'] = size_factors
    if condition_key is not None and condition_encoder is not None:
        adata.obs['trvae_conditions'], _ = label_encoder(
            adata,
            encoder=condition_encoder,
            condition_key=condition_key)
    if cell_type_key is not None and cell_type_encoder is not None:
        adata.obs['trvae_cell_types'], _ = label_encoder(
            adata,
            encoder=cell_type_encoder,
            condition_key=cell_type_key)
    return adata
