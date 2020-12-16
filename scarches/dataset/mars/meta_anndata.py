import numpy as np
import scanpy as sc

from scarches.dataset.trvae import AnnotatedDataset
from scarches.dataset.trvae._utils import label_encoder


class MetaAnnotatedDataset(object):
    def __init__(self,
                 adata: sc.AnnData,
                 task_key: str,
                 meta_test_task: str = None,
                 task_encoder=None,
                 cell_type_key=None,
                 cell_type_encoder=None,
                 size_factors_key=None,
                 ):
        self.adata = adata

        self.meta_test_task = meta_test_task

        self.task_key = task_key
        self.unique_tasks = list(np.unique(self.adata.obs[task_key]))
        _, self.task_encoder = label_encoder(self.adata,
                                             encoder=task_encoder,
                                             condition_key=self.task_key)

        self.meta_train_tasks = [task for task in self.unique_tasks if task != self.meta_test_task]
        self.meta_test_tasks = [task for task in self.unique_tasks if task == self.meta_test_task]

        self.cell_type_encoder = cell_type_encoder
        self.cell_type_key = cell_type_key
        self.unique_cell_types = np.unique(self.adata.obs[cell_type_key]) if self.cell_type_key else None

        if self.cell_type_encoder is None:
            self.cell_type_encoder = {k: v for k, v in zip(self.unique_cell_types, range(len(self.unique_cell_types)))}

        self.size_factors_key = size_factors_key

        self.meta_train_tasks_adata = [AnnotatedDataset(
            adata=self.adata[self.adata.obs[self.task_key] == task],
            condition_key=self.task_key,
            condition_encoder=self.task_encoder,
            cell_type_key=self.cell_type_key,
            cell_type_encoder=self.cell_type_encoder,
        ) for task in self.meta_train_tasks]

        if self.meta_test_task is not None:
            self.meta_test_task_adata = AnnotatedDataset(
                adata=self.adata[self.adata.obs[self.task_key] == self.meta_test_task],
                condition_key=self.task_key,
                condition_encoder=self.task_encoder,
                cell_type_key=self.cell_type_key,
                cell_type_encoder=None,
                size_factors_key=size_factors_key,
            )
        else:
            self.meta_test_task_adata = None
