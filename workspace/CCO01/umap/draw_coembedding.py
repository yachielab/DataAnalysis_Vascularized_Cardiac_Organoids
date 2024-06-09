import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata
import os
import glob

DATA_DIR = '/home/herbert/PyProjects/cocultured_organ/data/231221'
OUT_DIR = '/home/herbert/PyProjects/cocultured_organ/workspace/CCO01'

adata_bvo = sc.read_h5ad(f'{OUT_DIR}/processed/SC149__BVO-14D-JW-121123.h5ad')
adata_ho = sc.read_h5ad(f'{OUT_DIR}/processed/SC149__HO-14D-JW-121123.h5ad')
adata_cm3 = sc.read_h5ad(f'{OUT_DIR}/processed/SC149__CM3-14D-JW-121123.h5ad')

adata_all = adata_bvo.concatenate(adata_ho, adata_cm3, batch_key='batch')

batch_mapping = {
    '0': 'BVO',
    '1': 'HO',
    '2': 'CM3'
}
adata_all.obs['batch'] = adata_all.obs['batch'].map(batch_mapping)

for n_neighbors in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:

    sc.tl.pca(adata_all, svd_solver='arpack')
    sc.pp.neighbors(adata_all, n_pcs=50, n_neighbors=n_neighbors)
    sc.tl.umap(adata_all)

    fig, ax = plt.subplots(1, 1)
    sc.pl.umap(adata_all, color='batch', wspace=0.6, title=f'Co-Embedding, n_neighbors={n_neighbors}', ax=ax, show=False)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/umap/coembedding.n_neighbor{n_neighbors}.png')
    plt.close()