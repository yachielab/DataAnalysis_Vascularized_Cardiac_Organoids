import scanpy as sc
import matplotlib.pyplot as plt
import glob
import numpy as np
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import anndata
import os
import glob
import scipy

PREV_DIR = "/home/herbert/PyProjects/cocultured_organ/workspace/CCO05"
WORK_DIR = "/home/herbert/PyProjects/cocultured_organ/workspace/CCO06"

adata = sc.read_h5ad(f"{WORK_DIR}/adata/leiden_clustered/ALL.h5ad")
sc.external.pp.scrublet(adata, batch_key="batch")

adata = adata[adata.obs['predicted_doublet'] == False]

adata.write(f'{WORK_DIR}/adata/doublet_filtered/ALL.h5ad')

for ct in adata.obs['batch'].unique().to_list():
    if ct == "BHO":
        new_ct = "VHO"
    elif ct == "BVO":
        new_ct = "VO"
    elif ct == "CM3B":
        new_ct = "VCO"
    elif ct == "CM3O":
        new_ct = "CA"
    elif ct == "HO":
        new_ct = "HO"
    else:
        raise ValueError(f"Impossible celltype: {ct}")
    adata[adata.obs['batch'] == ct].write(f'{WORK_DIR}/adata/doublet_filtered/{new_ct}.h5ad')
