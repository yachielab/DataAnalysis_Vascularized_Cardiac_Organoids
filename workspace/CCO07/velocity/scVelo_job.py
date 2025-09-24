import scvelo as scv
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

WORK_DIR = "/home/herbert/PyProjects/cocultured_organ/workspace/CCO07"

loom_file_names = ['BVO', 'HO', 'CM3O', 'BHO', 'CM3B']
adata_file_names = ['VO', 'HO', 'CA', 'VHO', 'VCO']

# full_palette = {
#     'CM_0': '#1f77b4',  # blue
#     'CM_1': '#6baed6',  # lighter blue
#     'CM_2': '#08306b',  # dark navy blue
#     'CM_3': '#9ecae1',  # pale blue
#     'EN_H': '#d62728',  # red
#     'EN_V': '#ff9896',  # pinkish red
#     'PR_1': '#2ca02c',  # green
#     'PR_2': '#98df8a',  # light green
#     'FB':   '#9467bd',  # purple
#     'SM':   '#ff7f0e',  # orange
#     'EP':   '#8c564b',  # brown
#     'UN':   '#7f7f7f',  # grey (unassigned/unknown)
# }

full_palette = {
    'CM_1': '#1f77b4',  # blue
    'CM_2': '#6baed6',  # lighter blue
    'CM_3': '#08306b',  # dark navy blue
    'CM_4': '#9ecae1',  # pale blue
    'EC_1': '#d62728',  # strong red
    'EC_2': '#ff9896',  # light pinkish red
    'EC_3': '#a50f15',  # deep crimson red
    'PR_1': '#2ca02c',  # green
    'PR_2': '#2ca02c',  # green
    'PR_3': '#2ca02c',  # green
    'PR_4': '#2ca02c',  # green
    'PR_5': '#2ca02c',  # green
    'PR_6': '#2ca02c',  # green
    'PR_7': '#2ca02c',  # green
    'PR_8': '#2ca02c',  # green
    'FB':   '#9467bd',  # purple
    'SM':   '#ff7f0e',  # orange
    'EP':   '#8c564b',  # brown
    'UN':   '#7f7f7f',  # grey (unassigned/unknown)
}

xlim_full = [-12, 20]
xlim_cm = [5, 20]
ylim_full = [-7, 18]
ylim_cm = [5, 14]

for i in range(0, 5):

    loom_file = f'/home/herbert/PyProjects/cocultured_organ/data/240603/SC160__{loom_file_names[i]}-JW-052224/SC160__{loom_file_names[i]}-JW-052224.loom'
    ldata = sc.read(loom_file, cache=False)
    adata = sc.read_h5ad(f'{WORK_DIR}/adata/sample_recollect/{adata_file_names[i]}.h5ad')
    adata = scv.utils.merge(adata, ldata)

    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=50, n_neighbors=50)
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)
    palette = [full_palette[cat] for cat in adata.obs['cell_type_subcluster'].cat.categories]
    
    scv.pl.velocity_embedding_stream(
        adata, 
        basis='umap', 
        color='cell_type_subcluster', 
        palette=palette, 
        xlim=xlim_full,
        ylim=ylim_full,
        save=f"{WORK_DIR}/velocity/{adata_file_names[i]}.png"
    )
    
    adata_cm = adata[adata.obs["cell_type"] == "CM"]
    if adata_cm.n_obs == 0:
        continue
    scv.pl.velocity_embedding_stream(
        adata_cm, 
        basis='umap', 
        color='cell_type_subcluster', 
        palette=palette, 
        xlim=xlim_cm,
        ylim=ylim_cm,
        save=f"{WORK_DIR}/velocity/{adata_file_names[i]}.CM_only.png"
    )


# for i in range(0, 5):

#     loom_file = f'/home/herbert/PyProjects/cocultured_organ/data/240603/SC160__{loom_file_names[i]}-JW-052224/SC160__{loom_file_names[i]}-JW-052224.loom'
#     ldata = sc.read(loom_file, cache=False)
#     adata = sc.read_h5ad(f'{WORK_DIR}/adata/sample_recollect/{adata_file_names[i]}.h5ad')
#     if np.sum(adata.obs["cell_type"] == "CM") == 0:
#         continue
#     adata = adata[adata.obs["cell_type"] == "CM"]
#     adata = scv.utils.merge(adata, ldata)

#     scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
#     scv.pp.moments(adata, n_pcs=50, n_neighbors=50)
#     scv.tl.velocity(adata)
#     scv.tl.velocity_graph(adata)
#     palette = [full_palette[cat] for cat in adata.obs['cell_type_subcluster'].cat.categories]
#     scv.pl.velocity_embedding_stream(adata, basis='umap', color='cell_type_subcluster', palette=palette, save=f"{WORK_DIR}/velocity/{adata_file_names[i]}.CM_only.png")
    