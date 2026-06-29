import scvelo as scv
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

WORK_DIR = "/home/herbert/PyProjects/cocultured_organ/workspace/CCO12"

loom_file_names = ['BVO', 'HO', 'CM3O', 'BHO', 'CM3B']
adata_file_names = ['VO', 'HO', 'ReAgg-CM', 'Fus-HO', 'VasHO']

# full_palette = {
#     "CM_1": (30/255, 120/255, 30/255),
#     "CM_2": (150/255, 230/255, 150/255),
#     "CM_3": (15/255, 80/255, 15/255),
#     "CM_4": (90/255, 190/255, 90/255),
    
#     "EC_1": (255/255, 200/255, 230/255),
#     "EC_2": (230/255, 80/255, 160/255),
#     "EC_3": (180/255, 30/255, 120/255),

#     "FB": (240/255, 180/255, 20/255),
#     "SM": (220/255, 160/255, 30/255),
    
#     "PR_1": (255/255, 250/255, 200/255),
#     "PR_2": (255/255, 245/255, 175/255),
#     "PR_3": (255/255, 240/255, 150/255),
#     "PR_4": (255/255, 235/255, 125/255),
#     "PR_5": (255/255, 230/255, 100/255),
#     "PR_6": (255/255, 220/255, 75/255),
#     "PR_7": (255/255, 210/255, 50/255),
#     "PR_8": (255/255, 200/255, 25/255),

#     "EP": (60/255, 180/255, 220/255),

#     "UN": (190/255, 190/255, 190/255),
# }

colormap = {
    "CM": (60/255, 170/255, 60/255),
    "EC": (230/255, 80/255, 160/255),
    "MP": (141/255, 160/255, 203/255),
    "PR": (255/255, 235/255, 125/255),
    "CF": (240/255, 180/255, 20/255),
    "SM": (220/255, 160/255, 30/255),
    "OT": (190/255, 190/255, 190/255),
    "OTH": (170/255, 170/255, 170/255),
}

colormap = {
    "CM_1": (60/255, 170/255, 60/255),
    "CM_2": (230/255, 80/255, 160/255),
    "CM_3": (141/255, 160/255, 203/255),
    "CM_4": (255/255, 235/255, 125/255),
    "CM_5": (240/255, 180/255, 20/255),
    "CM_6": (220/255, 160/255, 30/255),
}

xlim_full = [-12, 20]
xlim_cm = [6, 19]
ylim_full = [-7, 18]
ylim_cm = [5, 17]

for i in range(0, 5):

    loom_file = f'/home/herbert/PyProjects/cocultured_organ/data/240603/SC160__{loom_file_names[i]}-JW-052224/SC160__{loom_file_names[i]}-JW-052224.loom'
    ldata = sc.read(loom_file, cache=False)
    # adata = sc.read_h5ad(f'{WORK_DIR}/adata/cell_type_split/{adata_file_names[i]}.h5ad')
    adata = sc.read_h5ad(f"{WORK_DIR}/adata/cell_type_subclustered/CM.h5ad")
    adata = adata[adata.obs["batch"] == adata_file_names[i]]
    adata = scv.utils.merge(adata, ldata)

    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=50, n_neighbors=50)
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)
    # palette = [full_palette[cat] for cat in adata.obs['cell_type_subcluster'].cat.categories]
    palette = [colormap[cat] for cat in adata.obs['cell_type_subcluster'].cat.categories]

    if adata.n_obs < 100:
        continue
    
    scv.pl.velocity_embedding_stream(
        adata, 
        density=5, 
        basis='umap', 
        color='cell_type_subcluster', 
        palette=palette, 
        xlim=xlim_cm,
        ylim=ylim_cm,
        save=f"{WORK_DIR}/scVelo/{adata_file_names[i]}.png"
    )
    
    # adata_cm = adata[adata.obs["cell_type"] == "CM"]
    # if adata_cm.n_obs == 0:
    #     continue
    # scv.pl.velocity_embedding_stream(
    #     adata_cm, 
    #     basis='umap', 
    #     color='cell_type', 
    #     palette=palette, 
    #     xlim=xlim_cm,
    #     ylim=ylim_cm,
    #     save=f"{WORK_DIR}/scVelo/{adata_file_names[i]}.CM_only.png"
    # )
    