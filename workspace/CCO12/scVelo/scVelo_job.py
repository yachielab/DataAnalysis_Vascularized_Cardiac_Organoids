import scvelo as scv
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe

WORK_DIR = "/home/herbert/PyProjects/cocultured_organ/workspace/CCO12"

loom_file_names = ['BVO', 'HO', 'CM3O', 'BHO', 'CM3B']
adata_file_names = ['VO', 'HO', 'ReAgg-CM', 'Fus-HO', 'VasHO']

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
    "CM_1": (43/255, 119/255, 56/255, ),
    "CM_2": (113/255, 181/255, 113/255, ),
    "CM_3": (30/255, 96/255, 45/255, ),
    "CM_4": (87/255, 159/255, 87/255, ),
    "CM_5": (174/255, 211/255, 161/255, ),
    "CM_6": (223/255, 239/255, 218/255, ),
}

xlim_full = [-12, 20]
xlim_cm = [6, 19]
ylim_full = [-7, 18]
ylim_cm = [5, 17]

for i in range(0, 5):

    loom_file = f'/home/herbert/PyProjects/cocultured_organ/data/240603/SC160__{loom_file_names[i]}-JW-052224/SC160__{loom_file_names[i]}-JW-052224.loom'
    ldata = sc.read(loom_file, cache=False)
    adata = sc.read_h5ad(f"{WORK_DIR}/adata/cell_type_subclustered/CM.h5ad")
    adata = adata[adata.obs["batch"] == adata_file_names[i]]
    adata = scv.utils.merge(adata, ldata)

    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=50, n_neighbors=50)
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)
    palette = [colormap[cat] for cat in adata.obs['cell_type_subcluster'].cat.categories]

    if adata.n_obs < 100:
        continue
    
    ax = scv.pl.velocity_embedding_stream(
        adata, 
        density=3, 
        basis='umap', 
        color='cell_type_subcluster', 
        palette=palette, 
        xlim=xlim_cm,
        ylim=ylim_cm,
        legend_loc="on data",
        show=False,
    )

    for text in ax.texts:
        text.set_fontsize(16)
        text.set_color("white")
        text.set_fontweight("bold")
        text.set_path_effects([
            pe.Stroke(linewidth=2, foreground="black"),
            pe.Normal()
        ])
    
    plt.savefig(f"{WORK_DIR}/scVelo/{adata_file_names[i]}.png", bbox_inches="tight")
    plt.savefig(f"{WORK_DIR}/scVelo/{adata_file_names[i]}.svg", bbox_inches="tight")
    plt.close()
