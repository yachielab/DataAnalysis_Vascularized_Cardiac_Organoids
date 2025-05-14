import scvelo as scv
import scanpy as sc
import matplotlib.pyplot as plt

WORK_DIR = "/home/herbert/PyProjects/cocultured_organ/workspace/CCO07"

loom_file_names = ['BVO', 'HO', 'CM3O', 'BHO', 'CM3B']
adata_file_names = ['VO', 'HO', 'CA', 'VHO', 'VCO']

for i in range(0, 5):

    loom_file = f'/home/herbert/PyProjects/cocultured_organ/data/240603/SC160__{loom_file_names[i]}-JW-052224/SC160__{loom_file_names[i]}-JW-052224.loom'
    ldata = sc.read(loom_file, cache=True)
    adata = sc.read_h5ad(f'{WORK_DIR}/adata/sample_recollect/{adata_file_names[i]}.h5ad')
    adata = scv.utils.merge(adata, ldata)

    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=50, n_neighbors=50)
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(adata, basis='umap', color='cell_type_subcluster', save=f"{WORK_DIR}/velocity/{adata_file_names[i]}.png")
    