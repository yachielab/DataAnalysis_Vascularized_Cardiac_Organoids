import scanpy as sc
import scvi

PREV_DIR = "/home/herbert/PyProjects/cocultured_organ/workspace/CCO06"
WORK_DIR = "/home/herbert/PyProjects/cocultured_organ/workspace/CCO07"
n_worker = 1

adata_HVCA = sc.read_h5ad(f"{WORK_DIR}/adata/HVCA/global_object_vasc_atlas_cxg.h5ad")
adata = sc.read_h5ad(f"{WORK_DIR}/adata/sample_recollect/ALL.h5ad")

adata_HVCA_EN = adata_HVCA[adata_HVCA.obs["cell_type"] == "Endothelial cells"]
sc.pp.filter_cells(adata_HVCA_EN, min_genes=200)
sc.pp.filter_genes(adata_HVCA_EN, min_cells=3)
sc.pp.normalize_total(adata_HVCA_EN, target_sum=1e4)
sc.pp.log1p(adata_HVCA_EN)

adata_ref = adata_HVCA_EN
adata_query = adata[adata.obs["cell_type"] == "EN"]

adata_query.obs["organ_uni"] = "Unknown"

# Combine into one AnnData object
adata_combined = adata_ref.concatenate(
    adata_query, 
    batch_key="dataset", 
    batch_categories=["ref", "query"]
)

# First, train a base scVI model (initializes weights well)
scvi.model.SCVI.setup_anndata(
    adata_combined, 
    batch_key="dataset", 
    labels_key="organ_uni", 
)
vae = scvi.model.SCVI(adata_combined)
vae.train(accelerator="cpu", devices=n_worker)
vae.save(f"{WORK_DIR}/transfer_learning/vae.model")

# Then initialize scANVI from the scVI model
scvi.model.SCANVI.setup_anndata(
    adata_combined, 
    batch_key="dataset", 
    labels_key="organ_uni", 
    unlabeled_category="Unknown"
)
model = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category="Unknown")
model.train(accelerator="cpu", devices=n_worker)
model.save(f"{WORK_DIR}/transfer_learning/model.model")

adata_combined.obs["predicted_labels"] = model.predict()

# Extract predictions for just the query cells
adata_query.obs["predicted_labels"] = adata_combined[
    adata_combined.obs["dataset"] == "query"
].obs["predicted_labels"]

adata_query.write(f"{WORK_DIR}/transfer_learning/out.h5ad")
