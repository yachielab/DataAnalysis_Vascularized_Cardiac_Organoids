def volcano_plot1(
    adata,
    label1,
    label2,
    title=None,
    label_type='leiden',
    keygenes=None,
    save=None,
    save_tsv=None,
    mean_diff=0.5,
    min_mean=0.1,
    xlim=None,
    ylim=None,
    log_y_axis=False,
    s=1,
    y0_ref_line=False,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # -----------------------------------
    # Select cells
    # -----------------------------------
    if label1 == "all":
        idx1 = np.arange(adata.n_obs)
    elif label1 == "rest":
        idx1 = np.where(adata.obs[label_type].values != label2)[0]
    else:
        idx1 = np.where(adata.obs[label_type].values == label1)[0]

    if label2 == "all":
        idx2 = np.arange(adata.n_obs)
    elif label2 == "rest":
        idx2 = np.where(adata.obs[label_type].values != label1)[0]
    else:
        idx2 = np.where(adata.obs[label_type].values == label2)[0]

    if len(idx1) == 0:
        raise ValueError(f"No cells found for label1={label1!r}")

    if len(idx2) == 0:
        raise ValueError(f"No cells found for label2={label2!r}")

    # -----------------------------------
    # Mean log1p expression
    # Works for both sparse and dense matrices
    # -----------------------------------
    mean1 = np.asarray(
        adata.X[idx1, :].mean(axis=0)
    ).ravel()

    mean2 = np.asarray(
        adata.X[idx2, :].mean(axis=0)
    ).ravel()

    # -----------------------------------
    # Effect size:
    # difference in mean log1p expression
    #
    # Positive = higher in label2
    # Negative = higher in label1
    # -----------------------------------
    mean_differences = mean2 - mean1

    # -----------------------------------
    # Overall mean expression
    # Preserve original cell-number weighting
    # -----------------------------------
    means = (
        mean1 * len(idx1) +
        mean2 * len(idx2)
    ) / (len(idx1) + len(idx2))

    # -----------------------------------
    # Marker selection
    # -----------------------------------
    idx_inc = np.where(
        (mean_differences >= mean_diff) &
        (means > min_mean)
    )[0]

    idx_dec = np.where(
        (mean_differences <= -mean_diff) &
        (means > min_mean)
    )[0]

    # -----------------------------------
    # Plot
    # -----------------------------------
    plt.scatter(
        mean_differences,
        means,
        label='not selected',
        color='grey',
        alpha=0.5,
        s=s
    )

    plt.scatter(
        mean_differences[idx_inc],
        means[idx_inc],
        label=f'higher in {label2}',
        color='red',
        alpha=0.5,
        s=s
    )

    plt.scatter(
        mean_differences[idx_dec],
        means[idx_dec],
        label=f'higher in {label1}',
        color='blue',
        alpha=0.5,
        s=s
    )

    # -----------------------------------
    # Label key genes
    # -----------------------------------
    if keygenes is not None:
        for keygene in keygenes:
            if keygene not in adata.var_names:
                print(f"Warning: {keygene!r} not found in adata.var_names")
                continue

            gene_idx = adata.var_names.get_loc(keygene)

            plt.text(
                mean_differences[gene_idx],
                means[gene_idx],
                keygene,
                ha='center',
                va='center'
            )

    if y0_ref_line:
        plt.axvline(0, linestyle="--")

    plt.xlabel('Mean log-expression difference')
    plt.ylabel('Mean log-expression')

    if title is not None:
        plt.title(title)

    if log_y_axis:
        plt.yscale('log')

    if xlim is not None:
        plt.xlim(*xlim)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.legend()

    # -----------------------------------
    # Save / show
    # -----------------------------------
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # -----------------------------------
    # Save TSV
    # -----------------------------------
    if save_tsv is not None:
        selected = np.zeros(len(means), dtype=int)
        selected[idx_inc] = 1
        selected[idx_dec] = 1
    
        increase = np.full(len(means), np.nan)
        increase[idx_inc] = 1
        increase[idx_dec] = 0
    
        de_dict = {
            'gene': adata.var_names.to_list(),
            label1: mean1,
            label2: mean2,
            'mean_diff': mean_differences,
            'mean': means,
            'selected': selected,
            'increase': increase,
        }
    
        df = pd.DataFrame(de_dict)
    
        # Save all genes
        df.to_csv(
            save_tsv,
            sep='\t',
            index=False
        )
    
        save_tsv = Path(save_tsv)
    
        if save_tsv.suffix == '.tsv':
            prefix = save_tsv.with_suffix('')
        else:
            prefix = save_tsv
    
        # Selected DE genes only
        deg_df = df[df['selected'] == 1].copy()
    
        # Optional: sort by direction first, then effect size
        deg_df['abs_mean_diff'] = deg_df['mean_diff'].abs()
    
        deg_df = deg_df.sort_values(
            ['increase', 'abs_mean_diff'],
            ascending=[False, False]
        )
    
        deg_df = deg_df.drop(columns='abs_mean_diff')
    
        deg_df.to_csv(
            f"{prefix}.deg.tsv",
            sep='\t',
            index=False
        )
    
        if keygenes is not None and len(keygenes) > 0:
            df[df['gene'].isin(keygenes)].to_csv(
                f"{prefix}.marker.tsv",
                sep='\t',
                index=False
            )