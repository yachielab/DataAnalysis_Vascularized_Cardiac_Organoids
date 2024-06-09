#! /usr/bin/bash

. ~/.bashrc
. ~/.bash_profile
. ~/.bashrc.intr
. ~/.bash_profile.intr

conda activate cco
python /home/herbert/PyProjects/cocultured_organ/workspace/CCO01/umap/draw_coembedding.py
