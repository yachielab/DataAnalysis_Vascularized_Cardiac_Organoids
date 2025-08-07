#!/bin/bash
#$ -l s_vmem=2G,arm
#$ -pe def_slot 128
#$ -o /home/herbert/standard_output
#$ -e /home/herbert/standard_output

. ~/.bashrc
. ~/.bash_profile
. ~/.bashrc.intr
. ~/.bash_profile.intr

conda activate scvi

cd /home/herbert/PyProjects/cocultured_organ/workspace/CCO07/transfer_learning
python job.py
