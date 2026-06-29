#!/bin/bash
#$ -l s_vmem=128G
#$ -pe def_slot 1
#$ -o /home/herbert/standard_output
#$ -e /home/herbert/standard_output

. ~/.bashrc
. ~/.bash_profile
. ~/.bashrc.intr
. ~/.bash_profile.intr

conda activate scVelo
cd /home/herbert/PyProjects/cocultured_organ/workspace/CCO12/scVelo
python scVelo_job.py
