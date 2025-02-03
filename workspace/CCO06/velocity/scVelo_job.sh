#!/bin/bash
#$ -l s_vmem=128G,lmem
#$ -pe def_slot 1
#$ -o /home/herbert/standard_output
#$ -e /home/herbert/standard_output

. ~/.bashrc
. ~/.bash_profile
. ~/.bashrc.intr
. ~/.bash_profile.intr

conda activate scVelo
python /home/herbert/PyProjects/cocultured_organ/workspace/CCO06/velocity/scVelo_job.py
