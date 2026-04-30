#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --time=06:00:00
#SBATCH --job-name=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
##SBATCH --constraint=rhel8
#SBATCH --cpus-per-task=4
#SBATCH --output=main.log

module purge
module load mamba/24.3.0
eval "$('/hpc/software/mamba/24.3.0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
source "/hpc/software/mamba/24.3.0/etc/profile.d/mamba.sh"
mamba activate /projects/b1042/MisharinLab/schakrabarty/pythonenvs/llm3-rcs

export LOGLEVEL=INFO
python /projects/b1042/MisharinLab/schakrabarty/Budget-Constrained-RLRS/src/main.py