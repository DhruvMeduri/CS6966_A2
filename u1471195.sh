#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=40GB
#SBATCH --mail-user=u1471195@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_2-%j

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CS6966

module load cuda/11.8.0

mkdir -p /scratch/general/vast/u1471195/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u1471195/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/general/vast/u1471195/huggingface_cache"


OUT_DIR=/scratch/general/vast/u1471195/cs6966/assignment2/models
mkdir -p ${OUT_DIR}
python ~/CS6966_A2/assignment2_incomplete.py --output_dir ${OUT_DIR} --seed 1471195