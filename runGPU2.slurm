#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --mem=96000
#SBATCH -p batch
#SBATCH --job-name=BTR_Atari_2
#SBATCH --time=60:00:00
#SBATCH --mail-user=tjc2g19@soton.ac.uk
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate BTR

cd /home/$USER/BeyondTheRainbow
pip install -e .
export WANDB_MODE=offline


staskfarm farm_atari2.txt