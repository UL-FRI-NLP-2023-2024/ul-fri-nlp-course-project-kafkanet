#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=result_%j.txt
#SBATCH --time=10:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem=64G

. venv/bin/activate
python src/llama.py
