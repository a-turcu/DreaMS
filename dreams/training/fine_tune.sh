#!/bin/bash
#SBATCH --job-name DreaMS_fine-tuning
#SBATCH --account OPEN-29-57
#SBATCH --partition qgpu
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --time 10:00:00

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate dreams

# Export project definitions
$(python -c "from dreams.definitions import export; export()")

# Move to running dir
cd "${DREAMS_DIR}" || exit 3

# Run the training script
# Replace `python3 training/train.py` with `srun --export=ALL --preserve-env python3 training/train.py \`
# when executing on a SLURM cluster via `sbatch`.
python dreams/training/train.py \
 --project_name dreams-contrastive \
 --job_key "default" \
 --run_name "default" \
 --train_objective contrastive_spec_embs \
 --train_regime fine-tuning \
 --dataset_pth "data/MoNA_experimental_split.hdf5" \
 --dformat A \
 --model DreaMS \
 --lr 3e-5 \
 --batch_size 64 \
 --prec_intens 1.1 \
 --num_devices 8 \
 --max_epochs 103 \
 --log_every_n_steps 5 \
 --head_depth 1 \
 --seed 3407 \
 --train_precision 64   \
 --pre_trained_pth "models/pretrained/ssl_model.ckpt" \
 --val_check_interval 0.1 \
 --max_peaks_n 100 \
 --save_top_k -1


python dreams/training/train.py --no_wandb --project_name "dreams-contrastive" --job_key "default" --run_name "default" --train_objective "contrastive_spec_embs" --train_regime "fine-tuning" --dataset_pth "data/MoNA_experimental_split_8000.pkl" --dformat A --model DreaMS --lr 3e-5 --batch_size 64 --prec_intens 1.1 --num_devices 8 --max_epochs 103 --log_every_n_steps 5 --head_depth 1 --seed 3407 --train_precision 64   --pre_trained_pth "dreams/models/pretrained/ssl_model.ckpt" --val_check_interval 0.1 --max_peaks_n 100 --save_top_k -1