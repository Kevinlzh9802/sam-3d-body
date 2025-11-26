#!/bin/bash
#SBATCH --partition=insy,general # Request partition. Default is 'general' 
#SBATCH --qos=short         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=2:00:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --mem=16G
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=/home/nfs/zli33/slurm_outputs/sam_3d_body/slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=/home/nfs/zli33/slurm_outputs/sam_3d_body/slurm_%j.err # Set name of error log. %j is the Slurm jobId

#SBATCH --gres=gpu:a40:1 # Request 1 GPU

apptainer exec --nv --bind /tudelft.net/staff-bulk/ewi/insy/SPCLab/zonghuan/large_models/sam-3d-body-dinov3:/mnt/sam-3d-body-dinov3 --bind /home/nfs/zli33:/mnt/zli33 /tudelft.net/staff-bulk/ewi/insy/SPCLab/zonghuan/large_builds/containers/detectron_env.sif python /mnt/zli33/projects/sam-3d-body/demo.py --image_folder /mnt/zli33/projects/sam_3d_data/inputs --output_folder /mnt/zli33/projects/sam_3d_data/outputs --checkpoint_path /mnt/sam-3d-body-dinov3/model.ckpt --mhr_path /mnt/sam-3d-body-dinov3/assets/mhr_model.pt