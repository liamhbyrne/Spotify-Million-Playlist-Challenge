#!/bin/bash

####################################
#  Iridis 5 slurm script template
#
#  Submit script: sbatch filename
#
####################################
#SBATCH --ntasks=1     # Number of processor cores (i.e. tasks)
#SBATCH --nodes=1     # Number of nodes requested
#SBATCH --cpus-per-task=2     # Threads per task
#SBATCH --time=60:00:00   # walltime
#SBATCH --partition=lyceum # Name of the parititon
#SBATCH --output=logs/train_track_to_vec.%j.out
#SBATCH --error=logs/train_track_to_vec.%j.err

module load conda

source activate dlt

cd /lyceum/lhb1g20/Spotify-Million-Playlist-Challenge/

python3 -m exploration.Track2Vec_CBOW