#! /bin/bash

# ------------------------------------------------------------------------------
#SBATCH --job-name=05_gen_graph  # variable: SLURM_JOB_NAME

#SBATCH --output=%x.out             # output file, redirection of stdout (and stderr (merged))
                                    #    %x : ${SLURM_JOB_NAME}

#SBATCH --partition=gpu             # partition "gpu" (-> node11)
#SBATCH --gres=gpu:rtx2080Ti:1      # generic consumable resources: (gpu:<type>:number_of_gpu)

#SBATCH --time=5:00:00              # limit on the total run time, format:
                                    #    "minutes"
                                    #    "minutes:seconds"
                                    #    "hours:minutes:seconds"
                                    #    "days-hours"
                                    #    "days-hours:minutes"
                                    #    "days-hours:minutes:seconds"
                                    # Default: partition DEFAULTTIME / type: sinfo -o "%P %L"
# ------------------------------------------------------------------------------

# -----------------------------------------
# Some settings to run pyvista properly ...
# -----------------------------------------
# set -x
export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3
# set +x
# -----------------------------------------

echo "Job: ${SLURM_JOB_NAME}"

echo -n "Running on: "
hostname

echo -n "Python version:"
python --version

echo ""

t1=$(date --iso-8601=ns)
# python ${SLURM_JOB_NAME}.py
python 05_gen_graph.py
t2=$(date --iso-8601=ns)

elaps=$(timediff -2 $t1 $t2)

echo ""

echo "Elapsed time in sec. : ${elaps}"
