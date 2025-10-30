#! /bin/bash

# Job dependencies
# ----------------
# 
#             --> 02        --> 07        
#            /             /        
#           /             /        
#     --> 01        --> 05        
#    /      \      /      \         
#   /        \    /        \       
# 00          )--(          --> 08 
#   \        /    \
#    \      /      \
#     --> 03        --> 06
#           \
#            \ 
#             --> 04
#

# Run all jobs (on cluster), respecting dependencies
jobid00=$(sbatch --parsable 00_graphData_collection_job.sh)
jobid01=$(sbatch --parsable --dependency=afterok:$jobid00 01_graphRNN_model_train_job.sh)
jobid02=$(sbatch --parsable --dependency=afterok:$jobid01 02_graphRNN_model_play_job.sh)
jobid03=$(sbatch --parsable --dependency=afterok:$jobid00 03_graphDDPM_model_train_job.sh)
jobid04=$(sbatch --parsable --dependency=afterok:$jobid03 04_graphDDPM_model_play_job.sh)
jobid05=$(sbatch --parsable --dependency=afterok:$jobid01,afterok:$jobid03 05_gen_graph_job.sh)
jobid06=$(sbatch --parsable --dependency=afterok:$jobid01,afterok:$jobid03 06_gen_graph_anim_job.sh)
jobid07=$(sbatch --parsable --dependency=afterok:$jobid05 07_gen_graph_stats_job.sh)
jobid08=$(sbatch --parsable --dependency=afterok:$jobid05 08_gen_graph_vario_job.sh)
