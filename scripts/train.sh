#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

CONFIG=${1:-configs/test}
source "$CONFIG"


mkdir -p tmp/$DOMAIN

deepxube train --domain qcircuit.$DOMAIN \
               --heur $HEUR \
               --heur_type V \
               --pathfind $PATHFIND \
               --dir tmp/$DOMAIN/$HEUR \
               --batch_size $BATCH_SIZE \
               --max_itrs $MAX_ITRS \
               --procs $PROCS \
	       --backup $BACKUP \
	       --up_itrs $UP_ITRS \
	       --up_gen_itrs $UP_GEN_ITRS \
               --step_max $STEP_MAX \
               --search_itrs $SEARCH_ITRS \
               --t_file $TEST_FILE \
               --t_pathfinds $PATHFIND \
               --t_search_itrs $TEST_SEARCH_ITRS \
	       --chkpt $CHECKPOINT \
               --up_v

# set bellman backup to -1
