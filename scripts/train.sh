#!/bin/bash

CONFIG=${1:-configs/test}
source "$CONFIG"


mkdir -p tmp/$DOMAIN

deepxube train --domain qcircuit.$DOMAIN \
               --heur $HEUR \
               --heur_type QFix \
               --pathfind $PATHFIND \
               --dir tmp/$DOMAIN/$HEUR \
               --batch_size $BATCH_SIZE \
               --max_itrs $MAX_ITRS \
               --procs $PROCS \
               --step_max $STEP_MAX \
               --search_itrs $SEARCH_ITRS \
               --t_file $TEST_FILE \
               --t_pathfinds $PATHFIND \
               --t_search_itrs $TEST_SEARCH_ITRS \
               --up_v
