#!/bin/bash

DOMAIN="n1_L15_P_H_e0.001"
HEUR="resnet_fc.2000H_3B_bn"
PATHFIND="bwqs.1_0.8_0.0"

BATCH_SIZE=20000
MAX_ITRS=100000
STEP_MAX=1200
SEARCH_ITRS=1000
TEST_SEARCH_ITRS=400
PROCS=20


mkdir -p tmp/heurs/$DOMAIN

deepxube train --domain qcircuit.$DOMAIN \
               --heur $HEUR \
               --heur_type QFix \
               --pathfind $PATHFIND \
               --dir tmp/heurs/$DOMAIN/$HEUR \
               --batch_size $BATCH_SIZE \
               --max_itrs $MAX_ITRS \
               --procs $PROCS \
               --step_max $STEP_MAX \
               --search_itrs $SEARCH_ITRS \
               --t_file tmp/goals/n1_goals_R_1K.pkl \
               --t_pathfinds $PATHFIND \
               --t_search_itrs $TEST_SEARCH_ITRS \
               --up_v
