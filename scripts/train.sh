#!/bin/bash

DOMAIN="n1_L15_P_H_e0.001"
HEUR="resnet_fc.2000H_2B_bn"
PATHFIND="bwqs.1_0.8_0.1"

BATCH_SIZE=20000
MAX_ITRS=100000
STEP_MAX=1000
SEARCH_ITRS=1000
TEST_SEARCH_ITRS=100
PROCS=24


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
               --t_file tmp/n1_goals_R_1K.pkl \
               --t_pathfinds $PATHFIND \
               --t_search_itrs $TEST_SEARCH_ITRS \
               --up_v
