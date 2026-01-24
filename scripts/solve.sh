#!/bin/bash

DOMAIN="n1_L15_P_H_e0.001"
HEUR="resnet_fc.2000H_3B_bn"
PATHFIND="bwqs.100_0.8_0.0"
TIME_LIMIT=10


mkdir -p tmp/paths/$DOMAIN

deepxube solve --domain qcircuit.$DOMAIN \
               --heur $HEUR \
               --heur_file tmp/$DOMAIN/$HEUR/heur.pt \
               --heur_type QFix \
               --pathfind $PATHFIND \
               --file tmp/goals/n1_goals_R_1K.pkl \
               --results tmp/paths/$DOMAIN/$HEUR \
               --time_limit $TIME_LIMIT \
               --redo
