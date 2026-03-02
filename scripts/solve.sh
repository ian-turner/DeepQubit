#!/bin/bash

DOMAIN="n1_L15_P_H_e0.01"
HEUR="resnet_fc.2000H_2B_bn"
PATHFIND="bwqs.1_1.0_0.0"
TIME_LIMIT=10


mkdir -p tmp/$DOMAIN/$HEUR/paths/$PATHFIND

deepxube solve --domain qcircuit.$DOMAIN \
               --heur $HEUR \
               --heur_file tmp/$DOMAIN/$HEUR/heur.pt \
               --heur_type QFix \
               --pathfind $PATHFIND \
               --file tmp/n1_goals_R_1K.pkl \
               --results tmp/$DOMAIN/$HEUR/paths/$PATHFIND \
               --time_limit $TIME_LIMIT \
               --redo
