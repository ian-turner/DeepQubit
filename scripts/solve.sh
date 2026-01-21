#!/bin/bash

NNET_NAME="n1_L15_P_H"
deepxube solve --domain qcircuit.$NNET_NAME \
               --heur resnet_fc.2000H_2B_bn \
	       --heur_file tmp/$NNET_NAME/heur.pt \
               --heur_type QFix \
               --pathfind bwqs.1_1.0_0.0 \
               --file tmp/n1_goals_R_1K.pkl \
               --results tmp/n1_paths_R_1K/ \
	       --time_limit 10 \
               --redo
