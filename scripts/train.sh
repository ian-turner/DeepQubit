#!/bin/bash

NNET_NAME="n1_L15_P_H"
deepxube train --domain qcircuit.$NNET_NAME \
               --heur resnet_fc.2000H_2B_bn \
               --heur_type QFix \
               --pathfind bwqs.1_0.8_0.1 \
               --dir tmp/$NNET_NAME \
               --batch_size 10000 \
               --max_itrs 100000 \
               --procs 20 \
               --step_max 1000 \
               --search_itrs 1000
