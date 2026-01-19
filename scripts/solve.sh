#!/bin/bash

deepxube solve --domain qcircuit.n1_L15_P \
               --heur_type QFix \
               --pathfind bwqs.1_0.8_0.1 \
               --file tmp/n1_goals_R_1K.pkl \
               --results tmp/n1_paths_R_1K/ \
               --redo
