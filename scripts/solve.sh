#!/bin/bash

CONFIG=${1:-configs/test}
source "$CONFIG"


mkdir -p tmp/$DOMAIN/$HEUR/paths/$SOLVE_PATHFIND

deepxube solve --domain qcircuit.$DOMAIN \
               --heur $HEUR \
               --heur_file tmp/$DOMAIN/$HEUR/heur.pt \
               --heur_type QFix \
               --pathfind $SOLVE_PATHFIND \
               --file $SOLVE_GOALS \
               --results tmp/$DOMAIN/$HEUR/paths/$PATHFIND \
               --time_limit $SOLVE_TIME_LIMIT \
               --redo
