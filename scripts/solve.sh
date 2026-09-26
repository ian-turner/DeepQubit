#!/bin/bash
# Usage: bash scripts/solve.sh [config]   (deepxube >= 0.3 CLI)

CONFIG=${1:-configs/test}
source "$CONFIG"

case "$SOLVE_PATHFIND" in
    *_q*) FN="heurq_fixout" ;;
    *)    FN="heurv" ;;
esac

deepxube solve --domain qcircuit.$DOMAIN \
               --fn $FN,$HEUR,tmp/$DOMAIN/$HEUR/heur.pt \
               --pathfind $SOLVE_PATHFIND \
               --file $SOLVE_GOALS \
               --results tmp/$DOMAIN/$HEUR/paths/$SOLVE_PATHFIND \
               --time_limit $SOLVE_TIME_LIMIT \
               --redo
