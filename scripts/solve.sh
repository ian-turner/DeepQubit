#!/bin/bash
# Usage: bash scripts/solve.sh [config]   (deepxube >= 0.3 CLI)

CONFIG=${1:-configs/test}
source "$CONFIG"
DOMAIN_NAME=${DOMAIN_NAME:-qcircuit}   # domain class: qcircuit (float, epsilon) or qcircuit_exact (integer ring)

case "$SOLVE_PATHFIND" in
    *_q*) FN="heurq_fixout" ;;
    *)    FN="heurv" ;;
esac

deepxube solve --domain $DOMAIN_NAME.$DOMAIN \
               --fn $FN,$HEUR,tmp/$DOMAIN/$HEUR/heur.pt \
               --pathfind $SOLVE_PATHFIND \
               --file $SOLVE_GOALS \
               --results tmp/$DOMAIN/$HEUR/paths/$SOLVE_PATHFIND \
               --time_limit $SOLVE_TIME_LIMIT \
               --redo
