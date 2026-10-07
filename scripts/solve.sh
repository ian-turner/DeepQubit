#!/bin/bash
# Usage: bash scripts/solve.sh [config] [goals]   (deepxube >= 0.3 CLI)
#   goals (default: the config's SOLVE_GOALS) is a goals .pkl or a directory of .txt targets; a directory is
#   converted to <results dir>/goals.pkl first. A per-goal summary is written to <results dir>/summary.txt.

CONFIG=${1:-configs/test}
source "$CONFIG"
SOLVE_GOALS=${2:-$SOLVE_GOALS}
DOMAIN_NAME=${DOMAIN_NAME:-qcircuit}   # domain class: qcircuit (float, epsilon) or qcircuit_exact (integer ring)

case "$SOLVE_PATHFIND" in
    *_q*) FN="heurq_fixout" ;;
    *)    FN="heurv" ;;
esac

RESULTS=tmp/$DOMAIN/$HEUR/paths/$SOLVE_PATHFIND
mkdir -p $RESULTS

if [ -d "$SOLVE_GOALS" ]; then
    GOALS_FILE=$RESULTS/goals.pkl
    REACHABLE=""
    [ "$SOLVE_REACHABLE_ONLY" = "1" ] && REACHABLE="--reachable_only"
    python scripts/goals_from_txt.py --input "$SOLVE_GOALS" --output $GOALS_FILE --domain $DOMAIN_NAME $REACHABLE || exit 1
else
    GOALS_FILE=$SOLVE_GOALS
fi

deepxube solve --domain $DOMAIN_NAME.$DOMAIN \
               --fn $FN,$HEUR,tmp/$DOMAIN/$HEUR/heur.pt \
               --pathfind $SOLVE_PATHFIND \
               --file $GOALS_FILE \
               --results $RESULTS \
               --time_limit $SOLVE_TIME_LIMIT \
               --redo || exit 1

python scripts/solve_summary.py --results $RESULTS --goals $GOALS_FILE
