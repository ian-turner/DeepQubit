#!/bin/bash
# Usage: bash scripts/train.sh [config]   (deepxube >= 0.3 CLI)

CONFIG=${1:-configs/test}
source "$CONFIG"
DOMAIN_NAME=${DOMAIN_NAME:-qcircuit}   # domain class: qcircuit (float, epsilon) or qcircuit_exact (integer ring)

# heuristic kind follows the pathfinder: *_q -> Q-function, otherwise V-function
case "$PATHFIND" in
    *_q*) FN="heurq_fixout"; UP="up_rl_q" ;;
    *)    FN="heurv";        UP="up_rl_v" ;;
esac

deepxube train --domain $DOMAIN_NAME.$DOMAIN \
               --fn $FN,$HEUR \
               --pathfind $PATHFIND \
               --up $UP.${PROCS}p_${STEP_MAX}sm_${SEARCH_ITRS}sitrs_${UP_ITRS:-100}up_${UP_GEN_ITRS:-$UP_ITRS}upg \
               --tr tr_h.${BATCH_SIZE}bs_${MAX_ITRS}maxit_${CHECKPOINT:-0}chkpt \
               --dir tmp/$DOMAIN/$HEUR
