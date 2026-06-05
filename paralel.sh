#!/bin/bash
# paralel.sh — run ml_train.py on multiple Modal accounts
# Format tokens.txt: token_id:token_secret (one per line)

SCRIPT="${1:-ml_train.py}"
TOKENS="${2:-tokens.txt}"

if [ ! -f "$TOKENS" ]; then
    echo "ERROR: $TOKENS not found"
    echo "Usage: bash paralel.sh [script] [tokens_file]"
    exit 1
fi

COUNT=0

while IFS=':' read -r tid tsec; do
    [ -z "$tid" ] && continue
    COUNT=$((COUNT + 1))
    SESSION="modal_${COUNT}"

    echo "[$COUNT] Launching session $SESSION with token ${tid:0:20}..."

    screen -dmS "$SESSION" bash -c "
        export MODAL_TOKEN_ID='$tid'
        export MODAL_TOKEN_SECRET='$tsec'
        echo \"=== Session $SESSION started ===\"
        modal run $SCRIPT 2>&1 | tee /tmp/modal_${SESSION}.log
        echo \"=== Session $SESSION ended ===\"
    "

    sleep 3
done < "$TOKENS"

echo ""
echo "✓ $COUNT sessions launched"
echo "Monitor: screen -ls"
echo "Logs:    ls /tmp/modal_modal_*.log"
