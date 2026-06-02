i=1
while read token; do
  tid="${token%%:*}"
  tsec="${token##*:}"
  screen -dmS "nasa-$(printf '%02d' $i)" bash -c "
    export MODAL_TOKEN_ID='$tid'
    export MODAL_TOKEN_SECRET='$tsec'
    modal run ml_train.py
  "
  echo "🚀 nasa-$i launched"
  ((i++))
  sleep 2
done < tokens.txt
