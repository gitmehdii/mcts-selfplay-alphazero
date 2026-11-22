#!/usr/bin/env bash
# Garde-fou local : lit la vraie facture Modal de l'app d'entraînement et l'arrête à 24,50 $.
APP_ID="$1"; CAP="${2:-24.5}"
while true; do
  COST=$(modal billing report --for today --resolution h --tz local --json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin); rows=d if isinstance(d,list) else d.get('data',d)
print(round(sum(float(r['cost']) for r in rows if r.get('object_id')=='$APP_ID'),3))" 2>/dev/null)
  STATE=$(modal app list 2>/dev/null | grep "$APP_ID" | grep -cvE "stopped")
  echo "$(date +%H:%M) facture réelle ${COST:-?} \$ (app active: $STATE)"
  if [ -n "$COST" ] && python3 -c "import sys; sys.exit(0 if float('$COST') >= $CAP else 1)"; then
    echo "PLAFOND ATTEINT : arrêt de l'app"; modal app stop -y "$APP_ID"; exit 0
  fi
  if [ "$STATE" = "0" ]; then echo "APP TERMINÉE"; exit 0; fi
  sleep 600
done
