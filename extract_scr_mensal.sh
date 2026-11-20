#!/usr/bin/env bash
# Extrai os CSVs mensais do SCR.data (um ZIP por ano, 12 arquivos por ano) para uma pasta própria.
# Uso: ./extract_scr_mensal.sh <pasta_com_scrdata_AAAA.zip> <pasta_de_destino>
# Os ZIPs vêm de dadosabertos.bcb.gov.br (SCR.data). Os CSVs extraídos não são versionados.
set -euo pipefail
ZIPS="${1:?pasta dos ZIPs anuais}"; DEST="${2:?pasta de destino}"
mkdir -p "$DEST"
for z in "$ZIPS"/scrdata_[0-9][0-9][0-9][0-9].zip; do
  unzip -n -q "$z" 'scrdata_??????.csv' -d "$DEST"
done
ls "$DEST" | wc -l
