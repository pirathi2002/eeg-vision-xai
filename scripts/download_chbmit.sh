#!/usr/bin/env bash
# Download the CHB-MIT subset used by all three projects (~2.5 GB) and build the
# preprocessed cache (18-ch double-banana montage, 0.5-45 Hz, 60 Hz notch, 128 Hz).
# Usage (from a project folder):  bash scripts/download_chbmit.sh data
set -euo pipefail
ROOT=${1:-data}
BASE=https://physionet-open.s3.amazonaws.com/chbmit/1.0.0
mkdir -p "$ROOT/chbmit"
cd "$ROOT/chbmit"
curl -sfL -O $BASE/RECORDS -O $BASE/RECORDS-WITH-SEIZURES
: > download_list.txt
for p in 01 02 03 05 08 10 20; do
  curl -sfL -o chb$p-summary.txt $BASE/chb$p/chb$p-summary.txt
  grep "^chb$p/" RECORDS-WITH-SEIZURES >> download_list.txt
  grep "^chb$p/" RECORDS | grep -vxFf RECORDS-WITH-SEIZURES | head -2 >> download_list.txt
done
# seizure-free recordings of 17 other patients, used only for self-supervised pre-training
: > pretrain_list.txt
for p in 04 06 07 09 11 12 13 14 15 16 17 18 19 21 22 23 24; do
  curl -sfL -o chb$p-summary.txt $BASE/chb$p/chb$p-summary.txt
  grep "^chb$p/" RECORDS | grep -vxFf RECORDS-WITH-SEIZURES | sed -n '3p;6p' >> pretrain_list.txt
done
cat download_list.txt pretrain_list.txt | xargs -P 6 -I{} sh -c \
  'mkdir -p $(dirname {}) && [ -s {} ] || curl -sfL --retry 3 -o {} '"$BASE"'/$(echo {} | sed "s/+/%2B/")'
echo "downloaded $(ls */*.edf | wc -l) EDF files"
