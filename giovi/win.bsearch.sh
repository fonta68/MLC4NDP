#!/bin/bash

	if [[ $# < 3 ]]; then
		  echo numero di parametri errato!
		  echo uso corretto:
		  echo "$0 <data_file>" "<window_size list (use quotes and spaces)>" "<n_future>" "[hole_dim]" "[percentage_hole]" "[test_only(True/False)]"
		  exit 1
		fi

		DAT=$1
		WIN=$2
		NFT=$3 
		HDM=${4:-0}
		HPR=${5:-0.0}
		HTS=${6:-True}
		
	
# enviroment activation
# conda init bash
# conda activate giovi		

  if [[ ! -x  bsearch.sh ]]; then
    echo "Error: file bsearch.sh does not exist."
    exit 1
  fi


  for W in $WIN; do
    ./bsearch.sh $DAT $W $NFT $HDM $HPR $HTS > bsearch$W.log
  done

 
  DIR=$(echo $DAT | cut -d. -f1)
  WL=$(echo $WIN | tr ' ' '-')
  
  #echo   bst_${DIR}_${WL}win_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv
  
  
  
  grep rgs $DIR/bst_${DIR}_*  | cut -d: -f2  | head -1 > $DIR/all_${DIR}_${WL}win_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv

  for F in $(ls $DIR/bst_*.csv); do
    tail -n +2 $F; 
  done | sort -n -t, -k4   >> $DIR/all_${DIR}_${WL}win_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv
  
  python3 sort_csv.py $DIR/all_${DIR}_${WL}win_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv 4

	exit 0	

	
