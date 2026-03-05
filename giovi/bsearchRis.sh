#!/bin/bash

	if [[ $# < 3 ]]; then
		  echo numero di parametri errato!
		  echo uso corretto:
		  echo "$0 <data_file>" "<window_size>" "<n_future>" "[hole_dim]" "[percentage_hole]" "[test_only(True/False)]"
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

# FOLD 0
# FOLD 2	
# FOLD 3	
# FOLD 4	
	
	DIR=`echo $DAT | cut -d. -f1`
	
#	exit 0

	
	if [ -d $DIR ]; then
    cd $DIR
  else
    echo "Directory $DIR does not exist!"
    exit 1
  fi
  
	cp ../comp_metricsOne.py .
	cp ../sort_csv.py . 
	
	
	CLS=`ls -l pred_* | cut -d_ -f3 | uniq`

  echo $CLS
	
	#Generazione file unico di predizioni
	for C in `echo $CLS`; do
		python comp_metricsOne.py --input_file pred_${DIR}_${C}_${WIN}win_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv --output_file res_${DIR}_${C}_${WIN}win_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv
	done	
	
	
			
	# Calcolo dei migliori
	#echo -n "" > resAll_${DIR}_${WIN}win_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv		
	#M=`expr 5 \* $NFT`
	#for T in `seq 5 5 $M`; do	
	#	awk -F, -v val="$T" '$2 == val' res_* | sort -t, -k 3  | head -1 >> resAll_${DIR}_${WIN}win_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv
	#done
 
  
  grep rgs res_${DIR}_*  | cut -d: -f2  | head -1 > bst_${DIR}_${WIN}win_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv
  for F in $(ls res_*.csv); do
    tail -n +2 $F; 
  done | sort -n -t, -k4   >> bst_${DIR}_${WIN}win_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv
  
   python3 sort_csv.py  bst_${DIR}_${WIN}win_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv  4
	

	mkdir -p res
	mv res_* res
	
	


	exit 0	

	
