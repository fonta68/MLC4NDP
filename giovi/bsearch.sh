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
	python giovi_ML-RegressorsOne_hole_Bsearch.py --input_file $DAT --window_size $WIN --n_future $NFT --time_column timestamp --hole_dim $HDM --percentage_hole $HPR --test_only $HTS --k_fold 0
# exit 0
# FOLD 1	
	python giovi_ML-RegressorsOne_hole_Bsearch.py --input_file $DAT --window_size $WIN --n_future $NFT --time_column timestamp --hole_dim $HDM --percentage_hole $HPR --test_only $HTS --k_fold 1
# FOLD 2	
	python giovi_ML-RegressorsOne_hole_Bsearch.py --input_file $DAT --window_size $WIN --n_future $NFT --time_column timestamp --hole_dim $HDM --percentage_hole $HPR --test_only $HTS --k_fold 2
# FOLD 3	
	python giovi_ML-RegressorsOne_hole_Bsearch.py --input_file $DAT --window_size $WIN --n_future $NFT --time_column timestamp --hole_dim $HDM --percentage_hole $HPR --test_only $HTS --k_fold 3
# FOLD 4	
	python giovi_ML-RegressorsOne_hole_Bsearch.py --input_file $DAT --window_size $WIN --n_future $NFT --time_column timestamp --hole_dim $HDM --percentage_hole $HPR --test_only $HTS --k_fold 4
	
	DIR=`echo $DAT | cut -d. -f1`
	
#	exit 0

  mv pars_${DIR}_* $DIR
	
	if [ -d $DIR ]; then
    cd $DIR
  else
    echo "Directory $DIR does not exist!"
    exit 1
  fi
  
	cp ../comp_metricsOne.py .
	cp ../sort_csv.py . 
	
	
	CLS=`ls -l pred_* | cut -d_ -f3 | uniq`
	
	#Generazione file unico di predizioni
	for C in `echo $CLS`; do
		cat pred_${DIR}_${C}_${WIN}win_0k_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv 	> pred_${DIR}_${C}_${WIN}win_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv
		rm pred_${DIR}_${C}_${WIN}win_0k_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv
		for K in `seq 1 4`; do 
			tail -n +2 pred_${DIR}_${C}_${WIN}win_${K}k_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv 	>> pred_${DIR}_${C}_${WIN}win_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv
			rm pred_${DIR}_${C}_${WIN}win_${K}k_${NFT}fut_${HDM}hd_${HPR}hp_${HTS}tso.csv		
		done
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
	
  mkdir -p pars
  mv pars_${DIR}_* pars

	mkdir -p res
	mv res_* res
	
	mkdir -p preds
	mv pred_* preds

  mkdir -p py
	mv  *.py	 py


	exit 0	

	
