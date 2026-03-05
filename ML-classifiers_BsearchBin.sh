#!/bin/bash

	if [[ $# < 3 ]]; then
		  echo numero di parametri errato!
		  echo uso corretto:
		  echo "$0 <CLS> <DATA_FILE>" "<#RUNS> <LABELS (use quotes if more than one)>"
		  exit 1
	fi

	CLS=$1
	IN=$2
	NUM=$3
	LABELS=$4

#	enviroment activation
#	conda init bash
#	conda activate giovi		

	NAME=$(echo $IN | cut -d. -f1)
	L=$(echo $LABELS | tr ' ' '-')

	echo "" > ${NAME}_${CLS}_${NUM}_res.txt
	for ((i=1; i <= $NUM; ++i )); do
		python3 ML-classifiers_BsearchBin.py --infile $IN --cls $CLS --labels $LABELS >> ${NAME}_${CLS}_${L}_${NUM}_res.txt
	done
		
	exit 0	
