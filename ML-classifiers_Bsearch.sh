#!/bin/bash

	if [[ $# < 3 ]]; then
		  echo numero di parametri errato!
		  echo uso corretto:
		  echo "$0 <CLS> <DATA_FILE>" "<#RUNS>"
		  exit 1
	fi

	CLS=$1
	IN=$2
	NUM=$3
		
	
#	enviroment activation
#	conda init bash
#	conda activate giovi		

	NAME=$(echo $IN | cut -d. -f1)

	echo "" > ${NAME}_${CLS}_${NUM}_res.txt
	for ((i=1; i <= $NUM; ++i )); do
		python ML-classifiers_Bsearch.py --infile $IN --cls $CLS  >> ${NAME}_${CLS}_${NUM}_res.txt
	done
		
	exit 0	

	
