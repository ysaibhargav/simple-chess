#/usr/bin/env bash
# generate jobs in batch

threads=(1) # The number of threads 
inputs=(puzzles/4.txt) # The name of the input files
depths=(3)
rm -f *.job

for f in ${inputs[@]}
do
    for t in ${threads[@]}
    do
	for d in ${depths[@]}
	do
	    ../scripts/generate_jobs.sh $t $f $d
	done
    done
done