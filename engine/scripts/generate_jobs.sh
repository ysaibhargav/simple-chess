#/usr/bin/env bash

# generate the job for latedays
input=$2
threads=$1
depth=$3

if [ ${#} -ne 3 ]; then
  echo "Usage: $0 <threads> <input> <depth>"
else
  inputfile=`basename ${input}`
  strlen=${#inputfile}
  strlen=$(($strlen-4))
  inputfile=${inputfile:0:$strlen}
  curdir=`pwd`
  curdir=${curdir%/templates}
  sed "s:PROGDIR:${curdir}:g" ../scripts/example.job.template > tmp1.job
  sed "s:INPUT:${input}:g" tmp1.job > tmp2.job
  sed "s:DEPTH:${depth}:g" tmp2.job > tmp3.job
  sed "s:THREADS:${threads}:g" tmp3.job > ${USER}_${inputfile}_${threads}.job
  rm -f tmp1.job tmp2.job tmp3.job
fi
