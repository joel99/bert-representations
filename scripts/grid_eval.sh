#!/bin/bash
for i in {s1,s2,s3}
do
    for j in {mp,mb,ms,pb,ps,bs}
    do
    echo ${i}_${j}
    sbatch ./scripts/eval.sh ${i}_${j} all
    done
done