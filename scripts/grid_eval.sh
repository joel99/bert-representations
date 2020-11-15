#!/bin/bash
for i in {stsb_pos,sst2_pos,pos_sst2}
# for i in {s1,s2,s3}
do
    # for j in {mp,mb,ms,pb,ps,bs}s
    for j in {eq-seq,eq-sam}
    # for j in {mp,mb,ms,pb,ps,bs}
    do
    echo ${i}_${j}
    sbatch ./scripts/extract.sh ${i}_${j} all
    # sbatch ./scripts/eval.sh ${i}_${j} all
    done
done