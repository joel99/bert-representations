#!/bin/bash
# for i in {stsb_pos,sst2_pos,pos_sst2}
# # for i in {s1,s2,s3}
# do
#     for j in {eq-seq,eq-sam}
#     # for j in {mp,mb,ms,pb,ps,bs}
#     do
#     echo ${i}_${j}
#     sbatch ./scripts/finetune.sh ${i}_${j}
#     done
# done
for i in {sst2_pos_eq-sam,pos_sst2_eq-sam,stsb_pos_eq-seq}
    do
    sbatch ./scripts/finetune.sh ${i}
done