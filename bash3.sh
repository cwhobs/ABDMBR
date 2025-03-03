#!/bin/bash
###
# @Author: Yidan Liu 1334252492@qq.com
# @Date: 2024-09-03 14:01:41
# @LastEditors: Yidan Liu 1334252492@qq.com
# @LastEditTime: 2025-02-21 11:45:39
# @FilePath: /MB-HGCN-main_3/bash.sh
# @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
###

# lr=('0.01' '0.005' '0.001' '0.0005')
# reg_weight=('0.01' '0.001' '0.0001')
emb_size=(64)
lr=('0.0005')
reg_weight=('1e-3')
distill_userK=(10)
distill_topK=(10)

dataset=('tmall')

for name in ${dataset[@]}; do
    for l in ${lr[@]}; do
        for reg in ${reg_weight[@]}; do
            for emb in ${emb_size[@]}; do
                for topk in ${distill_topK[@]}; do
                    for userk in ${distill_userK[@]}; do
                        echo 'start train: '$name
                        $(
                            python main_1_prj_tmall_t_2.py \
                                --lr ${l} \
                                --reg_weight ${reg} \
                                --data_name $name \
                                --embedding_size $emb --distill_topK $topk --distill_userK $userk
                        )
                        echo 'train end: '$name
                    done
                done
            done
        done
    done
done
