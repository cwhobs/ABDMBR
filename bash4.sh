#!/bin/bash
###
 # @Author: Yidan Liu 1334252492@qq.com
 # @Date: 2024-09-03 14:01:41
 # @LastEditors: Yidan Liu 1334252492@qq.com
 # @LastEditTime: 2024-10-15 20:57:51
 # @FilePath: /MB-HGCN-main_3/bash.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 


# lr=('0.01' '0.005' '0.001' '0.0005')
# reg_weight=('0.01' '0.001' '0.0001')
# emb_size=(64)
# lr=('0.0005')
# reg_weight=('1e-3')

layers=(1 2 3 4)
# dataset=('tmall' 'taobao')
dataset=('taobao')
for name in ${dataset[@]}
do
    for lay in ${layers[@]}
     do
                
        echo 'start train: '$name
         `
             python pretrain_1.py \
                --data_name $name \
                --layers $lay
        `
        echo 'train end: '$name
    done
done