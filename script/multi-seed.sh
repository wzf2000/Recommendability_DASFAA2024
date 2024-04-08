# print the command
set -x
python test.py -l $language -d $dataset -m $model -s $size -t $template -b $bs -e 20 --new $suffix --seed 2019
python test.py -l $language -d $dataset -m $model -s $size -t $template -b $bs -e 20 --new $suffix --seed 2020
python test.py -l $language -d $dataset -m $model -s $size -t $template -b $bs -e 20 --new $suffix --seed 2021
python test.py -l $language -d $dataset -m $model -s $size -t $template -b $bs -e 20 --new $suffix --seed 2022
python test.py -l $language -d $dataset -m $model -s $size -t $template -b $bs -e 20 --new $suffix --seed 2023