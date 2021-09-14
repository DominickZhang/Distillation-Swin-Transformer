#!/bin/bash
if [ -n "$1" ]
then
    name="$1"
else
    name="swin_transformer_distill.tar.gz"
fi
tar --exclude='./datasets' --exclude='./output' --exclude='./trained_models' --exclude='./pkgs' --exclude='./*.tar.gz' --exclude='./apex' -zcvf $name .
hadoop fs -rm "/edu/jinnian/$name"
hadoop fs -put $name /edu/jinnian/
echo 'Done!'