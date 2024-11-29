#!/bin/sh

mkdir -p log/${HOST}
fn=log/${HOST}/read

make -B numa=1 pthread=1

parallel -j1 --eta --joblog ${fn}.joblog --resume --header : \
	./mbw -a {ram_in} -b {ram_out} -c {cpu} -n 10 -N {nr_threads} -t4 4096 \
	::: ram_in $(seq 0 15) \
	:::+ ram_out $(seq 0 15) \
	::: cpu $(seq 0 7) \
	::: nr_threads $(seq 1 16) \
>> ${fn}.txt
