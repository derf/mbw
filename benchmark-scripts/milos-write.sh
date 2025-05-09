#!/bin/sh

mkdir -p log/${HOST}

make -B numa=1 pthread=1 avx512=1

fn=log/${HOST}/write-64bit
echo "\n${fn}\n"
echo "mbw $(git describe --all --long) $(git rev-parse HEAD)" >> ${fn}.txt
parallel -j1 --eta --joblog ${fn}.joblog --resume --header : \
	./mbw -a {ram_in} -b {ram_out} -c {cpu} -n 10 -N {nr_threads} -t5 4096 \
	::: ram_out $(seq 0 17) \
	:::+ ram_in $(seq 0 17) \
	::: cpu $(seq 0 7) \
	::: nr_threads $(seq 1 16) \
>> ${fn}.txt

fn=log/${HOST}/write-avx512
echo "\n${fn}\n"
echo "mbw $(git describe --all --long) $(git rev-parse HEAD)" >> ${fn}.txt
parallel -j1 --eta --joblog ${fn}.joblog --resume --header : \
	./mbw -a {ram_in} -b {ram_out} -c {cpu} -n 10 -N {nr_threads} -t7 4096 \
	::: ram_out $(seq 0 17) \
	:::+ ram_in $(seq 0 17) \
	::: cpu $(seq 0 7) \
	::: nr_threads $(seq 1 16) \
>> ${fn}.txt
