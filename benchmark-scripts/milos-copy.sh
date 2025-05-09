#!/bin/sh

mkdir -p log/${HOST}

make -B numa=1 pthread=1 avx512=1

fn=log/${HOST}/copy-memcpy
echo "\n${fn}\n"
echo "mbw $(git describe --all --long) $(git rev-parse HEAD)" >> ${fn}.txt
parallel -j1 --eta --joblog ${fn}.joblog --resume --header : \
	./mbw -a {ram_in} -b {ram_out} -c {cpu} -n 5 -N {nr_threads} -t0 4096 \
	::: ram_in $(seq 0 17) \
	::: ram_out $(seq 0 17) \
	::: cpu $(seq 0 7) \
	::: nr_threads $(seq 1 16) \
>> ${fn}.txt

fn=log/${HOST}/copy-avx512
echo "\n${fn}\n"
echo "mbw $(git describe --all --long) $(git rev-parse HEAD)" >> ${fn}.txt
parallel -j1 --eta --joblog ${fn}.joblog --resume --header : \
	./mbw -a {ram_in} -b {ram_out} -c {cpu} -n 5 -N {nr_threads} -t3 4096 \
	::: ram_in $(seq 0 17) \
	::: ram_out $(seq 0 17) \
	::: cpu $(seq 0 7) \
	::: nr_threads $(seq 1 16) \
>> ${fn}.txt
