#!/bin/sh

mkdir -p log
fn=log/${HOST}-memcpy

make -B numa=1 pthread=1 avx512=1

parallel -j1 --eta --joblog ${fn}.joblog --resume --header : \
	./mbw -a {ram_in} -b {ram_out} -c {cpu} -n 10 -N {nr_threads} -t0 4096 \
	::: ram_in $(seq 0 15) \
	::: ram_out $(seq 0 15) \
	::: cpu $(seq 0 7) \
	::: nr_threads $(seq 1 16) \
>> ${fn}.txt

fn=log/${HOST}-copy-avx512

parallel -j1 --eta --joblog ${fn}.joblog --resume --header : \
	./mbw -a {ram_in} -b {ram_out} -c {cpu} -n 10 -N {nr_threads} -t3 4096 \
	::: ram_in $(seq 0 15) \
	::: ram_out $(seq 0 15) \
	::: cpu $(seq 0 7) \
	::: nr_threads $(seq 1 16) \
>> ${fn}.txt
