#!/bin/sh

set -e

for ARRSIZE in 4 8 16 32 64 128 256 512 1024 2048 4096 8192; do
	./mbw -t0 ${ARRSIZE}
	./mbw -t1 ${ARRSIZE}
	for blocksize in 8 16 32 64 128 256 512 1024 \
			$((1024*2)) $((1024*4)) $((1024*8)) $((1024*16)) $((1024*32)) $((1024*64)) \
			$((1024*128)) $((1024*256)) $((1024*512)) $((1024*1024)) \
			$((1024*1024*2)) $((1024*1024*4)) $((1024*1024*8)) $((1024*1024*16)); do
		./mbw -b $blocksize -t2 ${ARRSIZE} || true
	done
done
