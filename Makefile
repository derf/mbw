EXTRA_CFLAGS =
EXTRA_LIBS =

native ?= 1
pthread ?= 0
numa ?= 0
avx512 ?= 0
debug ?= 0

ifeq (${native}, 1)
	EXTRA_CFLAGS += -march=native
endif

ifeq (${pthread}, 1)
	EXTRA_CFLAGS += -DMULTITHREADED -pthread
endif

ifeq (${numa}, 1)
	EXTRA_CFLAGS += -DNUMA
	EXTRA_LIBS += -lnuma
endif

ifeq (${avx512}, 1)
	EXTRA_CFLAGS += -DHAVE_AVX512
endif

ifeq (${debug}, 1)
	EXTRA_CFLAGS += -ggdb
endif

mbw: mbw.c
	gcc -Wall -Wextra -pedantic -O3 ${EXTRA_CFLAGS} -o mbw mbw.c ${EXTRA_LIBS}

.PHONY: clean
clean:
	rm -f mbw
