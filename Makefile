EXTRA_CFLAGS =
EXTRA_LIBS =
ifdef pthread
	EXTRA_CFLAGS += -DMULTITHREADED -pthread
endif

ifdef numa
	EXTRA_CFLAGS += -DNUMA
	EXTRA_LIBS += -lnuma
endif

ifdef avx512
	EXTRA_CFLAGS += -DHAVE_AVX512
endif

ifdef debug
	EXTRA_CFLAGS += -ggdb
endif

mbw: mbw.c
	gcc -Wall -Wextra -pedantic -O3 -march=native ${EXTRA_CFLAGS} -o mbw mbw.c ${EXTRA_LIBS}

.PHONY: clean
clean:
	rm -f mbw
