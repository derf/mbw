EXTRA_CFLAGS =
EXTRA_LIBS =
ifdef pthread
	EXTRA_CFLAGS += -DMULTITHREADED -pthread
endif

ifdef numa
	EXTRA_CFLAGS += -DNUMA
	EXTRA_LIBS += -lnuma
endif

mbw: mbw.c
	gcc -Wall -Wextra -pedantic -O2 ${EXTRA_CFLAGS} -o mbw mbw.c ${EXTRA_LIBS}

.PHONY: clean
clean:
	rm -f mbw
