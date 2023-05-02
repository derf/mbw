EXTRA_CFLAGS =
ifdef pthread
	EXTRA_CFLAGS = -DMULTITHREADED -pthread
endif

mbw: mbw.c
	gcc -Wall -Wextra -pedantic -O2 ${EXTRA_CFLAGS} -o mbw mbw.c

.PHONY: clean
clean:
	rm -f mbw
