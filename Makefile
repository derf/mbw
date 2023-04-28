mbw: mbw.c
	gcc -Wall -Wextra -pedantic -O2 -o mbw mbw.c

.PHONY: clean
clean:
	rm -f mbw
