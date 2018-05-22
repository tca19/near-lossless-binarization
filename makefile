CC=gcc
CFLAGS=-Wall -ansi -pedantic -O3 -march=native -lm

all: similarity_binary topk_binary similarity_real topk_real

similarity_binary: similarity_binary.c
	$(CC) -o similarity_binary similarity_binary.c $(CFLAGS)

topk_binary: topk_binary.c
	$(CC) -o topk_binary topk_binary.c $(CFLAGS)

similarity_real: similarity_real.c
	$(CC) -o similarity_real similarity_real.c $(CFLAGS)

topk_real: topk_real.c
	$(CC) -o topk_real topk_real.c $(CFLAGS)

clean:
	-rm *.o similarity_binary topk_binary similarity_real topk_real
