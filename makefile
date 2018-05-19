CC=gcc
CFLAGS=-Wall -ansi -pedantic -O3 -march=native -lm

all: similarity_binary topk_binary

similarity_binary: similarity_binary.c
	$(CC) -o similarity_binary similarity_binary.c $(CFLAGS)

topk_binary: topk_binary.c
	$(CC) -o topk_binary topk_binary.c $(CFLAGS)

clean:
	-rm *.o similarity_binary topk_binary
