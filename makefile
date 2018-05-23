CC=gcc
CFLAGS=-Wall -ansi -pedantic -O3 -march=native -lm

all: similarity_binary

similarity_binary: similarity_binary.o hashtab.o spearman.o
	$(CC) $(CFLAGS) similarity_binary.o hashtab.o spearman.o -o similarity_binary

similarity_binary.o: similarity_binary.c utils.h
	$(CC) $(CFLAGS) -c similarity_binary.c

hashtab.o: hashtab.c utils.h
	$(CC) $(CFLAGS) -c hashtab.c

spearman.o: spearman.c
	$(CC) $(CFLAGS) -c spearman.c

clean:
	-rm *.o similarity_binary
