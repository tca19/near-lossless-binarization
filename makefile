CC=gcc
CFLAGS=-Wall -ansi -pedantic -O3 -march=native -lm

all: similarity_binary

similarity_binary: similarity_binary.o hashtab.o spearman.o file_process.o
	$(CC) $(CFLAGS) similarity_binary.o hashtab.o spearman.o file_process.o \
	-o similarity_binary

similarity_real: similarity_real.o hashtab.o spearman.o
	$(CC) $(CFLAGS) similarity_real.o hashtab.o spearman.o -o similarity_real

similarity_binary.o: similarity_binary.c utils.h
	$(CC) $(CFLAGS) -c similarity_binary.c

similarity_real.o: similarity_real.c utils.h
	$(CC) $(CFLAGS) -c similarity_real.c

hashtab.o: hashtab.c utils.h
	$(CC) $(CFLAGS) -c hashtab.c

spearman.o: spearman.c
	$(CC) $(CFLAGS) -c spearman.c

file_process.o: file_process.c
	$(CC) $(CFLAGS) -c file_process.c

clean:
	-rm *.o similarity_binary similarity_real
