CC      = gcc
CFLAGS  = -ansi -pedantic -lm -pthread -Ofast -funroll-loops -lblas
CFLAGS += -Wall -Wextra -Wno-unused-result

all: binarize similarity_binary

binarize: binarize.c
	$(CC) $(CFLAGS) binarize.c -o binarize

similarity_binary: similarity_binary.o hashtab.o spearman.o file_process.o
	$(CC) $(CFLAGS) similarity_binary.o hashtab.o spearman.o file_process.o \
	-o similarity_binary

similarity_binary.o: similarity_binary.c utils.h
	$(CC) $(CFLAGS) -c similarity_binary.c

hashtab.o: hashtab.c utils.h
	$(CC) $(CFLAGS) -c hashtab.c

spearman.o: spearman.c
	$(CC) $(CFLAGS) -c spearman.c

file_process.o: file_process.c
	$(CC) $(CFLAGS) -c file_process.c

clean:
	-rm *.o binarize similarity_binary
