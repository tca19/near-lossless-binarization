# Copyright (c) 2019-present, All rights reserved.
# Written by Julien Tissier <30314448+tca19@users.noreply.github.com>
#
# This file is part of the "Near-lossless Binarization of Word Embeddings"
# software (https://github.com/tca19/near-lossless-binarization).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License at the root of this repository for
# more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

CC      = gcc
CFLAGS  = -g -Wall -Wextra -Wno-unused-result
LDLIBS  = -lblas -lm

all: binarize similarity_binary topk_binary

# who depends on cblas library (-lblas) ? -- only binarize.c
# who depends on math library (-lm) ? -- only spearman.c and binarize.c
binarize: binarize.c
	$(CC) binarize.c -o binarize $(CFLAGS) $(LDLIBS)

# $^ is a shortcut that means 'all the prerequisites'
similarity_binary: similarity_binary.o hashtab.o spearman.o file_process.o
	$(CC) $^ -o similarity_binary $(CFLAGS) $(LDLIBS)

topk_binary: topk_binary.c
	$(CC) topk_binary.c -o topk_binary $(CFLAGS)

similarity_binary.o: similarity_binary.c utils.h
	$(CC) $(CFLAGS) -c similarity_binary.c

hashtab.o: hashtab.c utils.h
	$(CC) $(CFLAGS) -c hashtab.c

spearman.o: spearman.c
	$(CC) $(CFLAGS) -c spearman.c

file_process.o: file_process.c
	$(CC) $(CFLAGS) -c file_process.c

clean:
	-rm *.o binarize similarity_binary topk_binary
