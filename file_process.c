/* Copyright (c) 2019-present, All rights reserved.
 * Written by Julien Tissier <30314448+tca19@users.noreply.github.com>
 *
 * This file is part of the "Near-lossless Binarization of Word Embeddings"
 * software (https://github.com/tca19/near-lossless-binarization).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License at the root of this repository for
 * more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy, strcmp, strcat */
#include "utils.h"

#define MAXLINES   5000 /* maximum number of pairs in an evaluation dataset */
#define MAXLENPATH 256  /* maximum length to access an evaluation dataset */
#define MAXLENWORD 256  /* maximum length of a word in an embedding file */

/* create_vocab: read each file in dirname to create vocab of unique words */
void create_vocab(const char *dirname)
{
	DIR *dp;
	FILE *fp;
	struct dirent *ent;
	char filepath[MAXLENPATH], word1[MAXLENWORD], word2[MAXLENWORD];
	float val;

	if ((dp = opendir(dirname)) == NULL)
	{
		fprintf(stderr, "create_vocab: can't open %s\n", dirname);
		exit(1);
	}

	while ((ent = readdir(dp)) != NULL)
	{
		if (strcmp(ent->d_name, ".") == 0
		 || strcmp(ent->d_name, "..") == 0)
			continue;

		strcpy(filepath, dirname);
		strcat(filepath, ent->d_name);
		if ((fp = fopen(filepath, "r")) == NULL)
		{
			fprintf(stderr, "create_vocab: can't open file %s\n",
			        filepath);
			continue;
		}

		while (fscanf(fp, "%s %s %f", word1, word2, &val) > 0)
		{
			lower(word1);
			lower(word2);

			/* No need to save the index of each word because we
			 * only iterate over the datasets to build the vocab,
			 * i.e. the list of word vectors we have to load from
			 * the embedding file.  For this reason, the second
			 * parameter is 0. */
			add_word(word1, 0);
			add_word(word2, 0);
		}
		fclose(fp);
	}
	closedir(dp);
}

/* load_vectors: read the vector file `name`. If `load_all_vectors` is set
 *               (i.e. not zero), read all word vectors from the file and
 *               return the binary embedding matrix (also add each word into
 *               the hashtab). If `load_all_vectors` is 0, only load the
 *               vectors of words already in hashtab (hashtab should have been
 *               populated with create_vocab()). In this case, no words are
 *               added to hashtab. Each binary word vector is loaded as an
 *               array of `long`, so to represent a vector of 256 bits it
 *               requires an array of 4 `long`. */
unsigned long **load_vectors(const char *name, long *n_vecs, int *n_bits,
	                     int *n_long, const int load_all_vectors)
{
	int i;
	long index;
	FILE *fp;                  /* to open vector file */
	char word[MAXLENWORD];     /* to read the word of each line in file */
	unsigned long **vec, tmp;  /* to store the binary embeddings values */

	if ((fp = fopen(name, "r")) == NULL)
	{
		fprintf(stderr, "load_vectors: can't open %s\n", name);
		exit(1);
	}

	if (fscanf(fp, "%ld %d", n_vecs, n_bits) <= 0)
	{
		fprintf(stderr, "load_vectors: can't read number of bits\n");
		exit(1);
	}

	/* Only allocate memory to save the pointers to vectors. The memory
	 * needed to load binary values is done individually for each vector a
	 * bit further (because in the case of the similarity_binary program,
	 * only the words from the evaluation datasets are loaded, so no need
	 * to allocate a lot of memory that we are not going to use). */
	*n_long = *n_bits / (sizeof(long) * 8);
	if ((vec = calloc(*n_vecs, sizeof *vec)) == NULL)
		return NULL;

	/* Allocate memory for the index->word array (declared in hashtab.c).
	 * It will contain the same number of words as the number of vectors in
	 * the embedding file. */
	if ((words = calloc(*n_vecs, sizeof *words)) == NULL)
		fprintf(stderr, "load_vectors: no memory for index<=>word\n");

	while (fscanf(fp, "%s", word) > 0)
	{
		index = get_index(word);

		if (load_all_vectors)
		{
			/* Word vector has already been loaded, skip it. To
			 * skip it, read all its vector values into the garbage
			 * variable `tmp` then go to next one. */
			if (index > 0)
			{
				for (i = 0; i < *n_long; ++i)
					fscanf(fp, "%lu", &tmp);
				continue;
			}

			/* Else, add it into the hashtab with `add_word()`. The
			 * word is also added into the index->word array with
			 * the second parameter set to 1 (done in the function
			 * `add_word()`). */
			else
			{
				/* When a word is added into the hash table
				 * with `add_word()`, its index is set to
				 * n_words (variable from hashtab.c). It is the
				 * current number of words already in hashtab.
				 * It is automatically increased within the
				 * function `add_word()`. This index is then
				 * used to know which row of the embedding
				 * matrix should be filled with values from the
				 * embedding file. */
				index = n_words;
				add_word(word, 1);
			}
		}
		else /* Do not load all vectors, only those in hashtab. */
		{
			if (index == -1) /* word not in hashtab; skip it */
			{
				for (i = 0; i < *n_long; ++i)
					fscanf(fp, "%lu", &tmp);
				continue;
			}
		}

		/* Allocate memory to read vector values and load them. */
		if ((vec[index] = calloc(*n_long, sizeof **vec)) == NULL)
			continue;
		for (i = 0; i < *n_long; ++i)
			fscanf(fp, "%lu", vec[index]+i);
	}

	fclose(fp);
	return vec;
}

/* evaluate: compute Spearman coefficient for each file in dirname */
void evaluate(const char *dirname, void **vec, int n_dim,
	      float (*sim)(const void*, const void*, const int))
{
	DIR *dp;
	FILE *fp;
	struct dirent *ent;
	char filepath[MAXLENPATH], word1[MAXLENWORD], word2[MAXLENWORD];
	float val;
	long index1, index2, found, nlines;
	float *simfile, *simvec;

	if ((simfile = malloc(MAXLINES * sizeof *simfile)) == NULL
	 || (simvec  = malloc(MAXLINES * sizeof *simvec))  == NULL)
	{
		fprintf(stderr, "evaluate: can't allocate memory to store"
		        " similarity values in datasets.\n");
		exit(1);
	}

	if ((dp = opendir(dirname)) == NULL)
	{
		fprintf(stderr, "evaluate: can't open %s\n", dirname);
		exit(1);
	}

	printf("%-12s | %-8s | %3s\n", "Filename", "Spearman", "OOV");
	printf("==============================\n");
	while ((ent = readdir(dp)) != NULL)
	{
		if (strcmp(ent->d_name, ".") == 0
		 || strcmp(ent->d_name, "..") == 0)
			continue;

		strcpy(filepath, dirname);
		strcat(filepath, ent->d_name);
		if ((fp = fopen(filepath, "r")) == NULL)
		{
			fprintf(stderr, "evaluate: can't open file %s\n",
			        filepath);
			continue;
		}

		found = nlines = 0;
		while (fscanf(fp, "%s %s %f", word1, word2, &val) > 0
		    && nlines < MAXLINES)
		{
			++nlines;
			lower(word1);
			lower(word2);
			index1 = get_index(word1);
			index2 = get_index(word2);

			if (vec[index1] == NULL || vec[index2] == NULL)
				continue;

			simfile[found] = val;
			simvec[found] = sim(vec[index1], vec[index2], n_dim);
			++found;
		}

		val = spearman_coef(simfile, simvec, found);
		printf("%-12s | %8.3f | %3ld%%\n", ent->d_name, val,
		       (nlines - found) * 100 /  nlines);
		fclose(fp);
	}
	closedir(dp);
}

/* binary_sim: return the Sokal-Michener binary similarity (#common / #bits) */
float binary_sim(const void *v1, const void *v2, const int n_long)
{
	int n, i;
	static unsigned long *ar1, *ar2;

	ar1 = (unsigned long *) v1;
	ar2 = (unsigned long *) v2;

	/* need the ~ because *ar1 ^ *ar2 sets the bit to 0 if same bit */
	for (n = 0, i = 0; i++ < n_long; ++ar1, ++ar2)
		n += __builtin_popcountl(~*ar1 ^ *ar2);

	return n / (float) (sizeof(long) * 8 * n_long);
}
