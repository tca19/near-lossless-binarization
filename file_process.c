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

#define MAXLINES   3500
#define MAXLENPATH 64
#define MAXLENWORD 64

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
			add_word(word1);
			add_word(word2);
		}
		fclose(fp);
	}
	closedir(dp);
}

/* load_vectors: read the vector file, only load the vectors of words present in
 *               hashtab (so at most word_index+1 vectors). Save each vector as
 *               an array of `long`, so to represent a vector of 256 bits
 *               it requires an array of 4 `long`. */
unsigned long **load_binary_vectors(const char *name, long *n_vecs,
	                            int *n_bits, int *n_long)
{
	int i;
	long index;
	FILE *fp;                /* to open vector file */
	char word[MAXLENWORD];   /* to read the word of each line in file */
	unsigned long **vec;     /* to store the binary embeddings */

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

	*n_long = *n_bits / (sizeof(long) * 8);
	if ((vec = calloc(word_index + 1, sizeof *vec)) == NULL)
		return NULL;

	while (fscanf(fp, "%s", word) > 0)
	{
		index = get_index(word);
		if (index == -1)    /* drop the words not in vocab */
			continue;
		if ((vec[index] = calloc(*n_long, sizeof **vec)) == NULL)
			continue;

		for (i = 0; i < *n_long; ++i)
			fscanf(fp, "%lu", vec[index]+i);
	}

	fclose(fp);
	return vec;
}

/* load_vectors: read the vector file, only load the vectors of words present in
 *               hashtab (so at most word_index+1 vectors). */
float **load_real_vectors(const char *name, long *n_vecs, int *n_dims)
{
	int i;
	long index;
	FILE *fp;                /* to open vector file */
	char word[MAXLENWORD];   /* to read the word of each line in file */
	float **vec;             /* to store the embeddings */

	if ((fp = fopen(name, "r")) == NULL)
	{
		fprintf(stderr, "load_vectors: can't open %s\n", name);
		exit(1);
	}

	if (fscanf(fp, "%ld %d", n_vecs, n_dims) <= 0)
	{
		fprintf(stderr, "load_vectors: can't read dimension\n");
		exit(1);
	}

	if ((vec = calloc(word_index + 1, sizeof *vec)) == NULL)
		return NULL;

	while (fscanf(fp, "%s", word) > 0)
	{
		index = get_index(word);
		if (index == -1)    /* drop the words not in vocab */
			continue;
		if ((vec[index] = calloc(*n_dims, sizeof **vec)) == NULL)
			continue;

		for (i = 0; i < *n_dims; ++i)
			fscanf(fp, "%f", vec[index]+i);
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
