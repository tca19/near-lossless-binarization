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

#include <stdio.h>       /* fprintf() */
#include <stdlib.h>      /* calloc()  */
#include <time.h>        /* clock()   */
#include "utils.h"

struct neighbor
{
	long index;
	float similarity;
};

/* find_topk: return the k nearest neighbors of word */
struct neighbor *find_topk(const char *word, const int k, const long n_vecs,
		           const int n_long,  unsigned long **vec)
{
	long i, j, index;
	struct neighbor *topk, tmp;

	if ((topk = calloc(k + 1, sizeof *topk)) == NULL)
	{
		fprintf(stderr, "find_topk: can't allocate memory for heap\n");
		exit(1);
	}

	/* word has no vector, can't find its neighbors */
	if ((index = get_index(word)) < 0)
		return NULL;

	for (i = 0; i < n_vecs; ++i)
	{
		/* a word cannot be its nearest neighbor; skip it */
		if (i == index)
			continue;

		/* values in topk are sorted by decreasing similarity. If the
		 * similarity with current vector is greater than minimal
		 * similarity in topk, insert current similarity into topk with
		 * bubble sort */
		topk[k].similarity = binary_sim(vec[index], vec[i], n_long);
		if (topk[k].similarity < topk[k-1].similarity)
			continue;

		for (topk[k].index = i, j = k;
		     j > 0 && topk[j].similarity > topk[j-1].similarity;
		     --j)
		{
			/* swap element j-1 with element j */
			tmp = topk[j-1];
			topk[j-1] = topk[j];
			topk[j] = tmp;
		}
	}
	return topk;
}

int main(int argc, char *argv[])
{
	int n_bits, n_long;         /* #bits per vector, #long per array */
	long n_vecs;                /* #vectors in embedding file */
	unsigned long **embedding;
	struct neighbor *topk;
	int i, k;
	clock_t start, end;

	if (argc < 4)
	{
		printf("usage: ./topk_binary EMBEDDING K QUERY...\n");
		exit(1);
	}

	embedding = load_vectors(*++argv, &n_vecs, &n_bits, &n_long, 1);
	k = atoi(*++argv);
	argc -= 2; /* because already used argument 0 and 1 */

	while (--argc > 0)
	{
		start = clock();
		topk = find_topk(*++argv, k, n_vecs, n_long, embedding);
		end = clock();

		if (topk == NULL)
		{
			printf("%s doesn't have a vector; can't find its"
			       " nearest neighbors.\n\n", *argv);
			continue;
		}

		printf("Top %d closest words of %s\n", k, *argv);
		for (i = 0; i < k; ++i)
			printf("  %-15s %.3f\n", words[topk[i].index],
			                         topk[i].similarity);
		printf("> Query processed in %.3f ms.\n",
		       (double) (end - start) * 1000 / CLOCKS_PER_SEC);
		printf("\n");
	}

	return 0;
}
