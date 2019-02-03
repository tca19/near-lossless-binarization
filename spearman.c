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

#include <stdlib.h>  /* malloc(), qsort(), free() */
#include <string.h>  /* memcpy() */
#include <math.h>    /* sqrt() */

/* cmpfloat: used in qsort to compare two floats */
int cmpfloat(const void *a, const void *b)
{
	return *(float*)a >= *(float*)b ? 1 : -1;
}

/* get_rank: return the rank of v in sorted array ar of length len. If v appears
 *           multiple time in ar, return average rank of all occurrences. */
float get_rank(float *ar, float v, int len)
{
	int low, mid, high;

	low = 0;
	high = len;

	while (ar[mid = (low+high)/2] != v && low <= high)
	{
		if (ar[mid] > v)
			high = mid-1;
		else
			low = mid+1;
	}

	/* find the left-most cell with same value as v */
	low = mid;
	while (low > 0 && ar[low-1] == v)
		--low;

	/* find the right-most cell with same value as v */
	high = mid;
	while (high < len && ar[high] == v)
		++high;

	return low + (high-low) / 2.0;
}

/* ranks: return the rank of each value in array ar of length len */
float* ranks(float *ar, int len)
{
	int i;
	float *copy;

	/* sort to get the ranks, but need original order, so save a copy */
	copy = malloc(len * sizeof *copy);
	memcpy(copy, ar, len * sizeof *copy);
	qsort(ar, len, sizeof(float), cmpfloat);

	for (i = 0; i < len; ++i)
		copy[i] = get_rank(ar, copy[i], len) + 1.0;

	return copy;
}

/* has_ties: return 1 if there are consecutive same values in ar, 0 otherwise */
int has_ties(float *ar, int n)
{
	int k;
	for (k = 0; k < n-1; ++k)
		if (ar[k] == ar[k+1])
			return 1;
	return 0;
}

/* spearman_coef: return Spearman rank correlation coefficient between array ar1
 *                and ar2, both of length len. */
float spearman_coef(float *ar1, float *ar2, int len)
{
	int i;
	float mean1, mean2, corr1, corr2, d, rho;
	float *ranks1, *ranks2;

	d = rho = 0;
	ranks1 = ranks(ar1, len);
	ranks2 = ranks(ar2, len);

	if (has_ties(ranks1, len) || has_ties(ranks2, len))
	{
		mean1 = mean2 = corr1 = corr2 = 0;

		for (i = 0; i < len; ++i)
		{
			mean1 += ranks1[i];
			mean2 += ranks2[i];
		}
		mean1 /= len;
		mean2 /= len;

		for (i = 0; i < len; ++i)
		{
			d += (ranks1[i] - mean1) * (ranks2[i] - mean2);
			corr1 += (ranks1[i] - mean1) * (ranks1[i] - mean1);
			corr2 += (ranks2[i] - mean2) * (ranks2[i] - mean2);
		}

		rho = d / (sqrt(corr1 * corr2));
	}
	else
	{
		d = 0;
		for (i = 0; i < len; ++i)
			d += (ranks1[i] - ranks2[i]) * (ranks1[i] - ranks2[i]);
		rho = 1.0 - (6.0*d) / (len * (len*len - 1.0));
	}

	free(ranks1);
	free(ranks2);
	return rho;
}
