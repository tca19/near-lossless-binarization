#include <math.h>
#include <stdio.h>
#include <time.h>
#include "utils.h"

#define DATADIR "datasets/"

/* cosine_sim: return cosine similarity between vector v1 and v2 */
float cosine_sim(const void *v1, const void *v2, const int n_dims)
{
	static float *ar1, *ar2;
	float dot, norm1, norm2;
	int i;

	ar1 = (float *) v1;
	ar2 = (float *) v2;
	dot = norm1 = norm2 = 0;

	for (i = 0;  i++ < n_dims; ++ar1, ++ar2)
	{
		dot   += *ar1 * *ar2;
		norm1 += *ar1 * *ar1;
		norm2 += *ar2 * *ar2;
	}

	return dot / sqrt(norm1 * norm2);
}

int main(int argc, char *argv[])
{
	long n_vecs;    /* #vectors   */
	int n_dims;     /* #dimension */
	float **embedding;
	clock_t start, end;

	if (argc != 2)
	{
		printf("usage: ./similarity_real EMBEDDING\n");
		return 1;
	}

	start = clock();
	create_vocab(DATADIR);
	end = clock();
	printf("create_vocab(): %fs\n", (double) (end-start) / CLOCKS_PER_SEC);

	start = clock();
	embedding = load_real_vectors(*++argv, &n_vecs, &n_dims);
	end = clock();
	printf("load_vectors(): %fs\n", (double) (end-start) / CLOCKS_PER_SEC);

	start = clock();
	evaluate(DATADIR, (void**) embedding, n_dims, cosine_sim);
	end = clock();
	printf("evaluate(): %fs\n", (double) (end-start) / CLOCKS_PER_SEC);

	return 0;
}
