#include <stdio.h>
#include <time.h>
#include "utils.h"

#define DATADIR "datasets/"

/* binary_sim: return the Sokal-Michener binary similarity (#common / #bits) */
float binary_sim(const void *v1, const void *v2, const int n_long)
{
	int n, i;
	static unsigned long *ar1, *ar2;

	ar1 = (unsigned long *) v1;
	ar2 = (unsigned long *) v2;

	for (n = 0, i = 0; i++ < n_long; ++ar1, ++ar2)
		/* need the ~ because *ar1 ^ *ar2 sets the bit to 0 if same bit */
		n += __builtin_popcountl(~*ar1 ^ *ar2);

	return n / (float) (sizeof(long) * 8 * n_long);
}

int main(int argc, char *argv[])
{
	int n_bits, n_long;         /* #bits per vector, #long per array */
	long n_vecs;                /* #vectors in embedding file */
	unsigned long **embedding;
	clock_t start, end;

	if (argc != 2)
	{
		printf("usage: ./similarity_binary EMBEDDING\n");
		return 1;
	}

	start = clock();
	create_vocab(DATADIR);
	end = clock();
	printf("create_vocab(): %fs\n", (double) (end-start) / CLOCKS_PER_SEC);

	start = clock();
	embedding = load_binary_vectors(*++argv, &n_vecs, &n_bits, &n_long);
	end = clock();
	printf("load_vectors(): %fs\n", (double) (end-start) / CLOCKS_PER_SEC);

	start = clock();
	evaluate(DATADIR, (void**) embedding, n_long, binary_sim);
	end = clock();
	printf("evaluate(): %fs\n", (double) (end-start) / CLOCKS_PER_SEC);

	return 0;
}
