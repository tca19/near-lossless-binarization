#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAXWORDLEN 32     /* buffer size when reading words of embedding */

/* load the list of words and vectors from `filename`; return the embedding */
float **load_embedding(const char *filename, char ***words,
		       long *n_vecs, int *n_dims)
{
	int i;
	long index;
	FILE *fp;                  /* to open the vector file */
	char buffer[MAXWORDLEN];   /* to read the word of each vector */
	float **vec;               /* to store the word vectors */
	*words = NULL;

	if ((fp = fopen(filename, "r")) == NULL)
	{
		fprintf(stderr, "load_embedding: can't open %s\n", filename);
		exit(1);
	}

	/* n_vecs and n_dims are pointers, no need of & */
	if (fscanf(fp, "%ld %d", n_vecs, n_dims) != 2)
	{
		fprintf(stderr, "load_embedding: first line of %s should "
		        "contain the number of words in file and the dimension "
			"of vectors\n", filename);
		exit(1);
	}

	/* `words` is supposed to be an array of strings (so char**) but we are
	 * passing it by reference to directly modify the variable passed as a
	 * parameter, so one more level of indirection. `*word` is the content
	 * of the passed pointer (the actual array of strings) */
	if ((*words = calloc(*n_vecs, sizeof *words)) == NULL)
	{
		fprintf(stderr, "load_embedding: can't allocate memory for "
		        "words\n");
		exit(1);
	}

	if ((vec = calloc(*n_vecs, sizeof *vec)) == NULL)
	{
		fprintf(stderr, "load_embedding: can't allocate memory for "
		        "embedding\n");
		exit(1);
	}

	/* start reading the word and vector values */
	index = 0;
	while (fscanf(fp, "%s", buffer) > 0)
	{
		(*words)[index] = strdup(buffer);
		if ((vec[index] = calloc(*n_dims, sizeof **vec)) == NULL)
			continue;

		for (i = 0; i < *n_dims; ++i)
			fscanf(fp, "%f", vec[index] + i);
		++index;
	}

	fclose(fp);
	return vec;
}

int main(int argc, char *argv[])
{
	char **words;
	float **embedding;
	long n_vecs;
	int n_dims;

	embedding = load_embedding(argv[1], &words, &n_vecs, &n_dims);
	printf("%s %s", words[0], words[1]);

	return 0;
}
