#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAXWORDLEN 128       /* buffer size when reading words of embedding */

/* read a word from ̣`fp` into `buffer`; read at most MAXWORDLEN characters */
void read_word(FILE *fp, char **buffer)
{
	static char tmp[MAXWORDLEN];
	int i = 0;

	/* skip white spaces (space or line feed (ascii code 0x0a)) */
	while (isspace((tmp[i] = getc_unlocked(fp))))
		;

	++i; /* move one position because tmp[i] is not a white space */
	while ((tmp[i] = getc_unlocked(fp)) != ' ' && i < MAXWORDLEN-1)
		++i;

	tmp[i] = '\0';
	*buffer = strdup(tmp);
}

/* load the list of words and vectors from `filename`; return the embedding */
float *load_embedding(const char *filename, char ***words,
		       long *n_vecs, int *n_dims)
{
	int i;
	long index;
	FILE *fp;                  /* to open the vector file */
	char buffer[MAXWORDLEN];   /* to read the word of each vector */
	float *vec;                /* to store the word vectors */

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
	 * parameter, so one more level of indirection (that's why it is
	 * char***). `*word` is the content of the passed pointer (the actual
	 * array of strings) */
	if ((*words = calloc(*n_vecs, sizeof **words)) == NULL)
	{
		fprintf(stderr, "load_embedding: can't allocate memory for "
		        "words\n");
		exit(1);
	}

	if ((vec = calloc(*n_vecs * *n_dims, sizeof *vec)) == NULL)
	{
		fprintf(stderr, "load_embedding: can't allocate memory for "
		        "embedding\n");
		exit(1);
	}

	/* start reading the word and vector values */
	index = 0;
	while (!feof(fp))
	{
		read_word(fp, *words + index);
		for (i = *n_dims * index; i < *n_dims * (index+1); ++i)
			fscanf(fp, "%f", vec + i);
		++index;
	}

	fclose(fp);
	return vec;
}

/* free the memory used to store the list of words */
void destroy_word_list(char **words, long n_vecs)
{
	/* each cell of `words` is a string created with strdup. Need to free
	 * the memory allocated for each cell */
	while (n_vecs--)
		free(words[n_vecs]);
	free(words);
}

int main(int argc, char *argv[])
{
	char **words;
	float *embedding;
	long n_vecs;
	int n_dims;

	embedding = load_embedding(argv[1], &words, &n_vecs, &n_dims);

	destroy_word_list(words, n_vecs);
	free(embedding); /* `embedding` is created with a single calloc */
	return 0;
}
