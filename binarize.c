#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAXWORDLEN 128       /* buffer size when reading words of embedding */

/* read a word from ̣`fp` into `buffer`; read at most MAXWORDLEN characters */
void read_word(FILE *fp, char **buffer)
{
	static char tmp[MAXWORDLEN];
	static size_t len;
	int i = 0;

	/* skip white spaces (space or line feed (ascii code 0x0a)) */
	while (isspace((tmp[i] = getc_unlocked(fp))))
		;

	++i; /* move one position because tmp[i] is not a white space */
	while ((tmp[i] = getc_unlocked(fp)) != ' ' && i < MAXWORDLEN-1)
		++i;
	tmp[i] = '\0';

	/* copy tmp into buffer; need to allocate memory for that */
	len = strlen(tmp);
	if ((*buffer = malloc(len + 1)) == NULL)
	{
		fprintf(stderr, "read_word: can't allocate memory for %s\n",
		        tmp);
		exit(1);
	}
	memcpy(*buffer, tmp, len+1);
}

/* read and return a float value from ̣`fp`, handle scientific notation */
float read_float(FILE *fp)
{
	float val, power, power_e;
	int sign, exponent;
	char c;

	/* skip white spaces */
	while (isspace((c= getc_unlocked(fp))))
		;

	/* handle optional sign */
	sign = (c == '-') ? -1 : 1;
	if (c == '+' || c == '-')
		c = getc_unlocked(fp);

	/* get integer part */
	for (val = 0.0; isdigit(c); c = getc_unlocked(fp))
		val = 10.0 * val + (c - '0');

	/* get decimal part */
	if (c == '.') c = getc_unlocked(fp);
	for (power = 1.0; isdigit(c); c = getc_unlocked(fp))
	{
		val = 10.0 * val + (c - '0');
		power *= 10.0;
	}

	/* get scientific notation part */
	if (c == 'e' || c == 'E') c = getc_unlocked(fp);
	/* if e (or E) is followed by '-', it means we need to divide the float
	 * value by a power of 10. Otherwise, we divide it by a power of 0.1
	 * (i.e. multiply by a power of 10) */
	power_e = (c == '-') ? 10 : 0.1;
	if (c == '-' || c == '+') c = getc_unlocked(fp);

	for (exponent = 0; isdigit(c); c = getc_unlocked(fp))
		exponent = 10 * exponent + (c - '0');
	while (exponent-- > 0)
		power *= power_e;

	return sign * val / power;
}

/* load the list of words and vectors from `filename`; return the embedding */
float *load_embedding(const char *filename, char ***words,
		      long *n_vecs, int *n_dims)
{
	int i;
	long index;
	FILE *fp;                  /* to open the vector file */
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
			vec[i] = read_float(fp);
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

/* transform the real-value word vectors of `embedding` into binary vectors */
void binarize(float *embedding, long n_vecs, int n_dims, int n_bits)
{
	float *W, *C;
	float norm_W, norm_C;
	int i;

	/* W is a (n_bits, n_dims) matrix, C is a (n_dims) vector */
	W = calloc(n_dims * n_bits, sizeof *W);
	C = calloc(n_dims, sizeof *C);

	/* initialize W and C with random float values in [-0.5, 0.5] */
	srand(0);
	for (i = 0, norm_W = 0.0f; i < n_dims * n_bits; ++i)
	{
		W[i]    = ((float) rand() / RAND_MAX) - 0.5f;
		norm_W += W[i];
	}
	for (i = 0, norm_C = 0.0f; i < n_dims; ++i)
	{
		C[i] = ((float) rand() / RAND_MAX) - 0.5f;
		norm_C += C[i];
	}

	/* normalize the W matrix and C vector */
	for (i = 0; i < n_dims * n_bits; ++i)
		W[i] /= norm_W;
	for (i = 0; i < n_dims; ++i)
		C[i] /= norm_C;

	free(W);
	free(C);
}

int main(int argc, char *argv[])
{
	char **words;
	float *embedding;
	long n_vecs;
	int n_dims;

	embedding = load_embedding(argv[argc-1], &words, &n_vecs, &n_dims);
	binarize(embedding, n_vecs, n_dims, 256);

	destroy_word_list(words, n_vecs);
	free(embedding); /* `embedding` is created with a single calloc */
	return 0;
}
