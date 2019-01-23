#include <cblas.h>
#include <ctype.h>
#include <math.h>
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

/* return a new memory allocated array of random floats, normalized to 1 */
float *random_array(long size)
{
	float *ar, norm;
	long i;

	ar = calloc(size, sizeof *ar);

	/* initalize ar with random float values in [-0.5, 0.5] */
	for (i = 0, norm = 0.0f; i < size; ++i)
	{
		ar[i] = ((float) rand() / RAND_MAX) - 0.5f;
		norm += ar[i] * ar[i];
	}

	norm = sqrt(norm);
	/* normalize ar */
	for (i = 0; i < size; ++i)
		ar[i] /= norm;
	return ar;
}

/* compute the gradient of the regularization w.r.t. W, update weigths of W */
void apply_regularizarion_gradient(float *W, int m, int n, float lr_reg)
{
	float *T, *copy;
	int i;

	/* T = W'.W - I;
	 * W is a (m,n) matrix, W' is a (n,m) matrix so T is a (n,n) matrix */
	T = calloc(n * n, sizeof *T);

	/* compute T = W'.W */
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
	            n, n, m,
	            1, W, n, W, n,
	            0, T, n);

	/* compute T = T - I */
	for (i = 0, copy = T; i++ < n; copy = copy + n + 1)
		*copy -= 1.0;

	/* gradient matrix is dRdW = 2 * W.T, and W is updated with
	 * W -= lr_reg * dRdW. Compute dRdW, but directly update
	 * the weights of W (the function cblas_dgemm(A, B, C) performs the
	 * matrix operation:  C = alpha * A.B + beta * C) */
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            m, n, n,
	            -2 * lr_reg, W, n, T, n,
	            1, W, n);

	free(T);
	return;
}

/* compute the gradients of the reconstruction loss w.r.t W and C, update the
 * weights of W and C. `embedding` should not be the whole embedding matrix, but
 * the embedding matrix of the batch, so dimension should be (batch_size,n). */
void apply_reconstruction_gradient(float *W, float *C, float *embedding,
		                   int m, int n, int batch_size)
{
	float *latent, *x_hat, *dldC, *dldW, v;
	int i, j;

	/* latent = bin(W.embedding') where x is the stacked vectors of the
	 * batch. W is a (m,n) matrix, embedding is a (batch_size,n) matrix, so
	 * latent is a (m,batch_size) matrix. */
	latent = calloc(m * batch_size, sizeof *latent);

	/* compute latent = bin(W.embedding'). bin() is a function that maps
	 * negative values to 0 and positive values to 1. */
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
	            m, batch_size, n,
	            1, W, n, embedding, n,
	            0, latent, batch_size);
	for (i = 0; i < m * batch_size; ++i)
		latent[i] = (latent[i] > 0) ? 1.0 : 0.0;

	/* x_hat = tanh(W'.latent + C);
	 * W' is a (n,m) matrix, latent is a (m,batch_size) matrix so x_hat is a
	 * (n,batch_size) matrix. C is a (n) vector and is column broadcasted.
	 * (as if C were added to each column of W'.latent) */
	x_hat = calloc(n * batch_size, sizeof *x_hat);

	/* compute x_hat = W'.latent */
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
	            n, batch_size, m,
	            1, W, n, latent, batch_size,
	            0, x_hat, batch_size);

	/* compute x_hat = tanh(x_hat + C) */
	for (i = 0; i < n * batch_size; ++i)
		x_hat[i] = tanh(x_hat[i] + C[i / n]);

	/* dldC = (x_hat' - x) * (1 - x_hat'**2)
	 * No BLAS subroutines implement element-wise matrices substraction,
	 * have to do it manually. */
	dldC = calloc(batch_size * n, sizeof *dldC);
	for (i = 0; i < batch_size; ++i)
		for (j = 0; j < n; ++j)
		{
			v = x_hat[j*batch_size + i]; /* = x_hat'[i][j] */
			dldC[i*n + j] = (v - embedding[i*n + j]) * (1 - v*v);
		}

	/* dldW = latent.dldC;
	 * latent is a (m,batch_size) matrix, dldC is a (batch_size,n) so dldW
	 * is a (m,n) matrix (like W). */
	dldW = calloc(m * n, sizeof *dldW);

	/* compute dldW = latent.dldC */
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            m, n, batch_size,
	            1, W, batch_size, latent, n,
	            0, dldW, n);

	free(latent);
	free(x_hat);
	free(dldC);
	free(dldW);
}

/* transform the real-value word vectors of `embedding` into binary vectors */
unsigned long *binarize(float *embedding, long n_vecs, int n_dims, int n_bits)
{
	float *latent, *W, *C, lr_reg;
	unsigned long *binary_vector, bits_group;
	int i, j, n_long, batch_size;

	/* W is a (n_bits, n_dims) matrix, C is a (n_dims) vector */
	srand(0);
	W = random_array(n_dims * n_bits);
	C = random_array(n_dims);

	lr_reg = 0.001;
	batch_size = 75;
	for (i = 0; i < 5; ++i) /* for each iteration */
	{
		for (j = 0; j + batch_size - 1 < n_vecs; j += batch_size)
		{
			apply_regularizarion_gradient(W, n_bits, n_dims, lr_reg);
			apply_reconstruction_gradient(W, C, embedding+j,
			                              n_bits, n_dims, batch_size);
		}

		if (j != n_vecs) /* process remaining vectors not in batch */
		{
			apply_regularizarion_gradient(W, n_bits, n_dims, lr_reg);
			apply_reconstruction_gradient(W, C, embedding+j,
			                              n_bits, n_dims, n_vecs-j);
		}
	}

	/* compute the binary vectors with the original embedding and W. Each
	 * binary vector is represented as a sequence of `long` so if the binary
	 * vectors have 256 bits and a `long` has a length of 64 bits, then each
	 * binary vector is an array of 4 `long` (4 * 64 = 256). The bit
	 * representation of each long are the bits of the vectors. */
	n_long = n_bits / (sizeof(long) * 8);
	latent = calloc(n_vecs * n_bits, sizeof *latent);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
	            n_vecs, n_bits, n_dims,
	            1, embedding, n_dims, W, n_dims,
	            0, latent, n_bits);

	binary_vector = calloc(n_vecs * n_long, sizeof *binary_vector);
	for (i = 0; i < n_vecs; ++i)         /* for each word */
	{
		bits_group = 0;
		for (j = 0; j < n_bits; ++j) /* for each bit */
		{
			/* the j-th bit of the i-th word is determined by the
			 * sign of the j-th value of the latent representation
			 * of the i-th embedding vector. This latent
			 * representation is the dot product between the
			 * original embedding and W. It has already been
			 * computed and stored in latent[i][j]. */

			/* bits are grouped by pack of (sizeof(long)). Add
			 * current bit to current group */
			bits_group <<= 1;
			bits_group |= (latent[i*n_bits + j] > 0);

			/* bits_group has enough bits to form a long, write it
			 * to the binary vector matrix and reset it */
			if ((j+1) % (sizeof(long) * 8) == 0)
			{
				binary_vector[i*n_long + j/(sizeof(long) * 8)] =
					bits_group;
				bits_group = 0;
			}
		}
	}

	free(W);
	free(C);
	return binary_vector;
}

/* write the binary vectors into `filename` */
void write_binary_vectors(char *filename, char **words,
		          unsigned long *binary_vector, long n_vecs, int n_bits)
{
	FILE *fo;
	long i;
	int j, n_long;

	if ((fo = fopen(filename, "w")) == NULL)
	{
		fprintf(stderr, "write_binary_vectors: can't open %s\n",
		        filename);
		exit(1);
	}

	/* first line is the number of vectors and number of bits per vectors */
	fprintf(fo, "%ld %d\n", n_vecs, n_bits);

	for (i = 0, n_long = n_bits / (sizeof(long) * 8); i < n_vecs; ++i)
	{
		fprintf(fo, "%s", words[i]);
		for (j = 0; j < n_long; ++j)
			fprintf(fo, " %lu", binary_vector[i*n_long + j]);
		fprintf(fo, "\n");
	}

	fclose(fo);
}

int main(int argc, char *argv[])
{
	char **words;
	float *embedding;
	unsigned long *binary_vector;
	long n_vecs;
	int n_dims;

	embedding = load_embedding(argv[argc-1], &words, &n_vecs, &n_dims);
	binary_vector = binarize(embedding, n_vecs, n_dims, 256);
	write_binary_vectors("output", words, binary_vector, n_vecs, 256);

	destroy_word_list(words, n_vecs);
	free(embedding); /* `embedding` is created with a single calloc */
	free(binary_vector);
	return 0;
}
