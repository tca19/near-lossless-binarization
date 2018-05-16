#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define HASHSIZE   1000000
#define MAXLENWORD 64

struct nlist
{
	struct nlist *next; /* next element in linked list */
	char *word;
	long index;
};

struct neighbor
{
	long index;
	float similarity;
};

static struct nlist *hashtab[HASHSIZE]; /* hashtab composed of linked lists */
static long word_index = 0;             /* index of vector associated to word */
static int n_words = 0, n_dims = 0;     /* #vectors, #dimension */
static char **words;

/* hash: form hash value for string s */
unsigned int hash(const char *s)
{
	unsigned int hashval;

	for (hashval = 0; *s != '\0'; ++s)
		hashval = *s + 31 * hashval;
	return hashval % HASHSIZE;
}

/* get_index: return vector index of word s, -1 if not found */
long get_index(const char *s)
{
	struct nlist *np;

	/* look for string s in hashtab */
	for (np = hashtab[hash(s)]; np != NULL; np = np->next)
		if (strcmp(np->word, s) == 0) /* found, return its index */
			return np->index;
	return -1;
}

/* add_word: add word s to hashtab (only if not present) */
void add_word(const char *s)
{
	struct nlist *np;
	unsigned int hashval = hash(s);

	if (get_index(s) > -1) /* already in hashtab */
		return;

	/* word not in hashtab, need to add it */
	if ((np = malloc(sizeof *np)) == NULL ||
	    (np->word = malloc(strlen(s) + 1)) == NULL)
		return;

	strcpy(np->word, s);
	words[word_index] = np->word;
	np->next = hashtab[hashval];
	np->index = word_index++;
	hashtab[hashval] = np;
}

/* load_vectors: read the vector file, add each word to hashtab, save each
 *               vector as an array of float */
float **load_vectors(char *name)
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

	if (fscanf(fp, "%d %d", &n_words, &n_dims) <= 0)
	{
		fprintf(stderr, "load_vectors: can't read dimension\n");
		exit(1);
	}

	if ((vec = calloc(n_words, sizeof *vec)) == NULL)
		return NULL;

	if ((words = calloc(n_words, sizeof *words)) == NULL)
		fprintf(stderr, "load_vectors: no memory for index<=>word\n");

	while (fscanf(fp, "%s", word) > 0)
	{
		/* add word to hashtab; its index is word_index because never
		 * seen before (so not already in hashtab)  */
		index = word_index;
		add_word(word);

		if ((vec[index] = calloc(n_dims, sizeof **vec)) == NULL)
			continue;

		for (i = 0; i < n_dims; ++i)
			fscanf(fp, "%f", vec[index]+i);
	}

	fclose(fp);
	return vec;
}

/* cosine_sim: return cosine similarity between vector v1 and v2 */
float cosine_sim(float *v1, float *v2)
{
	int i;
	float dot, norm1, norm2;

	dot = norm1 = norm2 = 0;
	for (i = 0; i < n_dims; ++i)
	{
		dot   += v1[i] * v2[i];
		norm1 += v1[i] * v1[i];
		norm2 += v2[i] * v2[i];
	}

	return dot / sqrt(norm1 * norm2);
}

/* find_topk: return the k nearest neighbors of word */
struct neighbor *find_topk(char *word, int k, float **vec)
{
	long i, j, index;
	struct neighbor *topk, tmp;

	if ((topk = calloc(k + 1, sizeof *topk)) == NULL)
	{
		fprintf(stderr, "find_topk: can't allocate memory for heap\n");
		exit(1);
	}

	index = get_index(word);
	for (i = 0; i < n_words; ++i)
	{
		if (i == index)  /* skip word (always its nearest neighbor) */
			continue;

		/* values in topk are sorted by decreasing similarity. If the
		 * similarity with current vector is greater than minimal
		 * similarity in topk, insert current similarity into topk */
		topk[k].similarity = cosine_sim(vec[index], vec[i]);
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
	float **embedding;
	struct neighbor *topk;
	int i, k;

	if (argc < 3)
	{
		printf("usage: ./topk_real K WORDS...\n");
		exit(1);
	}

	k = atoi(*++argv);
	embedding = load_vectors("gl-300d.vec");

	while (--argc > 1)
	{
		topk = find_topk(*++argv, k, embedding);

		printf("Top %d closest words of %s\n", k, *argv);
		for (i = 0; i < k; ++i)
			printf("  %-15s %.3f\n", words[topk[i].index],
			                         topk[i].similarity);
		printf("\n");
	}

	return 0;
}
