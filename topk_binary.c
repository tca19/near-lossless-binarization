#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
static int n_bits = 0, n_long = 0;      /* #bits per vector, #long per array */
static long n_words = 0;
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
 *               vector as an array of `long`, so to represent a vector of 256
 *               bits it requires an array of 4 `long`. */
unsigned long **load_vectors(const char *name)
{
	int i;
	long index;
	FILE *fp;                 /* to open vector file */
	char word[MAXLENWORD];    /* to read the word of each line in file */
	unsigned long **vec;      /* to store the binary embeddings */

	if ((fp = fopen(name, "r")) == NULL)
	{
		fprintf(stderr, "load_vectors: can't open %s\n", name);
		exit(1);
	}

	if (fscanf(fp, "%ld %d", &n_words, &n_bits) <= 0)
	{
		fprintf(stderr, "load_vectors: can't read number of bits\n");
		exit(1);
	}

	n_long = n_bits / (sizeof(long) * 8);
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

		if ((vec[index] = calloc(n_long, sizeof **vec)) == NULL)
			continue;

		for (i = 0; i < n_long; ++i)
			fscanf(fp, "%lu", vec[index]+i);
	}

	fclose(fp);
	return vec;
}

/* binary_sim: return the Sokal-Michener binary similarity (#common / size). */
float binary_sim(const unsigned long *v1, const unsigned long *v2)
{
	int n, i;

	/* need the ~ because *v1 ^ *v2 sets the bit to 0 if same bit */
	for (n = 0, i = 0; i++ < n_long; v1++, v2++)
		n += __builtin_popcountl(~*v1 ^ *v2);
	return n / (float) n_bits;
}

/* find_topk: return the k nearest neighbors of word */
struct neighbor *find_topk(const char *word, const int k, unsigned long **vec)
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
		topk[k].similarity = binary_sim(vec[index], vec[i]);
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
	unsigned long **embedding;
	struct neighbor *topk;
	int i, k;
	clock_t start, end;

	if (argc < 4)
	{
		printf("usage: ./topk_binary EMBEDDING K QUERY...\n");
		exit(1);
	}

	embedding = load_vectors(*++argv);
	k = atoi(*++argv);
	argc -= 2; /* because already used argument 0 and 1 */

	while (--argc > 0)
	{
		start = clock();
		topk = find_topk(*++argv, k, embedding);
		end = clock();

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
