#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATADIR    "similarity/"
#define HASHSIZE   10000
#define MAXLENPATH 64
#define MAXLENWORD 64

struct nlist
{
	struct nlist *next; /* next element in linked list */
	char *word;
	long index;
};

static struct nlist *hashtab[HASHSIZE]; /* hashtab composed of linked lists */
static long word_index = 0;             /* index of vector associated to word */
static int n_bits = 0, n_long = 0;      /* #bits per vector, #long per array */

/* hash: form hash value for string s */
unsigned int hash(const char *s)
{
	unsigned int hashval;

	for (hashval = 0; *s != '\0'; ++s)
		hashval = *s + 31 * hashval;
	return hashval % HASHSIZE;
}

/* get_index: return vector index of word s */
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
	np->next = hashtab[hashval];
	np->index = word_index++;
	hashtab[hashval] = np;
}

/* create_vocab: read each file in dirname to create vocab of unique words */
void create_vocab(char *dirname)
{
	DIR *dp;
	FILE *fp;
	struct dirent *ent;
	char filepath[MAXLENPATH], word1[MAXLENWORD], word2[MAXLENWORD];
	float simval;

	if ((dp = opendir(dirname)) == NULL)
	{
		fprintf(stderr, "create_vocab: can't open %s\n", dirname);
		exit(1);
	}

	while ((ent = readdir(dp)) != NULL)
	{
		if (strcmp(ent->d_name, ".") == 0
		 || strcmp(ent->d_name, "..") == 0)
			continue;

		strcpy(filepath, DATADIR);
		strcat(filepath, ent->d_name);
		if ((fp = fopen(filepath, "r")) == NULL)
		{
			fprintf(stderr, "create_vocab: can't open file %s\n",
			        filepath);
			continue;
		}

		while (fscanf(fp, "%s %s %f", word1, word2, &simval) > 0)
		{
			add_word(word1);
			add_word(word2);
		}
	}
}

/* read_binary_embeddings: read the file name to get binary vectors */
void read_binary_embeddings(char *name)
{
	int dim;
	int n_int, i;
	FILE *fp;
	unsigned long l;
	char word[MAXLENWORD];

	if ((fp = fopen(name, "r")) == NULL)
	{
		fprintf(stderr, "read_binary_embeddings: can't open %s\n",
		        name);
		exit(1);
	}

	if (fscanf(fp, "%d", &dim) <= 0)
	{
		fprintf(stderr, "read_binary_embeddings: can't read dimension"
		        " of binary vectors.\n");
		exit(1);
	}

	n_int = dim / (sizeof(long) * 8);
	printf("%d\n", n_int);
	while (fscanf(fp, "%s", word) > 0)
	{
		printf("\n%s", word);
		for (i = 0; i < n_int; ++i)
		{
			fscanf(fp, "%lu", &l);
			printf(" %lu", l);
		}
	}
}

int main(void)
{
	vocab_create(DATADIR);
	read_binary_embeddings("out.txt");
	return 0;
}
