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

static struct nlist *hashtab[HASHSIZE]; /* pointer table */

/* hash: form hash value for string s */
unsigned int hash(const char *s)
{
	unsigned int hashval;

	for (hashval = 0; *s != '\0'; ++s)
		hashval = *s + 31 * hashval;
	return hashval % HASHSIZE;
}

/* get_index: return index of string s; add it to hashtab if not present */
long get_index(const char *s)
{
	struct nlist *np;
	static long index = 0;
	unsigned int hashval = hash(s);

	/* look for string s in hashtab */
	for (np = hashtab[hashval]; np != NULL; np = np->next)
		if (strcmp(np->word, s) == 0) /* found, return its index */
			return np->index;

	/* word not found, need to add it */
	if ((np = malloc(sizeof(*np))) == NULL)
		return -1;
	if ((np->word = malloc(strlen(s) + 1)) == NULL)
		return -1;

	strcpy(np->word, s);
	np->next = hashtab[hashval];
	np->index = index++;
	hashtab[hashval] = np;

	return np->index;
}

/* vocab_create: read each file in dirname, print each word in these files */
void vocab_create(char *dirname)
{
	DIR *dp;
	FILE *fp;
	struct dirent *ent;
	char filepath[MAXLENPATH], word1[MAXLENWORD], word2[MAXLENWORD];
	float simval;
	int nlines;

	if ((dp = opendir(dirname)) == NULL)
	{
		fprintf(stderr, "vocab_create: can't open %s\n", dirname);
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
			fprintf(stderr, "vocab_create: can't open file %s\n",
			        filepath);
			continue;
		}

		nlines = 0;
		while (fscanf(fp, "%s %s %f", word1, word2, &simval) > 0)
		{
			++nlines;
			get_index(word1);
			get_index(word2);
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
