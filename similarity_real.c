#include <ctype.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spearman.c"

#define DATADIR    "datasets/"
#define HASHSIZE   10000
#define MAXLINES   3500
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
static int n_vecs = 0, n_dims = 0;      /* #vectors, #dimension */

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
	np->next = hashtab[hashval];
	np->index = word_index++;
	hashtab[hashval] = np;
}

/* lower: lowercase all char of s */
void lower(char *s)
{
	for (; *s; ++s)
		*s = tolower(*s);
}

/* create_vocab: read each file in dirname to create vocab of unique words */
void create_vocab(char *dirname)
{
	DIR *dp;
	FILE *fp;
	struct dirent *ent;
	char filepath[MAXLENPATH], word1[MAXLENWORD], word2[MAXLENWORD];
	float val;

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

		while (fscanf(fp, "%s %s %f", word1, word2, &val) > 0)
		{
			lower(word1);
			lower(word2);
			add_word(word1);
			add_word(word2);
		}
		fclose(fp);
	}
	closedir(dp);
}

/* load_vectors: read the vector file, only load the vectors of words present in
 *               hashtab (so at most word_index+1 vectors). */
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

	if (fscanf(fp, "%d %d", &n_vecs, &n_dims) <= 0)
	{
		fprintf(stderr, "load_vectors: can't read dimension\n");
		exit(1);
	}

	if ((vec = calloc(word_index + 1, sizeof *vec)) == NULL)
		return NULL;

	while (fscanf(fp, "%s", word) > 0)
	{
		index = get_index(word);
		if (index == -1)    /* drop the words not in vocab */
			continue;
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

/* evaluate: compute Spearman coefficient for each file in dirname */
void evaluate(char *dirname, float **vec)
{
	DIR *dp;
	FILE *fp;
	struct dirent *ent;
	char filepath[MAXLENPATH], word1[MAXLENWORD], word2[MAXLENWORD];
	float val;
	long index1, index2, found, nlines;
	float *simfile, *simvec;

	if ((simfile = malloc(MAXLINES * sizeof *simfile)) == NULL
	 || (simvec  = malloc(MAXLINES * sizeof *simvec))  == NULL)
	{
		fprintf(stderr, "evaluate: can't allocate memory to store"
		        " similarity values in datasets.\n");
		exit(1);
	}

	if ((dp = opendir(dirname)) == NULL)
	{
		fprintf(stderr, "evaluate: can't open %s\n", dirname);
		exit(1);
	}

	printf("%-12s | %-8s | %3s\n", "Filename", "Spearman", "OOV");
	printf("==============================\n");
	while ((ent = readdir(dp)) != NULL)
	{
		if (strcmp(ent->d_name, ".") == 0
		 || strcmp(ent->d_name, "..") == 0)
			continue;

		strcpy(filepath, DATADIR);
		strcat(filepath, ent->d_name);
		if ((fp = fopen(filepath, "r")) == NULL)
		{
			fprintf(stderr, "evaluate: can't open file %s\n",
			        filepath);
			continue;
		}

		found = nlines = 0;
		while (fscanf(fp, "%s %s %f", word1, word2, &val) > 0
		    && nlines < MAXLINES)
		{
			++nlines;
			lower(word1);
			lower(word2);
			index1 = get_index(word1);
			index2 = get_index(word2);

			if (vec[index1] == NULL || vec[index2] == NULL)
				continue;

			simfile[found] = val;
			simvec[found] = cosine_sim(vec[index1], vec[index2]);
			++found;
		}

		val = spearman_coef(simfile, simvec, found);
		printf("%-12s | %8.3f | %3ld%%\n", ent->d_name, val,
		       (nlines - found) * 100 /  nlines);
		fclose(fp);
	}
	closedir(dp);
}

int main(int argc, char *argv[])
{
	float **embedding;

	if (argc != 2)
	{
		printf("usage: ./similarity_real EMBEDDING\n");
		exit(1);
	}

	create_vocab(DATADIR);
	embedding = load_vectors(*++argv);
	evaluate(DATADIR, embedding);

	return 0;
}
