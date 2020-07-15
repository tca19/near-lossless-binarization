/* Copyright (c) 2019-present, All rights reserved.
 * Written by Julien Tissier <30314448+tca19@users.noreply.github.com>
 *
 * This file is part of the "Near-lossless Binarization of Word Embeddings"
 * software (https://github.com/tca19/near-lossless-binarization).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License at the root of this repository for
 * more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

#define HASHSIZE   1000000

struct nlist
{
	struct nlist *next; /* next element in linked list */
	char *word;
	long index;
};

/* Hash table composed of linked lists. This variable is made static because it
 * should only be accessed from functions defined in this file. Only the
 * functions get_index() and add_word() have access to it (either to find the
 * index of a word or to add a new word into the hash table). */
static struct nlist *hashtab[HASHSIZE];

/* Counter to know the current number of words in the hash table. Also used as
 * the index for new words (e.g. if there are 17 words in the hash table, the
 * new inserted word will have the index 17). Not static because other files
 * (like file_process.c) might need to know this value. */
long n_words = 0;

/* Array (used like a Python dictionary) to know a word given its index. Also
 * not declared as static because used by other files (like topk_binary.c). */
char **words = NULL;

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

/* add_word: add word s to hashtab (only if not present). If the flag
 *           `save_word_index` is on (i.e. not zero), also add the word s into
 *           the array `words`, which is used to find a word given its index. */
void add_word(const char *s, const int save_word_index)
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

	/* only add the word s to the array index->word if the argument flag
	 * has been set. Need a flag because the array words is not required in
	 * the similarity_binary program (only in topk_binary to know the
	 * word associated to a neighbor). */
	if (save_word_index != 0)
	{
		if (words == NULL)
		{
			fprintf(stderr, "add_word: no memory for `words`\n");
			exit(1);
		}
		words[n_words] = np->word;
	}

	np->next = hashtab[hashval];
	np->index = n_words++;
	hashtab[hashval] = np;
}

/* lower: lowercase all char of s */
void lower(char *s)
{
	for (; *s; ++s)
		*s = tolower(*s);
}
