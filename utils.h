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

/* hashtab.c */
extern long word_index;         /* index of vector associated to word */
unsigned int hash(const char*);
long get_index(const char*);
void add_word(const char*);
void lower(char*);

/* spearman.c */
float spearman_coef(float*, float*, int);

/* file_process.c */
void create_vocab(const char*);
unsigned long **load_binary_vectors(const char*, long*, int*, int*);
float         **load_real_vectors(const char*, long *, int*);
void evaluate(const char*, void**, int,
              float (*f)(const void*, const void*, const int));
