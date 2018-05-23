/* hashtab.c */
extern long word_index;         /* index of vector associated to word */
unsigned int hash(const char*);
long get_index(const char*);
void add_word(const char*);
void lower(char*);

/* spearman.c */
float spearman_coef(float*, float*, int);
