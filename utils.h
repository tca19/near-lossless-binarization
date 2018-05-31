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
void evaluate(const char*, void**, int,
              float (*f)(const void*, const void*, const int));
