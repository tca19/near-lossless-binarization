gcc -Wall -O3 -march=native -lm similarity_binary.c -o similarity_binary

echo      --- Semantic similarity ---
./similarity_binary glove-128bits.bin


echo
echo      --- Top-K query ---
gcc -Wall -O3 -march=native -lm topk_binary.c -o topk_binary

./topk_binary glove-128bits.bin 10 queen
