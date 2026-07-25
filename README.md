# My NLP projects

## Tokenization: BPE Algorithm

1. **Encode text into bytes** (typically via UTF-8).
2. **Initialize the vocabulary** with the 256 possible byte values (ids `0`–`255`).
3. **Count pair frequencies**: scan the byte sequence and tally how often each adjacent pair of tokens occurs.
4. **Merge the most frequent pair**: assign it a new id (`256 + idx`), replace all its occurrences in the sequence, and add it to the vocabulary.
5. **Repeat steps 3–4** until reaching the desired vocabulary size.
