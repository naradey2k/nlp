from models.basic_tokenizer import Tokenizer

with open("data.txt", "r") as f:
    text = "".join(f.readlines())

tokenizer = Tokenizer()
tokenizer.train(text, vocab_size=100)

new_text = "Cristiano Ronaldo, Lebron James."
enc_tokens = tokenizer.encode(new_text)
dec_tokens = [tokenizer.decode([token]) for token in enc_tokens]

print(enc_tokens)
print(" ".join(dec_tokens))
