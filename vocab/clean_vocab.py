from time import sleep

with open("vocab.txt", "r") as f:
	vocab = [w.strip("\n") for w in f.readlines()]

threshold = lambda x: ">" in x or "<" in x
bang = [x for x in vocab if threshold(x)]
new_vocab = vocab

print("extracted vocab.")
sleep(.2)

print(f"found {len(bang)} bangs")
print("cleaning now...")
sleep(.2)

print("please mark all bangs.")
sleep(.2)

for b in bang:
	mark = True if input(b+" ").lower() == "x" else False

	if mark:
		new_vocab.remove(b)

with open("vocab_clean.txt", "w") as f:
	for w in new_vocab:
		f.write(w + "\n")