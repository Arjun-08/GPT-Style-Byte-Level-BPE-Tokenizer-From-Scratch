def text_to_bytes(text):
    #Convert Unicode text into UTF-8 bytes
    print("\n[TEXT -> BYTES]")
    print("Original text:", repr(text))
    byte_data = text.encode("utf-8")
    print("UTF-8 bytes:", list(byte_data))
    return list(byte_data)


def bytes_to_text(byte_list):
    #Convert a list of byte values back into UTF-8 text
    byte_data = bytes(byte_list)
    text = byte_data.decode("utf-8")
    return text

def text_to_token_sequence(text):
    #Initial BPE representation. Every byte starts as its own token.
    return text_to_bytes(text)

def count_pairs(tokens):
    #Count every adjacent pair.
    pair_counts = {}

    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        if pair not in pair_counts:
            pair_counts[pair] = 0
        pair_counts[pair] += 1
    return pair_counts

def get_most_frequent_pair(pair_counts):
    #Find the pair occurring most frequently.
    if not pair_counts:
        return None, 0

    best_pair = None
    best_count = 0

    for pair, count in pair_counts.items():
        if count > best_count:
            best_pair = pair
            best_count = count
    return best_pair, best_count

def merge_pair(tokens, pair, new_token):
    #Replace every occurrence of pair = (A, B) with new_token
    merged = []
    i = 0

    while i < len(tokens):
        # Check whether the current and next token form the pair we want to merge.
        if (
            i < len(tokens) - 1
            and tokens[i] == pair[0]
            and tokens[i + 1] == pair[1]):
            merged.append(new_token)
            # Skip BOTH original tokens.
            i += 2

        else:
            merged.append(tokens[i])
            i += 1
    return merged

def train_bpe(text, target_vocab_size=300, verbose=True):
    print("\n")
    print("=" * 70)
    print("STARTING BPE TRAINING")
    print("=" * 70)

    # Initial vocabulary: 256 single-byte tokens
    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])

    print("\nInitial vocabulary size:", len(vocab))
    print("Initial tokens: byte values 0 through 255")

    # Convert training text into bytes
    tokens = text_to_token_sequence(text)

    print("\nInitial token sequence:")
    print(tokens)
    merges = {}

    next_token_id = 256
    number_of_merges = target_vocab_size - 256

    print("\nTarget vocabulary size:", target_vocab_size)
    print("Number of merges:", number_of_merges)

    for iteration in range(number_of_merges):
        print("\n")
        print("-" * 70)
        print("BPE ITERATION:", iteration + 1)
        print("-" * 70)

        # Count pairs
        pair_counts = count_pairs(tokens)

        print("\n COUNT ADJACENT PAIRS")

        # Show top pairs
        sorted_pairs = sorted(
            pair_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print("Number of unique pairs:", len(sorted_pairs))
        print("\nMost frequent pairs:")

        for pair, count in sorted_pairs[:10]:
            print("   ",pair, "->", count, "times")
        # Find most frequent pair
        
        best_pair, best_count = get_most_frequent_pair(pair_counts)
        if best_pair is None:
            print("\nNo more pairs available.")
            break

        print("Selected pair:", best_pair)
        print("Frequency:", best_count)

        # Create new token
        new_token = next_token_id

        print("\n CREATE NEW TOKEN")

        print("Pair", best_pair, "will become token", new_token)

        # Store merge rule
        merges[best_pair] = new_token

        #  Add token to vocabulary
        vocab[new_token] = ( vocab[best_pair[0]] + vocab[best_pair[1]] )

        print("\n ADD TOKEN TO VOCABULARY")

        print("Token ID:", new_token)
        print( "Raw bytes:",list(vocab[new_token]))

        # Try to display it as text.
        try:
            print("Decoded representation:", repr(vocab[new_token].decode("utf-8")))

        except UnicodeDecodeError:
            print("Decoded representation: <partial/invalid UTF-8>")

        # Merge pair in training sequence
        old_length = len(tokens)
        tokens = merge_pair(tokens,best_pair,new_token)
        new_length = len(tokens)

        print("\n MERGE TOKEN SEQUENCE")

        print("Sequence length:", old_length, "->", new_length)
        print("Current sequence:",tokens)

        # Move to next token ID
        next_token_id += 1

    print("\n")
    print("=" * 70)
    print("BPE TRAINING COMPLETE")
    print("=" * 70)

    print("\nFinal vocabulary size:", len(vocab))
    print("Number of learned merges:", len(merges))

    return vocab, merges

#  DISPLAY LEARNED VOCABULARY
def display_vocabulary(vocab, start=256, limit=50):

    print("\n")
    print("=" * 70)
    print("LEARNED VOCABULARY")
    print("=" * 70)

    end = min(start + limit, len(vocab))

    for token_id in range(start, end):

        if token_id not in vocab:
            continue

        token_bytes = vocab[token_id]

        print( f"{token_id:4d}", "->", list(token_bytes), "->",
            repr(token_bytes.decode("utf-8", errors="replace")))

def apply_merge(tokens, pair, new_token):
    #Apply one learned merge rule.

    return merge_pair(tokens,pair,new_token)

#  ENCODE TEXT
def encode(text, merges, verbose=True):

    print("\n")
    print("=" * 70)
    print("ENCODING")
    print("=" * 70)

    print("\nInput text:")
    print(repr(text))

    # Convert text to bytes
    tokens = text_to_token_sequence(text)

    print("\nInitial byte tokens:")
    print(tokens)

    for step, (pair, new_token) in enumerate(merges.items(), start=1):
        
        old_tokens = tokens
        tokens = apply_merge(tokens, pair, new_token)

        if tokens != old_tokens:
            print(f"\nMerge {step}:", pair,"->",new_token)

            print("Sequence:",tokens)

    print("\nFINAL TOKEN IDs:")
    print(tokens)

    return tokens

# DECODE
def decode(token_ids, vocab, verbose=True):

    print("\n")
    print("=" * 70)
    print("DECODING")
    print("=" * 70)

    print("\nToken IDs:")
    print(token_ids)

    byte_data = b""

    for token_id in token_ids:
        token_bytes = vocab[token_id]
        print(f"Token {token_id}"
            f" -> bytes {list(token_bytes)}")

        byte_data += token_bytes
    text = byte_data.decode("utf-8")

    print("\nCombined bytes:")
    print(list(byte_data))

    print("\nDecoded text:")
    print(repr(text))

    return text

#  SAVE TOKENIZER
def save_tokenizer(vocab, merges, filename="bpe_tokenizer.txt"):

    with open(filename, "w", encoding="utf-8") as f:
        f.write("VOCABULARY\n")
        f.write("=" * 50 + "\n")

        for token_id, token_bytes in vocab.items():
            f.write(f"{token_id}\t{list(token_bytes)}\n")

        f.write("\nMERGES\n")
        f.write("=" * 50 + "\n")

        for pair, token_id in merges.items():
            f.write( f"{pair[0]},{pair[1]}\t{token_id}\n")

    print("\nTokenizer saved to:", filename)

def load_tokenizer(filename="bpe_tokenizer.txt"):
    vocab = {}
    merges = {}
    mode = None

    with open(filename, "r", encoding="utf-8") as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line == "VOCABULARY":
                mode = "vocab"
                continue

            if line == "MERGES":
                mode = "merges"
                continue

            if line.startswith("="):
                continue

            if mode == "vocab":

                token_id, byte_string = line.split("\t")
                token_id = int(token_id)

                # Safely convert "[104, 101]" back to bytes.
                numbers = byte_string.strip("[]").split(",")

                if numbers == [""]:
                    numbers = []

                byte_values = [
                    int(x.strip())
                    for x in numbers
                ]

                vocab[token_id] = bytes(byte_values)

            elif mode == "merges":
                pair_string, token_id = line.split("\t")
                a, b = pair_string.split(",")
                pair = (int(a),int(b))

                merges[pair] = int(token_id)

    print("\nTokenizer loaded.")

    return vocab, merges


# SMALL DEMO
if __name__ == "__main__":

    training_text = (
        "hello hello hello "
        "hello world "
        "hello world "
        "hello hello "
        "world world "
    )

    vocab, merges = train_bpe( training_text, target_vocab_size=270)

    # Show learned vocabulary
    display_vocabulary(vocab, start=256, limit=20)

    # Show learned merge rules
    print("\n")
    print("=" * 70)
    print("LEARNED MERGE RULES")
    print("=" * 70)

    for pair, token_id in merges.items():
        print(pair,"->", token_id)

    # Encode
    text = "hello world"

    token_ids = encode(text,merges)

    # Decode
    reconstructed = decode(token_ids,vocab)

    print("\n")
    print("=" * 70)
    print("ROUND-TRIP TEST")
    print("=" * 70)

    print("Original:     ", repr(text))
    print("Reconstructed:", repr(reconstructed))

    if text == reconstructed:
        print("\nSUCCESS: encode -> decode is lossless!")

    else:
        print("\nERROR: reconstruction failed.")

    save_tokenizer(vocab,merges)