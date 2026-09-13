# GPT-Style Byte-Level BPE Tokenizer From Scratch

A minimal implementation of a GPT-style **Byte Pair Encoding (BPE) tokenizer**, implemented entirely from scratch using Python's standard library. The purpose of this project is to understand the core algorithm behind modern subword tokenization by implementing the complete BPE workflow without relying on tokenizer libraries such as `tiktoken`, Hugging Face Tokenizers, or SentencePiece.

The implementation demonstrates how raw text is converted into UTF-8 bytes, how frequent adjacent token pairs are identified, how new tokens are learned through iterative merging, and how the resulting vocabulary and merge rules are used for encoding and decoding.

---

# Project Overview

A language model does not directly process raw text. Instead, text is converted into a sequence of integer token IDs:

```text
Raw Text
   |
   v
Tokenizer
   |
   v
Token IDs
   |
   v
Language Model
```

For example:

```text
"hello world"
```

can be represented by this implementation as:

```text
[260, 265]
```

The tokenizer is responsible for learning what these token IDs represent and how text should be converted into them.

The core idea behind BPE is:

> Find the most frequent adjacent pair of tokens and replace that pair with a new token.

Repeating this operation allows the tokenizer to learn increasingly larger and more reusable sequences.

---

# Architecture

The tokenizer consists of two primary pipelines:

- A **training pipeline**, which learns the vocabulary and merge rules.
- An **inference pipeline**, which uses the learned rules to encode and decode text.

```text
                         GPT-STYLE BPE TOKENIZER
                                  |
                  +---------------+---------------+
                  |                               |
                  v                               v
             TRAINING                         INFERENCE
                  |                               |
                  v                               v
             Raw Corpus                       New Text
                  |                               |
                  v                               v
           UTF-8 Encoding                  UTF-8 Encoding
                  |                               |
                  v                               v
          Byte Tokenization                Initial Tokens
                  |                               |
                  v                               v
          Pair Frequency                  Apply Learned
             Counting                     Merge Rules
                  |                               |
                  v                               v
        Most Frequent Pair                  Token IDs
                  |
                  v
          Create New Token
                  |
                  v
           Store Merge Rule
                  |
                  v
                 Merge
                  |
                  v
          Updated Token Sequence
                  |
                  +------------------+
                                     |
                                     v
                             Repeat Until Target
                             Vocabulary Is Reached
                                     |
                                     v
                         Learned Vocabulary +
                          Learned Merge Rules
                                     |
                                     v
                                  Decoder
                                     |
                                     v
                               Token Bytes
                                     |
                                     v
                               UTF-8 Decode
                                     |
                                     v
                               Original Text
```

---

# Training Pipeline

The training pipeline learns the tokenizer from a training corpus.

```text
Raw Training Text
       |
       v
UTF-8 Encoding
       |
       v
Initial Byte Tokens
       |
       v
Count Adjacent Pairs
       |
       v
Find Most Frequent Pair
       |
       v
Create New Token ID
       |
       v
Store Merge Rule
       |
       v
Merge Pair
       |
       v
Updated Token Sequence
       |
       +--------------------+
                            |
                            v
                     Repeat BPE Steps
                            |
                            v
                  Learned Vocabulary
                            +
                    Learned Merge Rules
```

Each iteration adds one new token to the vocabulary.

---

# Encoding Pipeline

Once training is complete, new text can be encoded using the learned merge rules.

```text
New Text
   |
   v
UTF-8 Encoding
   |
   v
Byte Sequence
   |
   v
Initial Token IDs
   |
   v
Apply Learned Merge 1
   |
   v
Apply Learned Merge 2
   |
   v
Apply Learned Merge 3
   |
   v
        ...
   |
   v
Final Token IDs
```

The encoder does not learn new rules during inference.

It only applies the merge rules learned during training.

---

# Decoding Pipeline

Decoding converts token IDs back into their underlying bytes and then reconstructs the original UTF-8 text.

```text
Token IDs
   |
   v
Vocabulary Lookup
   |
   v
Underlying Byte Sequences
   |
   v
Concatenate Bytes
   |
   v
UTF-8 Decode
   |
   v
Original Text
```

The underlying byte sequence of every learned token is stored in the vocabulary.

---

# Core Components

The implementation can be viewed as five fundamental components:

```text
+---------------------------+
| Byte Vocabulary           |
| 0–255                     |
+-------------+-------------+
              |
              v
+---------------------------+
| Pair Counter              |
| Counts adjacent pairs     |
+-------------+-------------+
              |
              v
+---------------------------+
| Merge Learner             |
| Selects most frequent     |
| pair                      |
+-------------+-------------+
              |
              v
+---------------------------+
| Merge Table               |
| pair -> token ID          |
+-------------+-------------+
              |
              v
+---------------------------+
| Encoder / Decoder         |
| Text <-> Token IDs        |
+---------------------------+
```

---

# What Is BPE?

**Byte Pair Encoding (BPE)** is a compression-based subword tokenization algorithm.

The basic operation is:

```text
Find the most frequent adjacent pair
              |
              v
       Create new token
              |
              v
       Replace the pair
              |
              v
            Repeat
```

For example, suppose the training sequence is:

```text
a b a b a b a b
```

If:

```text
(a, b)
```

is the most frequent pair, BPE creates:

```text
(a, b) -> X
```

and the sequence becomes:

```text
X X X X
```

The tokenizer can then find another frequent pair:

```text
(X, X) -> Y
```

resulting in:

```text
Y Y
```

BPE therefore builds larger units progressively from smaller units.

---

# Why Byte-Level BPE?

Instead of starting with words or characters, this implementation starts with bytes.

A byte can represent 256 possible values:

```text
0, 1, 2, ..., 255
```

Therefore the initial vocabulary contains exactly:

```text
256 tokens
```

This provides a fixed and language-independent starting point.

For example:

```text
'h' -> 104
'e' -> 101
'l' -> 108
'o' -> 111
' ' -> 32
```

Therefore:

```text
hello
```

becomes:

```text
[104, 101, 108, 108, 111]
```

after UTF-8 encoding.

---

# Initial Vocabulary

The tokenizer begins with:

```text
Vocabulary size = 256
```

The initial token IDs are:

```text
0 through 255
```

In this implementation, these IDs directly correspond to byte values.

```text
Token ID    Meaning
--------------------
0           byte 0
1           byte 1
2           byte 2
...
32          byte 32
65          byte 65
97          byte 97
104         byte 104
108         byte 108
111         byte 111
...
255         byte 255
```

The learned BPE vocabulary begins at:

```text
256
```

Therefore:

```text
0–255
    Primitive byte tokens

256+
    Learned BPE tokens
```

For example:

```text
101 + 108
   |
   v
256
```

where:

```text
101 = 'e'
108 = 'l'
```

so:

```text
256 = "el"
```

---

# Bytes Are Not Characters

An important distinction is that:

```text
0–255
```

represent **byte values**, not characters.

For ASCII characters, the relationship is straightforward:

```text
'a' -> 97
'b' -> 98
'h' -> 104
'e' -> 101
'l' -> 108
'o' -> 111
' ' -> 32
```

However, Unicode characters can require multiple UTF-8 bytes.

Therefore byte-level tokenization does not assume that every character corresponds to a single token.

The tokenizer starts from the raw UTF-8 byte representation and learns useful byte sequences from the training data.

---

# Text to Bytes

The first transformation is:

```python
byte_data = text.encode("utf-8")
```

For:

```text
hello
```

the tokenizer obtains:

```text
[104, 101, 108, 108, 111]
```

These byte values become the initial token sequence.

---

# Pair Frequency Counting

Given a token sequence:

```text
[1, 2, 3, 2, 3]
```

the adjacent pairs are:

```text
(1, 2)
(2, 3)
(3, 2)
(2, 3)
```

Their frequencies are:

```text
(1, 2) -> 1
(2, 3) -> 2
(3, 2) -> 1
```

The implementation performs this operation using a Python dictionary.

The mathematical frequency of pair `(a,b)` is:

\[
C(a,b)
=
\sum_{i=1}^{n-1}
\mathbf{1}
[t_i=a \land t_{i+1}=b]
\]

where:

- \(t_i\) is the token at position \(i\)
- \(C(a,b)\) is the frequency of pair `(a,b)`
- \(\mathbf{1}\) is an indicator function

The indicator function is:

\[
\mathbf{1}[condition]
=
\begin{cases}
1, & \text{if condition is true}\\
0, & \text{otherwise}
\end{cases}
\]

---

# Selecting the Most Frequent Pair

After counting all adjacent pairs, BPE selects the pair with the highest frequency.

Mathematically:

\[
(a^*,b^*)
=
\underset{(a,b)}{\arg\max}
\ C(a,b)
\]

In words:

> Select the pair that appears next to each other most frequently.

For example:

```text
(104, 101) -> 7
(101, 108) -> 7
(108, 108) -> 7
(108, 111) -> 7
(111, 32)  -> 7
```

The implementation selects one of the highest-frequency pairs according to its iteration order.

In the experiment, the first selected pair was:

```text
(104, 101)
```

with frequency:

```text
7
```

---

# Creating a New Token

The first 256 token IDs are already occupied by byte values.

Therefore the first BPE-generated token receives:

```text
256
```

The first merge was:

```text
(104, 101) -> 256
```

The underlying bytes are:

```text
[104, 101]
```

which represent:

```text
"he"
```

The new vocabulary entry is therefore:

```text
256 -> "he"
```

---

# Underlying Token Representation

Every learned token is represented by its underlying bytes.

If:

```text
101 -> b'e'
108 -> b'l'
```

then:

```text
256 -> b'e' + b'l'
```

giving:

```text
256 -> b'el'
```

More generally:

\[
B(z)
=
B(x)\Vert B(y)
\]

where:

- \(B(z)\) is the byte representation of the new token
- \(B(x)\) is the byte representation of token \(x\)
- \(B(y)\) is the byte representation of token \(y\)
- \(\Vert\) represents concatenation

This allows every learned token to retain a direct path back to the original bytes.

---

# Merging Tokens

Suppose the sequence is:

```text
104 101 108 108 111
```

and BPE has learned:

```text
(104,101) -> 256
```

The pair is replaced:

```text
104 101 108 108 111
    |
    v
256 108 108 111
```

The sequence therefore changes from:

```text
5 tokens
```

to:

```text
4 tokens
```

This merge makes the representation more compact.

---

# Iterative Vocabulary Learning

The important part is that BPE does not stop after one merge.

After creating token `256`, the tokenizer counts pairs again.

Suppose:

```text
(256,108)
```

is now the most frequent pair.

The tokenizer creates:

```text
(256,108) -> 257
```

Token `257` represents:

```text
"hel"
```

Then:

```text
(257,108) -> 258
```

creates:

```text
"hell"
```

Then:

```text
(258,111) -> 259
```

creates:

```text
"hello"
```

The resulting hierarchy is:

```text
h + e
  |
  v
 he

he + l
   |
   v
 hel

hel + l
    |
    v
 hell

hell + o
     |
     v
 hello
```

This demonstrates how BPE learns larger representations from smaller frequently occurring components.

---

# Vocabulary Growth

If the target vocabulary size is:

```text
270
```

and the initial byte vocabulary contains:

```text
256
```

tokens, then:

\[
270-256=14
\]

new tokens can be learned.

The vocabulary therefore grows as:

```text
0–255
256
257
258
...
269
```

The general relationship is:

\[
M=V-256
\]

where:

- \(M\) = number of learned merges
- \(V\) = target vocabulary size

---

# Merge Rules

Each learned merge is stored in a merge table.

For example:

```text
(104,101) -> 256
(256,108) -> 257
(257,108) -> 258
(258,111) -> 259
```

The merge table is important during inference because it tells the encoder how to construct larger tokens.

The two primary tokenizer data structures are:

```text
Vocabulary:
token ID -> underlying bytes

Merge Table:
(token A, token B) -> new token ID
```

The vocabulary answers:

> What does this token represent?

The merge table answers:

> How can this token be constructed from previously learned tokens?

---

# Mathematical Formulation

Let the initial token sequence be:

\[
T^{(0)}
\]

where every token corresponds to a byte.

At iteration \(k\), define the pair-frequency function:

\[
C_k(a,b)
=
\sum_i
\mathbf{1}
[t_i^{(k)}=a
\land
t_{i+1}^{(k)}=b]
\]

Select the most frequent pair:

\[
(a_k,b_k)
=
\arg\max_{(a,b)}
C_k(a,b)
\]

Assign a new token ID:

\[
v_k=256+k
\]

Store the merge:

\[
(a_k,b_k)
\rightarrow
v_k
\]

Construct the new token's byte representation:

\[
B(v_k)
=
B(a_k)\Vert B(b_k)
\]

Then replace every occurrence of the selected pair:

\[
T^{(k+1)}
=
Merge
\left(
T^{(k)},
(a_k,b_k),
v_k
\right)
\]

The process continues until the desired vocabulary size is reached.

---

# Encoding

After training, a new text input is converted into UTF-8 bytes.

For example:

```text
hello world
```

becomes:

```text
[104, 101, 108, 108, 111, 32,
 119, 111, 114, 108, 100]
```

The tokenizer then applies the learned merge rules in their learned order.

The experiment learned:

```text
(104,101) -> 256
(256,108) -> 257
(257,108) -> 258
(258,111) -> 259
(259,32)  -> 260
```

Therefore:

```text
hello
```

progressively becomes:

```text
104 101 108 108 111
      |
      v
256 108 108 111
      |
      v
257 108 111
      |
      v
258 111
      |
      v
259
```

Then:

```text
259 32 -> 260
```

so:

```text
"hello "
```

becomes:

```text
260
```

The word:

```text
world
```

is similarly learned as:

```text
262 -> "wo"
263 -> "wor"
264 -> "worl"
265 -> "world"
```

Thus:

```text
hello world
```

becomes:

```text
[260, 265]
```

---

# Decoding

Given:

```text
[260, 265]
```

the decoder looks up the underlying bytes.

```text
260 -> [104,101,108,108,111,32]

265 -> [119,111,114,108,100]
```

The byte sequences are concatenated:

```text
[104,101,108,108,111,32,
 119,111,114,108,100]
```

and decoded using UTF-8:

```text
hello world
```

The important point is that decoding does not need to reconstruct the individual merge operations.

Each vocabulary entry already contains its complete underlying byte sequence.

---

# Round-Trip Property

A correct tokenizer should satisfy:

\[
Decode(Encode(x))=x
\]

for valid input text \(x\).

For the experiment:

```text
Original:
hello world

Encoded:
[260, 265]

Decoded:
hello world
```

Therefore:

```text
Decode(Encode("hello world"))
=
"hello world"
```

The implementation successfully verified this property.

---

# Experimental Configuration

The supplied experiment used the following training corpus:

```text
hello hello hello hello world hello world hello hello world world
```

The tokenizer configuration was:

```text
Initial vocabulary size: 256
Target vocabulary size: 270
Number of merges: 14
```

The training text was first converted into UTF-8 bytes and then represented as the initial token sequence.

---

# Training Results

The first BPE iteration identified:

```text
(104, 101) -> 256
```

with frequency:

```text
7
```

The new token represented:

```text
"he"
```

The token sequence length changed from:

```text
66 -> 59
```

The second iteration learned:

```text
(256,108) -> 257
```

representing:

```text
"hel"
```

and reduced the sequence length:

```text
59 -> 52
```

The third iteration learned:

```text
(257,108) -> 258
```

representing:

```text
"hell"
```

and reduced the sequence length:

```text
52 -> 45
```

The fourth iteration learned:

```text
(258,111) -> 259
```

representing:

```text
"hello"
```

and reduced the sequence length:

```text
45 -> 38
```

These results demonstrate the hierarchical nature of BPE learning.

---

# Learned Vocabulary

The final vocabulary size was:

```text
270
```

with:

```text
14 learned BPE merges
```

The learned tokens included:

| Token ID | Learned representation |
|---:|---|
| 256 | `he` |
| 257 | `hel` |
| 258 | `hell` |
| 259 | `hello` |
| 260 | `hello ` |
| 261 | `hello hello ` |
| 262 | `wo` |
| 263 | `wor` |
| 264 | `worl` |
| 265 | `world` |
| 266 | `world ` |
| 267 | `hello hello world ` |
| 268 | `hello hello hello hello world ` |
| 269 | `hello hello hello hello world hello ` |

These tokens were learned entirely from the supplied training corpus.

One important observation is that the tokenizer does not restrict tokens to individual words.

It learned:

```text
he
hel
hell
hello
wo
wor
worl
world
```

as well as larger sequences such as:

```text
hello hello world
```

because the training corpus contained repeated instances of those sequences.

---

# Learned Merge Rules

The tokenizer learned the following 14 merge rules:

```text
(104, 101) -> 256
(256, 108) -> 257
(257, 108) -> 258
(258, 111) -> 259
(259, 32)  -> 260
(260, 260) -> 261
(119, 111) -> 262
(262, 114) -> 263
(263, 108) -> 264
(264, 100) -> 265
(265, 32)  -> 266
(261, 266) -> 267
(261, 267) -> 268
(268, 260) -> 269
```

These rules show the complete hierarchy learned during training.

The first group learns the word:

```text
h
he
hel
hell
hello
hello 
```

The second group learns:

```text
w
wo
wor
worl
world
world 
```

The later merges occur because the training corpus contains repeated multi-word sequences.

---

# Encoding Result

The trained tokenizer was tested on:

```text
hello world
```

The initial byte sequence was:

```text
[104, 101, 108, 108, 111, 32,
 119, 111, 114, 108, 100]
```

The learned merges progressively transformed this sequence.

The final token IDs were:

```text
[260, 265]
```

where:

```text
260 -> "hello "
265 -> "world"
```

The complete encoding trace was produced by the implementation itself.

---

# Decoding Result

The encoded representation:

```text
[260, 265]
```

was decoded using the learned vocabulary.

The decoder retrieved:

```text
260 -> [104,101,108,108,111,32]

265 -> [119,111,114,108,100]
```

After concatenation:

```text
[104,101,108,108,111,32,
 119,111,114,108,100]
```

UTF-8 decoding produced:

```text
hello world
```

The round-trip test reported:

```text
Original:      'hello world'
Reconstructed: 'hello world'

SUCCESS: encode -> decode is lossless!
```

The tokenizer was also saved to:

```text
bpe_tokenizer.txt
```



---

# Implementation Structure

The implementation is organized around the following functions:

```text
text_to_bytes()
        |
        v
text_to_token_sequence()
        |
        v
count_pairs()
        |
        v
get_most_frequent_pair()
        |
        v
merge_pair()
        |
        v
train_bpe()
        |
        +------> Vocabulary
        |
        +------> Merge Rules
        |
        v
encode()
        |
        v
Token IDs
        |
        v
decode()
        |
        v
Original Text
```

The tokenizer also includes:

```text
save_tokenizer()
load_tokenizer()
```

for persistence.

---

# Core Data Structures

## Vocabulary

The vocabulary maps a token ID to its underlying bytes:

```python
vocab[token_id] = token_bytes
```

Example:

```text
104 -> b'h'
101 -> b'e'
256 -> b'he'
257 -> b'hel'
258 -> b'hell'
259 -> b'hello'
```

## Merge Table

The merge table maps a pair of tokens to a newly created token:

```python
merges[(token_a, token_b)] = new_token
```

Example:

```text
(104,101) -> 256
(256,108) -> 257
(257,108) -> 258
(258,111) -> 259
```

Together, these structures define the learned tokenizer.

---

# BPE as Learned Compression

BPE can be interpreted as a form of learned compression.

Initially:

```text
h e l l o
```

requires:

```text
5 tokens
```

After learning:

```text
hello -> 259
```

the same sequence can be represented as:

```text
259
```

requiring:

```text
1 token
```

The tokenizer therefore discovers recurring sequences and assigns them compact reusable representations.

The more frequently a sequence appears in the training corpus, the more likely its constituent parts are to be merged during BPE training.

---

# Why Tokens Are Not Necessarily Words

BPE should not be thought of as a simple word splitter.

A tokenizer may learn:

```text
he
ing
tion
hello
world
```

or portions of words.

It may also learn:

```text
" world"
```

including the leading space.

Depending on the training corpus and the surrounding tokenization pipeline, tokens can represent:

- individual bytes
- byte sequences
- character fragments
- subwords
- complete words
- whitespace
- repeated multi-word sequences

Therefore:

> A token is a learned reusable unit, not necessarily a word.

---

# Handling Unseen Text

One advantage of byte-level tokenization is that the tokenizer does not need to have seen every complete word during training.

Suppose the tokenizer encounters a new word:

```text
Supercalifragilistic
```

Even if the exact word was never present in the training corpus, the tokenizer can begin from its UTF-8 bytes and apply whatever learned byte-sequence merges are available.

Conceptually:

```text
Unknown word
     |
     v
UTF-8 bytes
     |
     v
Known byte-level units
     |
     v
Learned BPE merges
     |
     v
Token sequence
```

This gives byte-level tokenization broad coverage over arbitrary input.

---

# Relationship to Modern GPT Tokenizers

This project implements the **core byte-level BPE mechanism** rather than claiming to reproduce the complete tokenizer used by a specific modern GPT model.

A production GPT-family tokenizer can contain additional components such as:

```text
Raw Text
    |
    v
Text Pre-tokenization
    |
    v
Byte Representation
    |
    v
BPE
    |
    v
Special Token Handling
    |
    v
Token IDs
```

Production implementations also use optimized algorithms and data structures for large-scale training and inference.

This project intentionally focuses on the underlying BPE mechanism so that every major operation can be inspected directly.

---

# Limitations

This implementation is designed for learning and algorithmic understanding rather than production use.

The current implementation uses a straightforward approach that repeatedly:

- scans the token sequence
- counts adjacent pairs
- identifies the most frequent pair
- performs the merge

This becomes expensive for large corpora.

A production tokenizer would require significantly more efficient implementations for:

- pair-frequency updates
- memory management
- large vocabularies
- large training corpora
- parallel processing
- fast encoding
- special-token handling
- text pre-tokenization
- deterministic behavior across all edge cases

The current implementation also intentionally omits several tokenizer-specific components used by production GPT-family tokenizers.

---

# Implementation Philosophy

The project intentionally avoids abstraction-heavy libraries.

The objective is to expose the algorithm itself.

The implementation uses basic Python functionality for:

```text
UTF-8 encoding
Dictionary-based counting
List manipulation
Byte concatenation
File I/O
```

The training process prints intermediate information including:

```text
Pair frequencies
Selected pair
Pair frequency
New token ID
Underlying bytes
Decoded representation
Sequence length before and after merging
Current token sequence
```

This makes it possible to observe exactly how the tokenizer learns its vocabulary.

---

# Example End-to-End Flow

Consider:

```text
hello world
```

The complete pipeline is:

```text
                     "hello world"
                           |
                           v
                     UTF-8 Encoding
                           |
                           v
        [104,101,108,108,111,32,119,111,114,108,100]
                           |
                           v
                   Initial Byte Tokens
                           |
                           v
                    Apply BPE Rules
                           |
                           v
                       [260,265]
                           |
                           v
                        Decode
                           |
                           v
                     "hello world"
```

The learned vocabulary contains:

```text
260 -> "hello "
265 -> "world"
```

Therefore the input can be represented using only two learned tokens.

---

# Key Takeaways

The most important concepts demonstrated by this implementation are:

### Byte-Level Initialization

The tokenizer starts with:

```text
256 primitive byte tokens
```

corresponding to:

```text
0–255
```

### Frequency-Based Learning

At each iteration, BPE finds:

\[
(a,b)
=
\arg\max C(a,b)
\]

the most frequent adjacent token pair.

### Vocabulary Expansion

A new token is created:

```text
256
257
258
...
```

as new pairs are learned.

### Hierarchical Representation

Tokens are progressively constructed:

```text
h + e
  -> he

he + l
   -> hel

hel + l
    -> hell

hell + o
     -> hello
```

### Merge Rules

The tokenizer stores:

```text
pair -> token ID
```

so the learned behavior can be reproduced during encoding.

### Lossless Reconstruction

A correctly functioning tokenizer should satisfy:

\[
Decode(Encode(x))=x
\]

for valid input text.

### Tokens Are Learned Units

Tokens do not necessarily correspond to complete words.

They can represent arbitrary recurring byte sequences.

---

# Repository Structure

A minimal repository can be organized as:

```text
gpt-bpe-tokenizer/
|
├── bpe_tokenizer.py
├── bpe_tokenizer.txt
└── README.md
```

### `bpe_tokenizer.py`

Contains the complete implementation:

- byte conversion
- vocabulary initialization
- pair counting
- BPE training
- merge operations
- encoding
- decoding
- tokenizer saving
- tokenizer loading

### `bpe_tokenizer.txt`

Contains the learned vocabulary and merge rules.

### `README.md`

Contains the project explanation, architecture, mathematical formulation, implementation details, and experimental results.

---

# Final Insight

The core BPE algorithm can be reduced to a simple iterative process:

```text
Start with primitive byte tokens
            |
            v
Count adjacent pairs
            |
            v
Find the most frequent pair
            |
            v
Create a new token
            |
            v
Store the merge rule
            |
            v
Replace the pair
            |
            v
Repeat
```

Mathematically:

\[
(a_k,b_k)
=
\arg\max_{(a,b)} C_k(a,b)
\]

followed by:

\[
(a_k,b_k)\rightarrow v_k
\]

with:

\[
B(v_k)
=
B(a_k)\Vert B(b_k)
\]

Repeated application transforms primitive byte-level representations into a learned vocabulary of reusable token sequences.

The experiment demonstrates this progression directly:

```text
he
  ↓
hel
  ↓
hell
  ↓
hello
```

and eventually:

```text
hello world
     ↓
[260,265]
```

with successful reconstruction back to:

```text
hello world
```

The project therefore provides a transparent implementation of the core idea behind BPE-based tokenization and serves as a foundation for understanding how tokenizers interface with modern language models.
