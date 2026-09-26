<h1 align = "center">Similarity Strategy </h1>
<br>
<h2>String similarity calculation can help us with any of these problems but generally computationally expensive and don’t automatically produce ideal outcomes due to the diverse and fuzzy nature of all the possible data faults.
</h2> 
<h2> TECHNIQUES FOR STRING SIMILARITY:</h2>
source : https://www.baeldung.com/cs/string-similarity-edit-distance <br>
https://www.baeldung.com/cs/string-similarity-token-methods
https://www.baeldung.com/cs/string-similarity-sequence-based

<h3>1.Character Based (Distance) <br></h3>

Levenshtein Distance<br>
Levenshtein distance, like Hamming distance, is the smallest number of edit operations required to transform one string into the other. 
Unlike Hamming distance, the set of edit operations also includes insertions and deletions, thus allowing us to compare strings of different lengths.
Given two strings, the Levenshtein distance between them is the minimum number of single-character edits (insertions, deletions, or substitutions) 
required to change one string into the other. <br>


Damerau-Levenshtein Distance<br>
It has been observed that most of the human misspelling errors fall into the errors of these 4 types – insertion, deletion, substitution, and transposition. Motivated by this empirical observation, Damerau-Levenshtein distance extends the set of the edit operations allowed in Levenshtein distance with transposition of two adjacent characters.

For example, the Levenshtein distance between GIFT and FIT is 3:

In this example, we need two steps to transform IF into FI. However, if we allow transposition between two adjacent characters, 
we can use only one step to make the transformation. Therefore, the Damerau-Levenshtein distance between GIFT and FIT is 2:

Incorporating transposition into the original Levenshtein distance computation algorithms can be relatively challenging. 
Straightforward modification of the dynamic programming approach used for Levenshtein distance computation with the entry responsible for 
transposition unexpectedly calculates not Damerau-Levenshtein distance but “optimal string alignment distance” which is not only different 
from the expected result but also doesn’t hold triangle inequality property.

Efficient algorithms to calculate Damerau-Levenshtein distance provide the same time complexity – \mathcal{O}( |a| \times |b| ).
<br>

Jaro Distance / Jaro Winkler <br>


<h3>2.Sequence Based (Distance) <br></h3>

Longest Common Subsquence or Longest Common Substring<br>

Gestalt Pattern Matching<br>

<h3>3.Token Based (Distance) <br></h3>

Instead of comparing two strings character-by-character, you first split the text into words (tokens) 
and then compare the words.

For example:

Business 1:
ABC Technologies Private Limited

Business 2:
ABC Technologies Pvt Ltd

After tokenization:

Business 1 → {ABC, Technologies, Private, Limited}

Business 2 → {ABC, Technologies, Pvt, Ltd}

Now the algorithm compares the tokens/words rather than the exact characters.

Sure. These are all **string similarity techniques** that can be useful in your business entity-resolution challenge. The easiest way to understand them is with one pair of strings.

Let's use:

* **A:** `ABC TECHNOLOGIES`
* **B:** `ABC TECHNOLOGY`

---

## 1. Q-gram

A **q-gram** breaks a string into overlapping pieces of length \(q\).

If \(q=2\), we create **2-grams (bigrams)**.

For example:

```text
ABC
```

becomes:

```text
AB
BC
```

For:

```text
ABC TECHNOLOGY
```

the 2-grams include:

```text
AB, BC, C ,  T, TE, EC, CH, HN, NO, OL, LO, OG, GY
```

Usually, preprocessing such as lowercase and removing unnecessary spaces/punctuation is done first.

### Why useful?

Q-grams are good at detecting **small spelling differences**.

For example:

```text
TECHNOLOGY
TECHNOLOGIES
```

have many common character chunks even though the endings differ.

### In your challenge

You could calculate:

```text
name_qgram_similarity
address_qgram_similarity
```

# 2. Jaccard Similarity

Jaccard compares the **intersection** of two sets with their **union**.

Formula:

$$
\boxed{
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
}
$$

Suppose we tokenize:

```text
A = {ABC, TECHNOLOGIES, PRIVATE, LIMITED}

B = {ABC, TECHNOLOGIES, PVT, LTD}
```


Total unique tokens:

```text
A ∪ B =
{ABC, TECHNOLOGIES, PRIVATE, LIMITED, PVT, LTD}
```

Therefore:

$$
J(A,B)=\frac{2}{6}=0.333
$$

### Interpretation

| Jaccard | Meaning          |
| ------: | ---------------- |
|       0 | No overlap       |
|     0.5 | Moderate overlap |
|       1 | Identical sets   |

### Good for

**Token/word overlap**, especially when word order changes.

---

# 3. Dice Coefficient

Dice is also based on the **intersection**, but it gives the intersection twice as much weight.

Formula:

$$
\boxed{
Dice(A,B)=\frac{2|A\cap B|}{|A|+|B|}
}
$$

Using our example:

$$
|A|=4
$$

$$
|B|=4
$$

$$
|A\cap B|=2
$$

Therefore:

$$
Dice=\frac{2(2)}{4+4}
$$

$$
=\frac{4}{8}
$$

$$
\boxed{Dice=0.5}
$$

### Jaccard vs Dice

They are closely related.
Both are measuring **overlap**, but they use different formulas.

---

# 4. Overlap Coefficient

The **Overlap Coefficient** asks:

> How much of the smaller set is contained in the larger set?

Formula:

$$
\boxed{
Overlap(A,B)=
\frac{|A\cap B|}
{\min(|A|,|B|)}
}
$$

Using:

$$
|A|=4,\quad |B|=4,\quad |A\cap B|=2
$$

we get:

$$
Overlap=\frac{2}{\min(4,4)}
$$

$$
=\frac{2}{4}
$$

$$
\boxed{Overlap=0.5}
$$

### Why is it different?

Consider:

```text
A = {ABC, TECHNOLOGIES}

B = {ABC, TECHNOLOGIES, PRIVATE, LIMITED, PVT, LTD}
```

Here:

$$
|A\cap B|=2
$$

The smaller set has only 2 elements.

Therefore:

$$
Overlap=\frac{2}{2}=1
$$

So the overlap coefficient says:

> **100% of the smaller set is contained in the larger set.**

That's useful when one business record contains **more information than another**.

---

# 5. Bag Distance

This one is slightly different.

A **bag** is like a set where **duplicates matter**.

For example:

```text
A = "ABC ABC TECHNOLOGY"
```

has two occurrences of `ABC`.

A normal set would remove the duplicate:

```text
{ABC, TECHNOLOGY}
```

but a bag/multiset keeps the counts:

```text
ABC → 2
TECHNOLOGY → 1
```

### Bag distance

A common definition of bag distance is based on the number of unmatched elements after accounting for common occurrences:

$$
\boxed{
D(A,B)=|A|+|B|-2|A\cap B|
}
$$

where the intersection accounts for the **minimum count** of each token.

For example:

```text
A = [ABC, ABC, TECHNOLOGY]

B = [ABC, TECHNOLOGY, LTD]
```

Counts:

```text
A:
ABC        → 2
TECHNOLOGY → 1

B:
ABC        → 1
TECHNOLOGY → 1
LTD        → 1
```

Common occurrences:

```text
ABC        → 1
TECHNOLOGY → 1
```

So:

$$
|A|=3
$$

$$
|B|=3
$$

$$
|A\cap B|=2
$$

Therefore:

$$
D=3+3-2(2)
$$

$$
\boxed{D=2}
$$

A **lower bag distance means the strings are more similar**.

---

### For your Amazon challenge

I'd think of them like this:

```text
Business Name / Address
          ↓
   Normalization
          ↓
 ┌────────┼─────────┐
 ↓        ↓         ↓
Q-gram   Jaccard   Dice
 ↓        ↓         ↓
 └────────┼─────────┘
          ↓
    Similarity Features
          ↓
      ML Model
``` 
**One distinction to remember:** Q-gram is a way of **representing the string**, while Jaccard/Dice/Overlap are **ways of measuring overlap** between the resulting sets/bags.
