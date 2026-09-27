# ml_hackathon_26

 Absolutely. Here is the **Amazon ML Challenge problem statement explained in simple terms**, based directly on the PDF you uploaded.

## 1. What is the challenge about?

The main problem is called **Business Entity Resolution**.

Imagine there are **three different databases** containing information about businesses:

* **Source 1** → Reference/master list
* **Source 2** → Another database
* **Source 3** → Another database

The same business may appear in multiple sources, but the information may look different.

For example:

**Source 1**

> ABC Technologies Pvt Ltd
> 12 MG Road, Delhi

**Source 2**

> ABC Tech Private Limited
> 12, M.G. Rd, New Delhi

**Source 3**

> A.B.C. Technologies
> Near MG Road, Delhi

A human can understand that these are probably the **same business**, even though the names and addresses are different.

Your job is to build an **ML system that automatically finds these matches**. 

---

# 2. What exactly do we have to predict?

**Source 1 is the reference source.**

For **every business in Source 1**, you need to find all corresponding businesses in:

* Source 2
* Source 3

A Source 1 business can have:

* **No match**
* **One match**
* **Multiple matches**

For example:

| Source 1 | Matching Source 2/3 |
| -------- | ------------------- |
| S1-00001 | S2-00047, S3-00812  |
| S1-00002 | S3-00004            |
| S1-00003 | No match            |

This is the central task of the competition. 

---

# 3. Why is this difficult?

The data contains **noisy and inconsistent information**.

### Business names can differ

For example:

* Corporation → Corp
* Private → Pvt
* Limited → Ltd
* spelling mistakes
* punctuation differences
* different word order
* different trade/DBA names

The PDF specifically lists these types of name variations. 

### Addresses can differ

For example:

> 12 MG Road, Delhi

might appear as:

> 12 M.G. Rd, Delhi

or:

> Near SBI ATM, MG Road

or have missing PIN/state information.

The challenge therefore requires comparing **imperfect strings**, rather than simply checking whether two records are identical. 

---

# 4. What information does each business record contain?

Each source contains:

### `entity_id`

Unique ID of the record.

Examples:

```text
S1-00001
S2-00047
S3-00812
```

The prefix tells you which source it belongs to.

### `business_name`

Name of the business.

### `business_address`

Address of the business.

### `country`

Country of the business.

The training data contains **US and India**, while the test set additionally contains **France**. You therefore should not build your solution assuming that only US and India will occur. 

---

# 5. What data is given for training?

You get four training files:

```text
train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv
```

The first three contain business records.

The fourth tells you the **correct matches**.

For example:

```text
source1_entity_id    matched_entity_ids
S1-00001             S2-00047,S3-00812
S1-00002             S3-00004
S1-00003
```

So you can use this information to **train and validate your ML approach**. 

---

# 6. What happens with the test data?

For the test set, you get:

```text
test_source1.tsv
test_source2.tsv
test_source3.tsv
```

But there is **no ground truth** for the test set.

Your model has to predict the matches.

For **every Source 1 entity**, you must produce exactly one row in your final output. 

---

# 7. What does the ML pipeline look like?

The easiest way to understand the challenge is:

```text
Raw Business Data
       ↓
Data Cleaning / Normalization
       ↓
Candidate Generation / Blocking
       ↓
Feature Engineering
       ↓
Matching Model
       ↓
Final Match / No Match
       ↓
matching_results.tsv
```

### Step 1: Cleaning

Clean business names and addresses.

For example:

```text
"ABC Technologies Pvt. Ltd."
          ↓
"abc technologies private limited"
```

You might normalize:

* lowercase
* punctuation
* common abbreviations
* whitespace

---

### Step 2: Candidate Generation / Blocking

This is very important.

Suppose Source 1 contains **100,000 businesses** and Source 2 contains **500,000 businesses**.

You don't want to compare every possible pair.

Instead, find a smaller set of **potential matches**.

For example:

```text
S1-00001
   ↓
Possible candidates:
   S2-00125
   S2-00472
   S3-00891
   S3-01234
```

This is called **blocking/candidate generation**.

The challenge specifically says that the candidate file should contain the records your final matching model actually considers. 

---

# 8. Then how does the model decide if two businesses match?

For every candidate pair, you can create features such as:

### Name similarity

For example:

```text
ABC Technologies Pvt Ltd
ABC Technologies Private Limited
```

Could have high similarity.

### Address similarity

Compare:

```text
12 MG Road Delhi
12 M.G. Rd, New Delhi
```

### Country match

```text
India vs India → Match
India vs France → probably not
```

The challenge suggests techniques such as:

* **Jaccard similarity**
* **Levenshtein similarity**
* **TF-IDF cosine similarity**

for names and addresses. 

You can then feed these features into an ML model to predict whether the pair represents the **same business**.

---

# 9. What is the final output?

You have to produce:

### `matching_results.tsv`

This is the **important file that is scored**.

Example:

```text
source1_entity_id    matched_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812
S1-00002             S3-00004
S1-00003
```

Notice:

**S1-00003 has no match**, so its `matched_entity_ids` is empty.

Every Source 1 entity must have exactly one row. 

---

# 10. What is `candidate_pairs.tsv`?

This is different from your final answer.

Suppose your blocking stage says:

```text
S1-00001 → S2-00047, S2-00193, S3-00812, S3-00999
```

These are your **candidates**.

Then your final ML model decides:

```text
S1-00001 → S2-00047, S2-00193, S3-00812
```

So:

**candidate_pairs = possible matches**

**matching_results = final matches**

The final matches must always be a **subset of the candidates**. 

---

# 11. How is your model evaluated?

This is one of the **most important parts**.

The competition uses:

$$
\boxed{F_{0.5}}
$$

It is a precision-heavy metric.

The formula is:

$$
F_{0.5}
=
\frac{1.25\times Precision\times Recall}
{0.25\times Precision+Recall}
$$

The challenge gives more importance to **precision than recall**. 

### Why?

Consider this:

Your model says:

> "These two businesses are the same."

But they are actually two different businesses.

That's a **false merge**.

In real-world business data, incorrectly merging two different businesses can be particularly harmful.

Therefore, the challenge emphasizes avoiding incorrect matches. 

---

# 12. What about businesses with NO match?

This is called a **singleton** in the problem statement.

Suppose:

```text
S1-00003
```

doesn't exist in Source 2 or Source 3.

Your model should output:

```text
S1-00003
```

with an empty match list.

The challenge explicitly says that correctly identifying such a singleton receives a score of **1.0 for that entity**, whereas incorrectly predicting a match receives **0.0**. 

So **don't force every business to have a match**.

---

# 13. Important restriction: No external data

This is extremely important for the competition.

You **cannot** use external databases or APIs to look up businesses.

The PDF specifically prohibits:

* commercial entity-resolution APIs
* government business-registration databases
* geocoding APIs
* external internet data
* external data augmentation

The solution must use the **provided training data**. 

So your approach should be:

```text
Provided Data
     ↓
Cleaning
     ↓
Feature Engineering
     ↓
Blocking
     ↓
ML Model
     ↓
Predictions
```

and **not**:

```text
Provided Data
     ↓
Google/API/External Database
     ↓
Business Lookup
```

---

# 14. In one sentence, what are they asking you to do?

> **For every business in Source 1, use the noisy business names, addresses, countries, and patterns learned from the training data to identify which records in Source 2 and Source 3 refer to the same real-world business.**

---

## The whole problem in a simple example

Suppose you have:

**Source 1**

```text
S1-001 | Starbucks Coffee Pvt Ltd | MG Road, Delhi | India
```

**Source 2**

```text
S2-101 | Starbucks Coffee Private Limited | M.G. Road Delhi | India
S2-102 | ABC Restaurant | Connaught Place | India
```

**Source 3**

```text
S3-201 | Starbucks Coffee | MG Rd, New Delhi | India
S3-202 | Starbucks Coffee | Mumbai | India
```

Your system should determine:

```text
S1-001 → S2-101, S3-201
```

and produce:

```text
source1_entity_id    matched_entity_ids
S1-001               S2-101,S3-201
```

That's essentially the entire challenge.

### The 5 concepts you should remember

**1. Entity Resolution**
→ Find records that represent the same real-world business.

**2. Blocking**
→ Quickly reduce millions of possible comparisons to a manageable candidate set.

**3. Feature Engineering**
→ Calculate name/address/country similarity features.

**4. Matching Model**
→ Predict whether each candidate pair is actually the same business.

**5. F₀.₅ Evaluation**
→ Precision is emphasized, so incorrect merges are particularly costly.

