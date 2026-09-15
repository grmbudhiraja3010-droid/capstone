# Custom RAG Pipeline — README 

## 1. Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions from a local collection of domain documents.

The main design goal was **transparency**: instead of hiding the retrieval process behind LangChain/LlamaIndex abstractions, the important stages were implemented explicitly with Python, NumPy, JSON, regular expressions, and the Google GenAI SDK.

The final pipeline covers:

**Documents → Cleaning → Chunking → Embedding → Local Vector Store → Retrieval → Document Deduplication → LLM Generation → LLM-based Quality Evaluation**

The system also includes persistent caching and rate-limit handling so that repeated experiments do not unnecessarily consume API quota.

> **Important implementation note:** the current final code is a dense-vector NumPy RAG implementation. It does **not** currently implement BM25, Reciprocal Rank Fusion (RRF), or a cross-encoder reranker. Those are future extensions, not components.

---

## 2. What I Built

I built a small but complete RAG evaluation pipeline.

It has five practical goals:

1. Load and clean a document collection.
2. Convert the documents into retrievable chunks and embeddings.
3. Retrieve the most relevant parent documents for a query.
4. Generate an answer using only retrieved context.
5. Measure retrieval quality and generation quality separately.

This separation was important because a good-looking answer from an LLM does not automatically prove that retrieval was good.

---

## 3. Final Architecture

```text
Local .txt Documents
        ↓
Document Loading
        ↓
Whitespace Cleaning
        ↓
Character-based Chunking
        ↓
Gemini Embeddings
gemini-embedding-001
        ↓
NumPy Vector Store
embeddings.npy
        ↓
Cosine Similarity Retrieval
        ↓
Parent Document ID Deduplication
        ↓
Top-K Context
        ↓
Gemini Generation
gemini-3.6-flash
        ↓
Grounded Answer
        ↓
LLM-as-a-Judge
        ↓
Faithfulness + Answer Relevance
```

The embedding matrix is saved in `embeddings.npy`, while chunk metadata/text is saved in `chunks.json`.

---

# 4. Engineering Log

## Phase 1 — Repository and input setup

### Task
Create a reproducible local RAG environment and define the input/output files.

### Implementation
The script defines:

- `data/` — source documents
- `embeddings.npy` — saved embedding matrix
- `chunks.json` — saved chunk metadata and text
- `index_metadata.json` — indexing configuration
- `generated_answers_cache.json` — persistent generation cache

The API key is loaded through `.env` rather than hard-coded into the script.

### Why
Keeping credentials outside source code prevents accidental exposure and makes the project easier to run in another environment.

---

## Phase 2 — Document ingestion

### Task
Read the document collection deterministically.

### Implementation
The code scans:

```python
DATA_DIR.glob("*.txt")
```

and sorts the filenames before loading them.

Each document is represented by:

```python
{
    "doc_id": file.stem,
    "text": text
}
```

### Why sorting matters
A deterministic ordering makes experiments easier to reproduce. If the input order changes between runs, debugging retrieval differences becomes unnecessarily difficult.

---

## Phase 3 — Text cleaning

### Task
Remove unnecessary whitespace while preserving the text content.

### Implementation

Newlines are converted into spaces and repeated whitespace is collapsed with a regular expression.

### Why
The project uses character-based chunking. Cleaning the text before chunking gives more consistent chunk boundaries.

I intentionally kept this cleaning stage simple instead of aggressively transforming the source text.

---

## Phase 4 — Chunking

### Configuration

```text
Chunk size = 800 characters
Overlap    = 150 characters
```

The next chunk begins at:

```text
start + CHUNK_SIZE - OVERLAP
```

### Why overlap?

Without overlap, an important sentence near a chunk boundary can be separated from the surrounding context.

The 150-character overlap provides continuity between neighbouring chunks.

### Validation
The script explicitly checks whether the expected overlap exists between the first two chunks.

---

## Phase 5 — Embedding generation

### Model

```text
gemini-embedding-001
```

Each chunk is converted into a dense vector.

The final embedding matrix is stored as a NumPy array.

### Important optimization

The first version of the pipeline could make an API call for every individual chunk. That is inefficient under a free-tier request limit.

The final implementation therefore batches embeddings:

```text
Batch size = 16 chunks
```

For approximately 204 chunks, this means roughly 13 embedding requests instead of roughly 204 individual requests.

A deliberate six-second pause is also inserted between successful batches.

---

## Phase 6 — Rate-limit handling

### Problem discovered

Embedding APIs can return HTTP 429 / `RESOURCE_EXHAUSTED` errors.

A naive retry loop can make the situation worse if it immediately retries again.

### Final approach

The code:

1. Detects rate-limit errors.
2. Attempts to read Google's requested `retryDelay`.
3. Adds a small safety margin.
4. Falls back to exponential backoff if no server delay can be extracted.
5. Limits the number of retries.

This made the embedding stage more robust under constrained API quotas.

---

## Phase 7 — Persistent index

After embedding, the system saves:

```text
embeddings.npy
chunks.json
index_metadata.json
```

The metadata records the embedding model, chunk size, overlap, document count, chunk count, embedding dimension, and batch size.

### Why this matters

The document embeddings are expensive relative to a local NumPy lookup.

Therefore, normal retrieval should **not** regenerate document embeddings.

The intended workflow is:

```text
Indexing:
documents → chunks → document embeddings → save

Later queries:
query → query embedding → local retrieval
```

---

## Phase 8 — Index validation

Before retrieval, the code checks that:

```text
number of embedding rows
=
number of chunks
```

If they differ, the index is rejected.

This protects against accidentally pairing an old `embeddings.npy` with a new `chunks.json`.

This was an important reproducibility safeguard because changing the documents or chunking strategy invalidates the old index.

---

## Phase 9 — Cosine similarity retrieval

The document embeddings are normalized once.

For a query, only the new query is embedded.

Cosine similarity is then computed using a matrix multiplication:

```python
scores = normalized_embeddings @ normalized_query
```

This is the core retrieval operation.

Conceptually:

```text
query vector
     ↓
compare against every stored document-chunk vector
     ↓
similarity score
     ↓
rank candidates
```

No external vector database is required for this scale of experiment.

---

## Phase 10 — Parent-document deduplication

### Problem

The vector store operates at the **chunk** level, but the evaluation ground truth is defined at the **document** level.

A query can therefore retrieve:

```text
doc01 chunk A
doc01 chunk B
doc01 chunk C
doc07 chunk A
doc08 chunk A
```

If these are counted independently, one document can dominate the top-K results.

### Final solution

The retriever obtains a larger candidate pool and then performs order-preserving deduplication:

```text
candidate chunks
      ↓
read parent doc_id
      ↓
keep first occurrence
      ↓
discard repeated doc_id
      ↓
return K unique documents
```

The code uses a `seen_docs` set.

### Why this matters

This makes the retrieval window more diverse and aligns the retrieval output with the document-level evaluation labels.

---

# 5. Retrieval Evaluation

The evaluation set contains both normal in-context questions and an out-of-context question.

The out-of-context FIFA question is deliberately skipped by the retrieval accuracy calculation because it has no relevant source document.

For normal questions, the final code reports:

- Hit@1
- Hit@3
- Recall@3
- Recall@5
- AP@5
- MAP@5

## Hit@1

Was the correct document ranked first?

```text
1 = yes
0 = no
```

This measures the quality of the first retrieval decision.

## Hit@3

Did at least one relevant document appear in the first three unique documents?

## Recall@3

How many of the known relevant documents were found within the first three?

## Recall@5

How many of the known relevant documents were found within the first five?

This is particularly useful for multi-document questions.

## AP@5

Average Precision@5 rewards relevant documents appearing earlier in the ranking.

It therefore captures ranking quality rather than only whether a relevant document appeared somewhere in the result list.

## MAP@5

Mean Average Precision@5 is the mean AP@5 across all evaluated questions.

---

# 6. Generation Layer

The generation model is configured as:

```text
gemini-3.6-flash
```

The prompt explicitly instructs the model to answer using only the retrieved context.

If the context is insufficient, the model is instructed to say that the information is not available in the provided documents.

This creates a clear grounding boundary between:

```text
retrieval evidence
```

and

```text
model knowledge
```

---

# 7. Persistent Generation Cache

The file:

```text
generated_answers_cache.json
```

stores previous answers.

For a repeated question:

```text
Question
   ↓
Cache lookup
   ↓
Found?
 ┌─Yes → reuse saved answer
 │
 └─No  → call LLM → save answer
```

This reduces unnecessary generation API calls during repeated experiments.

The cache is also useful when an experiment is interrupted halfway through.

---

# 8. Generation Evaluation

The project uses a second LLM call as an evaluator.

It scores:

### Faithfulness

Does the generated answer remain supported by the retrieved context?

### Answer Relevance

Does the answer actually address the user's question?

The evaluator returns structured JSON containing:

```text
faithfulness_score
faithfulness_reason
relevance_score
relevance_reason
```

The code then averages these scores across successfully evaluated records.

### Why separate these metrics?

An answer can be:

- relevant but unsupported,
- supported but incomplete,
- both relevant and faithful,
- neither.

Therefore, retrieval quality and generation quality should not be collapsed into one number.

---

# 9. Why the Evaluation Was Decoupled

The final script separates generation from generation evaluation.

## Phase 1

Generate answers and save them.

## Phase 2

Evaluate the saved answers.

This design was introduced because generation and evaluation each consume API requests.

Separating the phases makes the run easier to resume and reduces the chance that an intermediate quota failure destroys all previous results.

---

# 10. Key Findings and Experimental Interpretation

The supplied benchmark notes reported the following comparison:

| Metric | K=3 | K=5 | Interpretation |
|---|---:|---:|---|
| Hit@1 | 0.8235 | 0.8235 | First result unchanged |
| Hit@3 | 0.8824 | 0.8824 | Top-3 boundary unchanged |
| Recall@3 | 0.8333 | 0.8333 | Top-3 coverage unchanged |
| Recall@5 | 0.8333 | 0.8627 | More relevant evidence captured |
| MAP@5 | 0.8039 | 0.8333 | Ranking/coverage improved |
| Avg search latency | 684.80 ms | 676.07 ms | Essentially similar |
| Avg input tokens/query | 493.20 | 765.20 | Larger context |
| Estimated cost / 10k queries | $6.2888 | $9.7568 | Higher prompt cost |

### Main conclusion

K=5 is a reasonable operating point when retrieval coverage is more important than minimizing context size.

The quality gain comes mainly from the additional retrieval positions, not from an improvement in the first three results.

The additional context also has a measurable token/cost trade-off.

---

# 11. Problems Encountered and What I Changed

## Problem 1 — Too many embedding API calls

### Symptom
Embedding every chunk individually increased request pressure.

### Fix
Batch 16 chunks per embedding request and add controlled pacing.

---

## Problem 2 — 429 rate limits

### Symptom
The API returned `RESOURCE_EXHAUSTED`.

### Fix
Read server retry delays and use bounded exponential backoff.

---

## Problem 3 — Stale index risk

### Symptom
Changing documents or chunking can leave an old embedding matrix on disk.

### Fix
Validate:

```text
embeddings.shape[0] == len(chunks)
```

and store indexing metadata.

---

## Problem 4 — Duplicate parent documents

### Symptom
Several top chunks can come from the same document.

### Fix
Oversample candidates, then perform order-preserving document-ID deduplication.

---

## Problem 5 — Generation quota consumption

### Symptom
Repeated runs can repeatedly call the LLM for the same question.

### Fix
Persist generated answers in `generated_answers_cache.json`.

---

## Problem 6 — Mixing generation and evaluation

### Symptom
A failure during evaluation can make a long generation run harder to recover.

### Fix
Separate generation and evaluation into two phases.

---

# 12. What I Would Do Differently

The most important next improvement would be to improve **retrieval quality before deduplicating the context**.

The current system selects the first high-ranked chunk from each parent document. This is simple and defensible, but it can discard another chunk from the same document that may contain better evidence.

A stronger design would be:

```text
Query
 ↓
Dense candidate retrieval
 ↓
Wider candidate pool
 ↓
Optional lexical retrieval
 ↓
Candidate fusion
 ↓
Cross-encoder reranking
 ↓
Best chunk per document
 ↓
Final K documents
 ↓
LLM
```

This would make the ranking decision more semantic before the final document-level diversity constraint is applied.

I would also add controlled ablation experiments rather than adding components without measuring their contribution.

For example:

```text
Dense only
vs
Dense + lexical
vs
Dense + lexical + reranker
```

Then compare Hit@1, Recall@K, MAP@K, latency, token usage, and cost.

---

# 13. Future Scope — Enterprise Knowledge Assistant

**Enterprise Knowledge Assistant:** PDF/document → Parser → Chunking → Embedding → Vector DB → Retriever → Reranker → LLM → Answer → Citation, implemented with Python and optionally LangChain/LlamaIndex, Hugging Face, FAISS/Chroma, FastAPI and Docker while keeping the underlying retrieval and ranking mechanics understandable rather than treating frameworks as black boxes.

---

# 14. Repository Structure

```text
rag-track-a/
│
├── data/
│   ├── doc01.txt
│   ├── doc02.txt
│   └── ...
│
├── rag.py
├── embeddings.npy
├── chunks.json
├── index_metadata.json
├── generated_answers_cache.json
├── .env
├── .gitignore
├── README.md
└── presentation_script.md
```


---

# 15. Takeaway

The main learning from this project was that a RAG system is not just:

```text
embed → retrieve → ask an LLM
```

The difficult engineering work is around the boundaries:

- What exactly is being indexed?
- How are chunks created?
- Are saved vectors still aligned with the current chunks?
- Are retrieval metrics measured at chunk or document level?
- Does the ranking contain duplicate sources?
- How much context is actually useful?
- What happens when the API rate limit is reached?
- Can an experiment be resumed without repeating expensive calls?
- Is the generated answer actually supported by retrieved evidence?

Building these pieces explicitly made the system easier to inspect, debug, and reason about.
