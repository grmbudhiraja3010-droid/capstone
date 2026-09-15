from pathlib import Path
import numpy as np
import re
import os
import json
import time

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = Path("data")

EMBEDDINGS_FILE = Path("embeddings.npy")
CHUNKS_FILE = Path("chunks.json")
METADATA_FILE = Path("index_metadata.json")

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.6-flash"

CHUNK_SIZE = 800
OVERLAP = 150

# ------------------------------------------------------------
# IMPORTANT
#
# True  = create document embeddings again
# False = load previously saved document embeddings
#
# Use True ONLY when:
#   - documents changed
#   - chunking changed
#   - embedding model changed
#   - embeddings.npy does not exist
#
# After a successful indexing run, change this to False.
# ------------------------------------------------------------

REBUILD_EMBEDDINGS = False


# Batch size controls how many chunks are sent in ONE API request.
#
# 204 chunks / 16 = about 13 API requests.
#
# This is still far below 100 requests/minute, but we also
# deliberately pause between batches.
BATCH_SIZE = 16

# Delay between successful embedding batches.
#
# This is deliberately conservative for the free tier.
BATCH_DELAY = 6

# Maximum number of retries after a rate-limit/transient error.
MAX_RETRIES = 5


# ============================================================
# 2. SET UP GEMINI
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found.\n"
        "Check that your .env file contains:\n"
        "GEMINI_API_KEY=your_key_here"
    )

client = genai.Client(
    api_key=api_key
)


# ============================================================
# 3. LOAD DOCUMENTS
# ============================================================

documents = []

for file in sorted(DATA_DIR.glob("*.txt")):

    text = file.read_text(encoding="utf-8")

    documents.append({
        "doc_id": file.stem,
        "text": text
    })


print("=" * 60)
print("LOAD CHECK")
print("=" * 60)

print("Documents:", len(documents))

if len(documents) == 0:
    raise ValueError(
        "No .txt documents were found inside the data/ folder."
    )

for doc in documents[:3]:

    print("\n", doc["doc_id"])
    print(doc["text"][:300])


# ============================================================
# 4. CLEAN DOCUMENTS
# ============================================================

def clean_text(text):
    """
    Basic whitespace cleaning.

    Newlines are converted to spaces so that the original
    character-based chunking remains consistent.
    """

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


for doc in documents:

    doc["text"] = clean_text(doc["text"])


print("\n" + "=" * 60)
print("CLEANING CHECK")
print("=" * 60)

for doc in documents[:2]:

    print("\nDocument:", doc["doc_id"])
    print(doc["text"][:500])


# ============================================================
# 5. CHUNK DOCUMENTS
# ============================================================

chunks = []

for doc in documents:

    text = doc["text"]

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk_text = text[start:end].strip()

        # Do not create empty chunks.
        if chunk_text:

            chunks.append({
                "doc_id": doc["doc_id"],
                "text": chunk_text
            })

        start += CHUNK_SIZE - OVERLAP


print("\n" + "=" * 60)
print("CHUNKING CHECK")
print("=" * 60)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3]):

    print("\nChunk:", i)
    print("DOC:", chunk["doc_id"])
    print("Characters:", len(chunk["text"]))
    print(chunk["text"])


# ============================================================
# 6. CHECK OVERLAP
# ============================================================

print("\n" + "=" * 60)
print("OVERLAP CHECK")
print("=" * 60)

if len(chunks) >= 2:

    overlap = chunks[0]["text"][-OVERLAP:]

    print(
        "Overlap exists:",
        overlap == chunks[1]["text"][:OVERLAP]
    )

else:

    print("Not enough chunks to check overlap.")


# ============================================================
# 7. RATE-LIMIT HELPER
# ============================================================

def get_retry_delay(error, attempt):
    """
    Try to extract Google's requested retry delay.

    If it cannot be extracted, use exponential backoff.
    """

    error_text = str(error)

    # Examples:
    #
    # retryDelay: 41s
    # retryDelay='41s'
    # retryDelay": "41s"

    match = re.search(
        r"retryDelay[^\d]*(\d+)\s*s",
        error_text,
        flags=re.IGNORECASE
    )

    if match:

        server_delay = int(match.group(1))

        # Add a small safety margin.
        return server_delay + 3

    # Fallback:
    return min(2 ** attempt, 60)


# ============================================================
# 8. EMBED A BATCH OF TEXTS
# ============================================================

def embed_batch(texts, batch_number=None):

    if not texts:

        raise ValueError(
            "Cannot embed an empty batch."
        )

    for i, text in enumerate(texts):

        if not text or not text.strip():

            raise ValueError(
                f"Empty text found at position {i} in batch."
            )

    for attempt in range(MAX_RETRIES):

        try:

            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts
            )

            vectors = [
                np.array(
                    embedding.values,
                    dtype=np.float32
                )
                for embedding in response.embeddings
            ]

            if len(vectors) != len(texts):

                raise RuntimeError(
                    "Number of returned embeddings does not "
                    "match number of input texts."
                )

            return vectors

        except ClientError as e:

            error_text = str(e)

            is_rate_limit = (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota exceeded" in error_text.lower()
            )

            if not is_rate_limit:

                raise

            if attempt == MAX_RETRIES - 1:

                raise RuntimeError(
                    "Embedding failed after maximum retries "
                    "because of rate limiting."
                ) from e

            wait_time = get_retry_delay(
                e,
                attempt
            )

            print(
                "\nRate limit reached."
            )

            print(
                f"Waiting {wait_time} seconds "
                f"before retry "
                f"{attempt + 1}/{MAX_RETRIES}..."
            )

            time.sleep(wait_time)

        except Exception as e:

            if attempt == MAX_RETRIES - 1:

                raise RuntimeError(
                    "Embedding failed after maximum retries."
                ) from e

            wait_time = min(
                2 ** attempt,
                30
            )

            print(
                "\nTransient error:"
            )

            print(e)

            print(
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)


    raise RuntimeError(
        "Unexpected embedding failure."
    )


# ============================================================
# 9. CREATE DOCUMENT EMBEDDINGS
# ============================================================

def create_document_embeddings(chunks):

    all_texts = [
        chunk["text"].strip()
        for chunk in chunks
    ]

    total_chunks = len(all_texts)

    all_vectors = []

    total_batches = (
        total_chunks + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print("\n" + "=" * 60)
    print("DOCUMENT EMBEDDING")
    print("=" * 60)

    print("Total chunks:", total_chunks)
    print("Batch size:", BATCH_SIZE)
    print("Total API batches:", total_batches)

    for batch_number, start in enumerate(
        range(0, total_chunks, BATCH_SIZE),
        start=1
    ):

        end = min(
            start + BATCH_SIZE,
            total_chunks
        )

        batch = all_texts[start:end]

        print(
            f"\nEmbedding batch "
            f"{batch_number}-{total_batches}: "
            f"chunks {start + 1}-{end} "
            f"of {total_chunks}"
        )

        vectors = embed_batch(
            batch,
            batch_number=batch_number
        )

        all_vectors.extend(vectors)

        print(
            f"Batch {batch_number}/{total_batches} "
            f"completed."
        )

        # Pause between successful batches.
        #
        # This reduces the probability of hitting the
        # free-tier request limit.
        if end < total_chunks:

            print(
                f"Waiting {BATCH_DELAY} seconds "
                "before next batch..."
            )

            time.sleep(BATCH_DELAY)

    embeddings = np.array(
        all_vectors,
        dtype=np.float32
    )

    return embeddings


# ============================================================
# 10. SAVE INDEX
# ============================================================

def save_index(embeddings, chunks):

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "embedding_dimension": int(embeddings.shape[1]),
        "batch_size": BATCH_SIZE
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nEmbeddings saved to:", EMBEDDINGS_FILE)
    print("Chunks saved to:", CHUNKS_FILE)
    print("Metadata saved to:", METADATA_FILE)


# ============================================================
# 11. LOAD OR BUILD INDEX
# ============================================================

print("\n" + "=" * 60)
print("INDEX")
print("=" * 60)

if (not REBUILD_EMBEDDINGS and EMBEDDINGS_FILE.exists() and CHUNKS_FILE.exists()):
    print("Loading saved embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE)
    
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    print("Loaded embeddings:", embeddings.shape)
    print("Loaded chunks:", len(chunks))
else:
    if REBUILD_EMBEDDINGS:
        print("REBUILD_EMBEDDINGS=True")
        print("Creating document embeddings...")
    else:
        print("Saved index not found.")
        print("Creating document embeddings...")
        
    embeddings = create_document_embeddings(chunks)
    
    print("\n" + "=" * 60)
    print("EMBEDDING CHECK")
    print("=" * 60)
    print("Embedding shape:", embeddings.shape)
    
    assert embeddings.ndim == 2
    assert (embeddings.shape[0] == len(chunks))
    save_index(embeddings, chunks)


# ============================================================
# 12. VALIDATE LOADED / CREATED INDEX
# ============================================================

print("\n" + "=" * 60)
print("INDEX VALIDATION")
print("=" * 60)
print("Embedding shape:", embeddings.shape)
print("Number of chunks:", len(chunks))
print("Number of documents:", len(documents))

if embeddings.ndim != 2:
    raise ValueError("Embeddings must be a 2-dimensional matrix.")

if embeddings.shape[0] != len(chunks):
    raise ValueError(
        "\nMismatch detected!\n"
        f"Embeddings: {embeddings.shape[0]}\n"
        f"Chunks: {len(chunks)}\n\n"
        "embeddings.npy and chunks.json do not "
        "belong to the same chunk set.\n"
        "Delete/rebuild the index."
    )

print("\nIndex is consistent.")


# ============================================================
# 13. NORMALIZE DOCUMENT EMBEDDINGS ONCE
# ============================================================

document_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

# Prevent division by zero.
document_norms[document_norms == 0] = 1.0
normalized_embeddings = (embeddings / document_norms)


# ============================================================
# 14. QUERY EMBEDDING
# ============================================================

def embed_query(query):
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")
        
    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=query)
    query_vector = np.array(response.embeddings[0].values, dtype=np.float32)
    return query_vector


# ============================================================
# 15. RETRIEVE FROM EXISTING VECTOR (DE-DUPLICATED METRIC WINDOWS)
# ============================================================

def retrieve_from_vector(query_vector, k=5):
    """
    Retrieves and runs order-preserving de-duplication across parent
    document IDs to prevent metrics from breaking on duplicate chunks.
    """
    # Request a larger pool (e.g., 20) from vector store to guarantee
    # we have at least k unique Document IDs after de-duplication.
    fetch_k = min(20, len(chunks))
    query_norm = np.linalg.norm(query_vector)
    
    if query_norm == 0:
        raise ValueError("Query embedding has zero norm.")
        
    normalized_query = (query_vector / query_norm)
    
    # Cosine similarity because both vectors are normalized.
    scores = (normalized_embeddings @ normalized_query)
    top_indices = np.argsort(scores)[::-1][:fetch_k]
    
    results = []
    seen_docs = set()
    
    for idx in top_indices:
        doc_id = chunks[idx]["doc_id"]
        
        # Order-preserving de-duplication at document level
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            results.append({
                "chunk_id": int(idx),
                "doc_id": doc_id,
                "score": float(scores[idx]),
                "text": chunks[idx]["text"]
            })
            
        # Stop once we fulfill our requested unique document target limit
        if len(results) == k:
            break
            
    return results


# ============================================================
# 16. RETRIEVE
# ============================================================

def retrieve(query, k=5):
    """
    Embed a new query and retrieve top-k unique documents.
    IMPORTANT:
    Document embeddings are NOT regenerated here.
    Only the query is embedded.
    """
    query_vector = embed_query(query)
    return retrieve_from_vector(query_vector, k=k)


# # ============================================================
# # 17. RETRIEVAL TEST
# # ============================================================

query = "Some central roles in banking today?"
results = retrieve(query, k=5)

print("\n" + "=" * 60)
print("RETRIEVAL RESULTS")
print("=" * 60)
print("Query:", query)

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("-" * 40)
    print("Chunk ID:", result["chunk_id"])
    print("Document:", result["doc_id"])
    print("Score:", round(result["score"], 4))
    print("Text:", result["text"])


# ============================================================
# 18. RETRIEVAL EVALUATION DATA
# ============================================================

evaluation_queries = [
    {"question": "What is credit risk?", "relevant_docs": {"doc03"}},
    {"question": "How is machine learning used in financial risk assessment?", "relevant_docs": {"doc01"}},
    {"question": "What approaches are used for detecting fraudulent transactions?", "relevant_docs": {"doc08"}},
    {"question": "What kind of bias can ml models create in underwriting?", "relevant_docs": {"doc09"}},
    {"question": "According to the documents, who won the 2026 FIFA World Cup?", "out_of_context": True},
    {"question": "Some central roles in banking today?", "relevant_docs": {"doc19"}},
    {"question": "Risks of the most utilised transaction method ?", "relevant_docs": {"doc18"}},{"question": "Most utilised transaction method?", "relevant_docs": {"doc14"}},
    {"question": "Which sector is observing the highest investments?", "relevant_docs": {"doc20"}},
    {"question": "The metrics developed for private credit measure it on which scale - individual, organizational?", "relevant_docs": {"doc17"}},
    {"question": "Fraud numbers with extensive UPI?", "relevant_docs": {"doc14", "doc16"}},
    {"question": "How to remain vigilant against scams?", "relevant_docs": {"doc14"}},
    {"question": "Who can take a home loan in 2026?", "relevant_docs": {"doc12"}},
    {"question": "How to evaluate bank performance?", "relevant_docs": {"doc11"}},
    {"question": "How many accounts can 1 have?", "relevant_docs": {"doc10"}},
    {"question": "What tells loan amount disbursed?", "relevant_docs": {"doc06"}},
    {"question": "Emerging tech to learn for risk prediction?", "relevant_docs": {"doc14", "doc20", "doc07"}},
    {"question": "Big problems machine learning face in Banking?", "relevant_docs": {"doc01"}}
]


# # ============================================================
# # 19. RETRIEVAL EVALUATION (UPDATED COMPREHENSIVE ENGINE)
# # ============================================================

print("\n" + "=" * 60)
print("RETRIEVAL EVALUATION")
print("=" * 60)

hit_at_1 = 0
hit_at_3 = 0
recall_at_3_total = 0.0
recall_at_5_total = 0.0
average_precision_total = 0.0
evaluated_count = 0


def average_precision_at_k(retrieved_results, relevant_docs, k=5):
    """
    Calculate Average Precision@K for one query, strictly capped at 1.0.
    Safely handles both lists of doc_ids and lists of chunk dictionaries.
    """
    if not relevant_docs:
        return 0.0

    # Extract clean document IDs if passed raw chunk result dictionaries
    if retrieved_results and isinstance(retrieved_results[0], dict):
        raw_docs = [res["doc_id"] for res in retrieved_results]
    else:
        raw_docs = list(retrieved_results)

    # 1. Order-preserving de-duplication up to position K
    unique_retrieved_docs = []
    for doc_id in raw_docs:
        if doc_id not in unique_retrieved_docs:
            unique_retrieved_docs.append(doc_id)
        if len(unique_retrieved_docs) == k:
            break

    # 2. Compute Precision at each unique hit position
    hits = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(unique_retrieved_docs, start=1):
        if doc_id in relevant_docs:
            hits += 1
            precision_at_rank = hits / rank
            precision_sum += precision_at_rank

    # 3. Standardize denominator
    denominator = min(len(relevant_docs), k)
    if denominator == 0:
        return 0.0

    return precision_sum / denominator



for item in evaluation_queries:
    question = item["question"]
    
    if item.get("out_of_context", False):
        print("\nQuestion:", question)
        print("Type: Out-of-context / abstention test")
        continue
        
    query_vector = embed_query(question)
    
    # Fetch up to 5 unique documents
    results = retrieve_from_vector(query_vector, k=5)
    retrieved_docs = [result["doc_id"] for result in results]
    relevant_docs = item["relevant_docs"]
    
    # ========================================================
    # HIT@1
    # ========================================================
    top1_hit = False
    if len(retrieved_docs) > 0:
        top1_hit = (retrieved_docs[0] in relevant_docs)
    # ========================================================
    # HIT@3 (Strictly considering the first 3 items)
    # ========================================================
    top3_docs = retrieved_docs[:3]
    top3_hit = bool(set(top3_docs) & relevant_docs)

    # ========================================================
    # RECALL@3
    # ========================================================
    relevant_retrieved_at_3 = set(top3_docs) & relevant_docs
    recall_at_3 = len(relevant_retrieved_at_3) / len(relevant_docs)

    # ========================================================
    # RECALL@5
    # ========================================================
    top5_docs = retrieved_docs[:5]
    relevant_retrieved_at_5 = set(top5_docs) & relevant_docs
    recall_at_5 = len(relevant_retrieved_at_5) / len(relevant_docs)

    # ========================================================
    # MAP@5 (Calculated via AP@5 for this query)
    # ========================================================
    ap_at_5 = average_precision_at_k(retrieved_docs, relevant_docs, k=5)

    # Update totals
    if top1_hit:
        hit_at_1 += 1

    if top3_hit:
        hit_at_3 += 1

    recall_at_3_total += recall_at_3
    recall_at_5_total += recall_at_5
    average_precision_total += ap_at_5
    evaluated_count += 1

    # Printing individual runtime indicators
    print("\nQuestion:", question)
    print("Expected:", sorted(relevant_docs))
    print("Retrieved unique docs:", retrieved_docs)
    print("Hit@1:", top1_hit)
    print("Hit@3:", top3_hit)
    print("Recall@3:", round(recall_at_3, 4))
    print("Recall@5:", round(recall_at_5, 4))
    print("AP@5:", round(ap_at_5, 4))


# ============================================================
# 20. FINAL RETRIEVAL METRICS
# ============================================================

print("\n" + "=" * 60)
print("FINAL RETRIEVAL EVALUATION")
print("=" * 60)

print("Evaluated questions:", evaluated_count)

if evaluated_count > 0:
    print("Hit@1:", round(hit_at_1 / evaluated_count, 4))
    print("Hit@3:", round(hit_at_3 / evaluated_count, 4))
    print("Recall@3:", round(recall_at_3_total / evaluated_count, 4))
    print("Recall@5:", round(recall_at_5_total / evaluated_count, 4))
    print("MAP@5:", round(average_precision_total / evaluated_count, 4))



import time
from tabulate import tabulate  # Print clean comparison tables

# ============================================================
# 21. GENERATION ENGINE (PRODUCTION-GRADE RECOVERY WRAPPER)
# ============================================================

def generate_answer(query, results, max_retries=5):
    """
    Generates text answers using only provided context snippets.
    Dynamically tracks the high-quota model variable string.
    """
    context = "\n\n".join(result["text"] for result in results)

    prompt = f"""
You are a helpful assistant answering questions using the provided retrieved context.

Answer the user's question using ONLY the information in the context.
If the context does not contain enough information, say that the
information is not available in the provided documents.

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

ANSWER:
"""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt
            )
            return response.text
        except ClientError as e:
            error_text = str(e)
            is_quota_hit = (
                "429" in error_text 
                or "RESOURCE_EXHAUSTED" in error_text 
                or "quota exceeded" in error_text.lower()
            )
            
            if is_quota_hit and attempt < max_retries - 1:
                wait_time = get_retry_delay(e, attempt) if 'get_retry_delay' in globals() else 45
                print(f"\n[GENERATION RATE LIMIT] Server pacing active. Pausing loop for {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise e
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError("Generation layer broke permanently due to environment congestion.") from e
            wait_time = min(2 ** attempt + 5, 30)
            print(f"\n[TRANSIENT EXCEPTION] Retrying turn in {wait_time}s...")
            time.sleep(wait_time)

    return "GENERATION FAILED: Quota limits could not clear."


# ============================================================
# 22. SINGLE-CALL LLM GENERATION EVALUATOR (BATCH PASS ENGINE)
# ============================================================

def evaluate_generation_single_call(query, context_list, generated_answer, max_retries=5):
    """
    Evaluates both Faithfulness and Relevance in ONE SINGLE API CALL.
    """
    context_blob = "\n\n".join(context_list)
    
    eval_prompt = f"""
You are an expert AI Auditor evaluating a Retrieval-Augmented Generation (RAG) system.
Analyze the relationship between the USER QUERY, the RETRIEVED CONTEXT, and the GENERATED ANSWER.

Your task is to calculate two distinct metrics:

1. FAITHFULNESS SCORE (0.0 to 1.0):
- Measure if the GENERATED ANSWER stays strictly true to the RETRIEVED CONTEXT without making things up.
- 1.0 means every claim made can be directly verified in the context.
- Note: If the answer correctly abstains by stating "information is not available", the faithfulness score MUST be 1.0.

2. ANSWER RELEVANCE SCORE (0.0 to 1.0):
- Measure how directly the GENERATED ANSWER addresses the specific question asked.
- 1.0 means the response perfectly, clearly, and completely answers the user's core prompt.

Provide a short justification reason for each score.

[INPUT DETAILS]
USER QUERY: {query}
RETRIEVED CONTEXT: {context_blob}
GENERATED ANSWER: {generated_answer}

Return your evaluation strictly in the following JSON format structure:
{{
    "faithfulness_score": float,
    "faithfulness_reason": "string justification",
    "relevance_score": float,
    "relevance_reason": "string justification"
}}
"""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=eval_prompt,
                config={"response_mime_type": "application/json"}
            )
            metrics = json.loads(response.text)
            return metrics
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n[CRITICAL] Evaluator failed after maximum attempts: {e}")
                break
            time.sleep(min(2 ** attempt + 5, 30))

    return {
        "faithfulness_score": 0.0,
        "faithfulness_reason": "FAILED TO EVALUATE: Quota restrictions or transient error.",
        "relevance_score": 0.0,
        "relevance_reason": "FAILED TO EVALUATE: Quota restrictions or transient error."
    }


# ============================================================
# 23. PHASE 1: SEQUENTIAL GENERATION & PERMANENT STORAGE CACHE
# ============================================================
import os

CACHE_FILE = Path("generated_answers_cache.json")
generation_queries = evaluation_queries
cached_evaluation_records = []

# Load existing answers from your storage drive if you ran this before
loaded_cache = {}
if CACHE_FILE.exists():
    print(f"\nFound existing generation cache at {CACHE_FILE}. Loading historical records...")
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        loaded_cache = json.load(f)

print("\n" + "=" * 60)
print("PHASE 1: LIVE BATCH GENERATION SHOWCASE (WITH HARD DRIVE CACHE)")
print("=" * 60)
print(f"Target Queue Workload: {len(generation_queries)} Questions")

PIPELINE_PACING_DELAY = 12.0

for i, query_data in enumerate(generation_queries, 1):
    loop_start = time.perf_counter()
    question = query_data["question"]

    if query_data.get("out_of_context", False):
        print(f"\nQUERY {i}/{len(generation_queries)}: SKIPPED (Out-of-Context Validation Entry)")
        continue

    print("\n" + "-" * 60)
    print(f"PROCESSING RUN {i}/{len(generation_queries)}")
    print("-" * 60)
    print(f"Question: {question}")

    # 1. Pipeline Retrieval (Always get the latest matching chunks)
    try:
        results = retrieve(question, k=5)
    except Exception as e:
        print(f"  [CRITICAL ERROR] Skipping item slot {i} due to search crash: {e}")
        continue

    context_chunks = [res["text"] for res in results]
    retrieved_docs = [res["doc_id"] for res in results]
    print(f"Retrieved Sources: {retrieved_docs}")

    # 2. Check if this exact question was already answered in a previous script run
    if question in loaded_cache:
        print("\n>>> [CACHE HIT] Found saved answer on hard drive. Skipping API Call!")
        answer = loaded_cache[question]["generated_answer"]
    else:
        # Not found in cache -> Call the Gemini API
        print("\n>>> [CACHE MISS] Calling Gemini API to generate new answer...")
        answer = generate_answer(question, results)
        
        # Save it to our local tracking dictionary instantly
        loaded_cache[question] = {
            "context_chunks": context_chunks,
            "generated_answer": answer
        }
        # Commit it to the physical file immediately so data isn't lost if the script crashes halfway
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(loaded_cache, f, ensure_ascii=False, indent=2)

    print(f"\nGENERATED ANSWER:\n{answer}")
    
    # Store in memory array for the Phase 2 evaluation pass
    cached_evaluation_records.append({
        "question": question,
        "context_chunks": context_chunks,
        "generated_answer": answer
    })
    
    # Velocity Pacing (Only sleep if we actually made a live API call)
    if question not in loaded_cache:
        loop_end = time.perf_counter()
        elapsed = loop_end - loop_start
        if elapsed < PIPELINE_PACING_DELAY and i < len(generation_queries):
            time.sleep(PIPELINE_PACING_DELAY - elapsed)

print(f"\n>>> Phase 1 Complete. All solutions permanently committed to {CACHE_FILE}")

# ============================================================
# 24. PHASE 2: BATCH METRICS EVALUATION (RUN AT THE VERY END)
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2: BATCH QUALITY METRICS CONVERSIONS (AUDITING ALL AT ONCE)")
print("=" * 60)
print(f"Processing evaluation logic across {len(cached_evaluation_records)} records...")

cumulative_faithfulness = 0.0
cumulative_relevance = 0.0
successful_evals_count = 0

for idx, record in enumerate(cached_evaluation_records, start=1):
    eval_start = time.perf_counter()
    
    question = record["question"]
    context_chunks = record["context_chunks"]
    answer = record["generated_answer"]

    print(f"\nAUDITING DATA RECORD {idx}/{len(cached_evaluation_records)} -> Question: '{question}'")
    
    if "GENERATION FAILED" in answer:
        print("  [SKIPPED] Cannot audit corrupted text payloads.")
        continue

    # Execute evaluation API call via our fault-tolerant script
    eval_metrics = evaluate_generation_single_call(question, context_chunks, answer)
    
    print(f"  -> Faithfulness Score: {eval_metrics['faithfulness_score']} | Reason: {eval_metrics['faithfulness_reason']}")
    print(f"  -> Relevance Score:    {eval_metrics['relevance_score']} | Reason: {eval_metrics['relevance_reason']}")

    cumulative_faithfulness += eval_metrics["faithfulness_score"]
    cumulative_relevance += eval_metrics["relevance_score"]
    successful_evals_count += 1

    # Apply strict padding between requests to keep the evaluation stream clean
    eval_end = time.perf_counter()
    eval_elapsed = eval_end - eval_start
    if eval_elapsed < PIPELINE_PACING_DELAY and idx < len(cached_evaluation_records):
        time.sleep(PIPELINE_PACING_DELAY - eval_elapsed)


# ============================================================
# 25. FINAL GENERATION EVALUATION SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL CONSOLIDATED GENERATION METRICS REPORT")
print("=" * 60)
print(f"Total Evaluated Test Cases: {successful_evals_count}")

if successful_evals_count > 0:
    final_faithfulness = (cumulative_faithfulness / successful_evals_count) * 100
    final_relevance = (cumulative_relevance / successful_evals_count) * 100
    
    print(f"\n[DASHBOARD ACCURACY METRICS]")
    print(f"  - System-Wide Faithfulness Score (No-Hallucinations): {round(final_faithfulness, 2)}%")
    print(f"  - System-Wide Answer Relevance Score:                {round(final_relevance, 2)}%")
else:
    print("\n[CRITICAL ERROR] No query records completed evaluation successfully.")
