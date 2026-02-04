from dotenv import load_dotenv
load_dotenv()

import os
import time
import io
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import openai
from pinecone import Pinecone, ServerlessSpec
import cohere

from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import PyPDF2

# -----------------------------
# App & Middleware
# -----------------------------

app = FastAPI(title="RAG Application API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Clients
# -----------------------------

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# -----------------------------
# Pinecone Config
# -----------------------------

INDEX_NAME = "rag-app-index"
DIMENSION = 1536

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# -----------------------------
# Chunking Config
# -----------------------------

CHUNK_SIZE = 1000          # tokens
OVERLAP_PERCENT = 0.15
CHUNK_OVERLAP = int(CHUNK_SIZE * OVERLAP_PERCENT)

MAX_CONTEXT_TOKENS = 6000  # safety guard for prompt

# -----------------------------
# Pricing Constants (USD per 1K tokens)
# -----------------------------

PRICING = {
    "embedding": 0.00002,      # text-embedding-3-small
    "gpt4_input": 0.01,        # gpt-4-turbo input
    "gpt4_output": 0.03,       # gpt-4-turbo output
    "rerank_per_search": 0.002 # Cohere rerank approximate
}

# -----------------------------
# Models
# -----------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 15
    rerank_top_n: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    timing: Dict
    token_estimate: Dict

class UploadResponse(BaseModel):
    message: str
    chunks_created: int
    document_id: str
    total_tokens: int

# -----------------------------
# Helpers
# -----------------------------

def get_token_count(text: str) -> int:
    """Count tokens using tiktoken"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to rough approximation
        return len(text) // 4

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF with error handling"""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            raise ValueError("PDF appears to be empty or contains only images")
        
        return text
    except Exception as e:
        raise HTTPException(400, f"Failed to extract PDF text: {str(e)}")

def chunk_text(text: str, metadata: Dict) -> List[Dict]:
    """Chunk text using tiktoken-based splitting for accurate token counting"""
    try:
        # Use tiktoken-based splitter for accurate token counting
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            model_name="gpt-4",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_text(text)

        results = []
        for i, chunk in enumerate(chunks):
            results.append({
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "position": f"{i + 1}/{len(chunks)}",
                    "token_count": get_token_count(chunk)
                }
            })

        return results
    except Exception as e:
        raise HTTPException(500, f"Chunking failed: {str(e)}")

def get_embedding(text: str) -> List[float]:
    """Get OpenAI embedding with error handling"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except openai.RateLimitError:
        raise HTTPException(429, "OpenAI rate limit exceeded. Please try again later.")
    except openai.APIError as e:
        raise HTTPException(503, f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Embedding generation failed: {str(e)}")

def mmr_diversify(docs: List[Dict], max_per_doc: int = 2) -> List[Dict]:
    """Lightweight diversity: limit chunks per document_id"""
    seen = {}
    diversified = []

    for doc in docs:
        doc_id = doc["metadata"].get("document_id", "unknown")
        seen.setdefault(doc_id, 0)
        if seen[doc_id] < max_per_doc:
            diversified.append(doc)
            seen[doc_id] += 1

    return diversified

def rerank_documents(query: str, docs: List[Dict], top_n: int) -> List[Dict]:
    """Rerank documents using Cohere with error handling"""
    if not docs:
        return []

    try:
        rerank = cohere_client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=[d["text"] for d in docs],
            top_n=min(top_n, len(docs))  # Ensure top_n doesn't exceed available docs
        )

        results = []
        for r in rerank.results:
            doc = docs[r.index]
            doc["rerank_score"] = r.relevance_score
            results.append(doc)

        return results
    except Exception as e:
        # If reranking fails, return original docs with warning
        print(f"Reranking failed: {str(e)}, returning original order")
        return docs[:top_n]

def generate_answer(query: str, docs: List[Dict]) -> Dict:
    """Generate answer with comprehensive token tracking"""
    context_blocks = []
    sources = []

    total_context_tokens = 0
    for i, doc in enumerate(docs, 1):
        block = f"[{i}] {doc['text']}"
        block_tokens = get_token_count(block)
        
        if total_context_tokens + block_tokens > MAX_CONTEXT_TOKENS:
            break

        context_blocks.append(block)
        total_context_tokens += block_tokens

        sources.append({
            "id": i,
            "preview": doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"],
            "metadata": doc["metadata"],
            "score": doc.get("rerank_score", doc.get("score", 0))
        })

    context = "\n\n".join(context_blocks)

    system_prompt = """You are a retrieval-augmented assistant that provides accurate, well-cited answers.

Rules:
- Use ONLY the provided context to answer questions
- EVERY claim must be followed by inline citations like [1], [2], or [1,2]
- Multiple related claims can share citations if from the same source
- Do NOT invent citation numbers that don't exist
- If the context doesn't contain enough information, explicitly state what's missing
- Be concise but thorough
- Structure your answer clearly with proper paragraphs
"""

    user_prompt = f"""Context:
{context}

Question:
{query}

Provide a well-structured answer with inline citations [1], [2], etc. for every claim you make.
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            temperature=0.2,
            max_tokens=800
        )

        usage = response.usage

        return {
            "answer": response.choices[0].message.content,
            "tokens": {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "total": usage.total_tokens
            },
            "sources": sources
        }
    except openai.RateLimitError:
        raise HTTPException(429, "OpenAI rate limit exceeded. Please try again later.")
    except openai.APIError as e:
        raise HTTPException(503, f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Answer generation failed: {str(e)}")

# -----------------------------
# Routes
# -----------------------------

@app.get("/")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "RAG API",
        "version": "1.0.0"
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    title: str = Form("Untitled"),
    source: str = Form("uploaded")
):
    """Upload and process a document (PDF, TXT, or plain text)"""
    
    if not file and not text:
        raise HTTPException(400, "Must provide either a file or text content")

    # Extract content based on input type
    try:
        if file:
            data = await file.read()
            filename = file.filename.lower()
            
            if filename.endswith(".pdf"):
                content = extract_text_from_pdf(data)
            elif filename.endswith((".txt", ".md")):
                try:
                    content = data.decode("utf-8")
                except UnicodeDecodeError:
                    raise HTTPException(400, "File must be UTF-8 encoded")
            else:
                raise HTTPException(400, "Unsupported file type. Supported: .pdf, .txt, .md")
            
            doc_id = f"{file.filename}_{int(time.time())}"
        else:
            content = text.strip()
            if not content:
                raise HTTPException(400, "Text content cannot be empty")
            doc_id = f"text_{int(time.time())}"

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to process file: {str(e)}")

    # Prepare metadata
    metadata = {
        "title": title,
        "source": source,
        "document_id": doc_id,
        "uploaded_at": time.time()
    }

    # Chunk the document
    chunks = chunk_text(content, metadata)
    
    if not chunks:
        raise HTTPException(400, "Document produced no valid chunks")

    # Generate embeddings and prepare vectors
    vectors = []
    total_tokens = 0
    
    try:
        for c in chunks:
            embedding = get_embedding(c["text"])
            total_tokens += c["metadata"]["token_count"]
            
            vectors.append({
                "id": f"{doc_id}_{c['metadata']['chunk_index']}",
                "values": embedding,
                "metadata": {**c["metadata"], "text": c["text"]}
            })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate embeddings: {str(e)}")

    # Upsert to Pinecone in batches
    try:
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
    except Exception as e:
        raise HTTPException(500, f"Failed to store vectors in Pinecone: {str(e)}")

    return UploadResponse(
        message=f"Successfully indexed document with {len(chunks)} chunks",
        chunks_created=len(chunks),
        document_id=doc_id,
        total_tokens=total_tokens
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """Query the knowledge base with retrieval, reranking, and answer generation"""
    
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    
    timing = {}
    
    # Step 1: Generate query embedding
    t0 = time.time()
    try:
        query_vec = get_embedding(req.query)
        query_tokens = get_token_count(req.query)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate query embedding: {str(e)}")
    timing["embedding"] = round(time.time() - t0, 3)

    # Step 2: Retrieve from Pinecone
    t0 = time.time()
    try:
        results = index.query(
            vector=query_vec,
            top_k=req.top_k,
            include_metadata=True
        )
    except Exception as e:
        raise HTTPException(500, f"Vector search failed: {str(e)}")
    timing["retrieval"] = round(time.time() - t0, 3)

    # Extract documents
    docs = []
    for m in results.matches:
        if "text" in m.metadata:
            docs.append({
                "text": m.metadata["text"],
                "metadata": {k: v for k, v in m.metadata.items() if k != "text"},
                "score": m.score
            })

    # Handle no results
    if not docs:
        return QueryResponse(
            answer="I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing or upload relevant documents first.",
            sources=[],
            timing=timing,
            token_estimate={
                "embedding_tokens": query_tokens,
                "llm_input_tokens": 0,
                "llm_output_tokens": 0,
                "total_tokens": query_tokens,
                "costs": {
                    "embedding_usd": round((query_tokens / 1000) * PRICING["embedding"], 6),
                    "llm_usd": 0.0,
                    "rerank_usd": 0.0,
                    "total_usd": round((query_tokens / 1000) * PRICING["embedding"], 6)
                }
            }
        )

    # Step 3: Apply MMR diversity
    docs = mmr_diversify(docs, max_per_doc=2)

    # Step 4: Rerank
    t0 = time.time()
    docs = rerank_documents(req.query, docs, req.rerank_top_n)
    timing["reranking"] = round(time.time() - t0, 3)

    # Step 5: Generate answer
    t0 = time.time()
    try:
        result = generate_answer(req.query, docs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Answer generation failed: {str(e)}")
    timing["generation"] = round(time.time() - t0, 3)
    timing["total"] = round(sum(timing.values()), 3)

    # Calculate comprehensive costs
    embedding_cost = (query_tokens / 1000) * PRICING["embedding"]
    llm_input_cost = (result["tokens"]["input"] / 1000) * PRICING["gpt4_input"]
    llm_output_cost = (result["tokens"]["output"] / 1000) * PRICING["gpt4_output"]
    rerank_cost = PRICING["rerank_per_search"]
    total_cost = embedding_cost + llm_input_cost + llm_output_cost + rerank_cost

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        timing=timing,
        token_estimate={
            "embedding_tokens": query_tokens,
            "llm_input_tokens": result["tokens"]["input"],
            "llm_output_tokens": result["tokens"]["output"],
            "total_tokens": query_tokens + result["tokens"]["total"],
            "costs": {
                "embedding_usd": round(embedding_cost, 6),
                "llm_input_usd": round(llm_input_cost, 6),
                "llm_output_usd": round(llm_output_cost, 6),
                "rerank_usd": round(rerank_cost, 6),
                "total_usd": round(total_cost, 6)
            }
        }
    )

@app.delete("/clear")
async def clear_index():
    """Clear all vectors from the index"""
    try:
        index.delete(delete_all=True)
        return {"message": "Index cleared successfully"}
    except Exception as e:
        raise HTTPException(500, f"Failed to clear index: {str(e)}")

@app.get("/stats")
async def stats():
    """Get index statistics"""
    try:
        s = index.describe_index_stats()
        return {
            "total_vectors": s.total_vector_count,
            "dimension": s.dimension,
            "index_fullness": s.index_fullness,
            "namespaces": s.namespaces
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get stats: {str(e)}")

@app.get("/health")
async def detailed_health():
    """Detailed health check with service status"""
    health_status = {
        "api": "ok",
        "pinecone": "unknown",
        "openai": "unknown",
        "cohere": "unknown"
    }
    
    # Check Pinecone
    try:
        index.describe_index_stats()
        health_status["pinecone"] = "ok"
    except:
        health_status["pinecone"] = "error"
    
    # Check OpenAI (lightweight)
    try:
        openai_client.models.list()
        health_status["openai"] = "ok"
    except:
        health_status["openai"] = "error"
    
    # Cohere check is expensive, skip for now
    health_status["cohere"] = "not_checked"
    
    return health_status

# -----------------------------
# Entry
# -----------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)