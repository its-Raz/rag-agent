from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Configuration
LLMSTUDIO_API_KEY = os.environ.get('LLMSTUDIO_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
LLM_MODEL = "RPRTHPB-gpt-5-mini"
PINECONE_INDEX_NAME = "ted-talks-rag-full-talks-new"

# RAG Configuration
MAX_TOKENS = 1024
OVERLAP_RATIO = 0.2
TOP_K = 20

# System prompt
SYSTEM_PROMPT = """You are a TED Talk assistant that answers questions strictly and only based on
the TED dataset context provided to you (metadata and transcript passages).
You must not use any external knowledge, the open internet, or information that
is not explicitly contained in the retrieved context. If the answer cannot be
determined from the provided context, respond: "I don't know based on the provided TED data."
Always explain your answer using the given context,
quoting or paraphrasing the relevant transcript or metadata when helpful.
Context consists of transcript chunks from TED talks. Multiple chunks may belong to the SAME talk.

TASK HANDLING RULES:

1. Precise Fact Retrieval
Goal: Locate a single, specific entity or fact based on semantic criteria.
Expected Behavior:
- Identify ONE relevant talk.
- Return the title and speaker(s) only.
- Do NOT include multiple talks.
- Base the answer strictly on retrieved transcript evidence.

2. Multi-Result Topic Listing (Up to 3 Results)
Goal: Return multiple TED talk titles that match a given theme or topic.
Expected Behavior:
- Return EXACTLY 3 DISTINCT talk titles.
- Each title must refer to a different talk.
- Do NOT return multiple chunks from the same talk.
- Do NOT return more or fewer than 3 results.

Example behavior:
Question: "Which TED talks focus on education? Return exactly 3 talk titles."
Correct response format:
- <Title 1>
- <Title 2>
- <Title 3>

3. Key Idea Summary Extraction
Goal: Identify a relevant talk and generate a concise summary of its main idea.
Expected Behavior:
- Choose ONE relevant talk.
- Provide the title.
- Provide a short, concise summary grounded in transcript chunk evidence.
- Do NOT summarize information not supported by the context.

4. Recommendation with Evidence-Based Justification
Goal: Recommend one relevant TED talk and justify the recommendation.
Expected Behavior:
- Recommend EXACTLY ONE talk.
- Provide the title and speaker(s).
- Justify the recommendation using evidence from the retrieved transcript chunks.
- Do NOT use popularity, views, or external facts.

OUTPUT RULES:
- Follow the question's requested format strictly.
- Do not add extra explanations unless requested.
"""

# Initialize FastAPI
app = FastAPI(title="TED Talk RAG Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients (lazy loading)
embedding_client = None
llm_client = None
pinecone_index = None


def get_embedding_client():
    global embedding_client
    if embedding_client is None:
        embedding_client = OpenAIEmbeddings(
            api_key=LLMSTUDIO_API_KEY,
            base_url="https://api.llmod.ai/v1",
            model=EMBEDDING_MODEL
        )
    return embedding_client


def get_llm_client():
    global llm_client
    if llm_client is None:
        llm_client = ChatOpenAI(
            api_key=LLMSTUDIO_API_KEY,
            base_url="https://api.llmod.ai/v1",
            model=LLM_MODEL,
            temperature=1
        )
    return llm_client


def get_pinecone_index():
    global pinecone_index
    if pinecone_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return pinecone_index


def build_augmented_prompt(question, retrieved_matches):
    """Build prompt with context"""
    context_parts = []

    for match in retrieved_matches:
        metadata = match['metadata']
        score = match['score']

        speakers = metadata.get('all_speakers', 'N/A')
        full_text = metadata.get('text', 'N/A')

        context_part = f"""[Relevance Score: {score:.3f}]
Title: {metadata.get('title', 'N/A')}
Speakers: {speakers}

{full_text}"""
        context_parts.append(context_part.strip())

    full_context = "\n\n---\n\n".join(context_parts)

    user_prompt = f"""Context from TED Talks:

{full_context}

Question: {question}

Answer:"""

    return {
        'system': SYSTEM_PROMPT,
        'user': user_prompt
    }


def answer_question(question, top_k=TOP_K):
    """Complete RAG pipeline"""
    # Get clients
    embed_client = get_embedding_client()
    llm = get_llm_client()
    index = get_pinecone_index()

    # 1. Embed query
    query_embedding = embed_client.embed_query(question)

    # 2. Retrieve similar chunks
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    matches = results['matches']

    # 3. Build augmented prompt
    prompts = build_augmented_prompt(question, matches)

    # 4. Generate answer
    messages = [
        ("system", prompts['system']),
        ("human", prompts['user'])
    ]
    response = llm.invoke(messages)
    answer = response.content

    # 5. Format context for response
    context = []
    for match in matches:
        context.append({
            "talk_id": match['metadata'].get('talk_id', ''),
            "title": match['metadata'].get('title', ''),
            "chunk": match['metadata'].get('text', ''),
            "score": match['score']
        })

    return {
        "response": answer,
        "context": context,
        "Augmented_prompt": {
            "System": prompts['system'],
            "User": prompts['user']
        }
    }


# Request/Response models
class QuestionRequest(BaseModel):
    question: str


class StatsResponse(BaseModel):
    chunk_size: int
    overlap_ratio: float
    top_k: int


# API Endpoints
@app.post("/api/prompt")
async def prompt_endpoint(request: QuestionRequest):
    """Query the RAG system with a question"""
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        result = answer_question(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def stats_endpoint():
    """Return RAG system configuration"""
    return StatsResponse(
        chunk_size=MAX_TOKENS,
        overlap_ratio=OVERLAP_RATIO,
        top_k=TOP_K
    )


# Remove or modify the root endpoint
@app.get("/")
async def root():
    """API root - redirect to docs or return simple message"""
    return {"message": "TED Talk RAG API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)