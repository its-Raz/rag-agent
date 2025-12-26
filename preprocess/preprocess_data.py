<<<<<<< HEAD
import pandas as pd
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# ============================================================================
# CONFIGURATION
# ============================================================================

LLMSTUDIO_API_KEY = "your-llmstudio-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"

EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
EMBEDDING_DIM = 1536

# RAG Hyperparameters
MAX_TOKENS = 1024
OVERLAP_RATIO = 0.2
TOP_K = 20

PINECONE_INDEX_NAME = "ted-talks-rag-full-talks"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


# ============================================================================
# 1. LOAD DATA
# ============================================================================

def load_data(filepath='ted_talks_en.csv'):
    """Load TED talks dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ File loaded successfully. Shape: {df.shape}")

        columns_to_keep = [
            'talk_id', 'title', 'all_speakers', 'about_speakers', 'occupations',
            'topics', 'related_talks', 'description', 'transcript',
        ]
        df = df[columns_to_keep]
        df = df.dropna(subset=['transcript'])
        print(f"After cleaning: {len(df)} talks")

        return df
    except FileNotFoundError:
        print("File not found. Please upload 'ted_talks_en.csv'")
        return None


# ============================================================================
# 2. IMPROVED CHUNKING WITH TEXT_SPLITTER
# ============================================================================

def chunk_text_with_splitter(talk, max_tokens=1024, overlap_ratio=0.2):
    """
    Chunk text using LangChain's RecursiveCharacterTextSplitter
    with token-based splitting for precise control
    """
    text = talk['transcript']
    title = talk['title']
    topics = talk.get('topics', '')
    description = talk.get('description', '')

    if not text or not text.strip():
        return []

    # Prepare metadata
    metadata = {
        'talk_id': talk.get('talk_id', ''),
        'title': talk.get('title', ''),
        'all_speakers': talk.get('all_speakers', ''),
        'about_speakers': talk.get('about_speakers', ''),
        'occupations': talk.get('occupations', ''),
        'topics': topics,
        'related_talks': talk.get('related_talks', ''),
        'description': description
    }

    # Calculate metadata token overhead
    encoding = tiktoken.get_encoding("cl100k_base")
    metadata_prefix = f"Topics: {topics}\nDescription: {description}\n\nTranscript: "
    metadata_tokens = len(encoding.encode(metadata_prefix))

    # Adjust chunk size to account for metadata
    effective_chunk_size = max_tokens - metadata_tokens
    if effective_chunk_size < 100:
        effective_chunk_size = 100  # Minimum viable chunk size

    # Calculate overlap in tokens
    overlap_tokens = int(effective_chunk_size * overlap_ratio)

    # Initialize text splitter with token-based splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=overlap_tokens,
        length_function=lambda t: len(encoding.encode(t)),
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )

    # Split the transcript
    chunks = text_splitter.split_text(text)

    # Create chunk objects with metadata
    chunk_objects = []
    for i, chunk in enumerate(chunks):
        # Combine metadata with chunk
        full_text = (
            f"Topics: {topics}\n"
            f"Description: {description}\n\n"
            f"Transcript: {chunk}"
        )

        # Verify token count
        final_tokens = len(encoding.encode(full_text))
        if final_tokens > max_tokens:
            print(f"⚠️ WARNING: Chunk exceeds {max_tokens} tokens ({final_tokens})")

        chunk_obj = {"text": full_text}
        chunk_obj.update(metadata)
        chunk_obj['chunk_index'] = i
        chunk_objects.append(chunk_obj)

    return chunk_objects


# ============================================================================
# 3. EMBEDDING
# ============================================================================

class LLMStudioEmbedding:
    """Embedding using OpenAI-compatible API"""

    def __init__(self, api_key, model=EMBEDDING_MODEL):
        self.embedding_model = OpenAIEmbeddings(
            api_key=api_key,
            base_url="https://api.llmod.ai/v1",
            model=model
        )

    def embed(self, texts, batch_size=50):
        """Embed texts using llmstudio API"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"  Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}...")

            try:
                batch_embeddings = self.embedding_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"❌ Error embedding batch: {e}")
                raise

        return np.array(all_embeddings)


# ============================================================================
# 4. PINECONE VECTOR STORE
# ============================================================================

class PineconeVectorStore:
    """Pinecone vector store for RAG"""

    def __init__(self, api_key, index_name, dimension=EMBEDDING_DIM):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.index = None

    def create_index(self):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Creating index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            print("✓ Index created")
        else:
            print(f"✓ Index '{self.index_name}' already exists")

        self.index = self.pc.Index(self.index_name)

    def upsert_vectors(self, embeddings, chunks, batch_size=100):
        """Upload vectors to Pinecone with full text storage"""
        vectors = []

        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            vector_id = f"chunk_{i}"

            full_text = str(chunk.get('text', ''))

            max_text_bytes = 35000
            text_to_store = full_text

            text_bytes = text_to_store.encode('utf-8')
            if len(text_bytes) > max_text_bytes:
                text_to_store = text_to_store[:max_text_bytes // 2]
                print(f"⚠️ Truncated text for chunk {i} (too large for Pinecone)")

            metadata = {
                'talk_id': str(chunk.get('talk_id', '')),
                'title': str(chunk.get('title', '')),
                'all_speakers': str(chunk.get('all_speakers', ''))[:500],
                'text': text_to_store,
            }

            vectors.append({
                'id': vector_id,
                'values': embedding.tolist(),
                'metadata': metadata
            })

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            print(f"✓ Uploaded batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}")

        print(f"✓ Total vectors uploaded: {len(vectors)}")

    def search(self, query_embedding, top_k=5):
        """Search for similar vectors"""
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return results['matches']


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TED TALKS RAG SYSTEM - DATA PROCESSING")
    print("=" * 70)

    # Load data
    print("\n1. LOADING DATA...")
    df = load_data('ted_talks_en.csv')

    # Process all talks or subset for testing
    # df = df.head(100)  # Uncomment for testing with subset
    print(f"✓ Processing {len(df)} talks")

    # Chunk talks with new splitter
    print("\n2. CHUNKING WITH TEXT_SPLITTER...")
    all_chunks = []
    for idx, talk in df.iterrows():
        chunks = chunk_text_with_splitter(talk, MAX_TOKENS, OVERLAP_RATIO)
        all_chunks.extend(chunks)
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} talks...")

    print(f"✓ Created {len(all_chunks)} chunks")

    # Initialize embedding client
    print("\n3. INITIALIZING EMBEDDING CLIENT...")
    embedding_client = LLMStudioEmbedding(LLMSTUDIO_API_KEY)

    # Embed chunks
    print("\n4. EMBEDDING CHUNKS...")
    chunk_texts = [c['text'] for c in all_chunks]
    embeddings = embedding_client.embed(chunk_texts, batch_size=50)
    print(f"✓ Created embeddings shape: {embeddings.shape}")

    # Initialize Pinecone and upload
    print("\n5. UPLOADING TO PINECONE...")
    vector_store = PineconeVectorStore(PINECONE_API_KEY, PINECONE_INDEX_NAME)
    vector_store.create_index()
    vector_store.upsert_vectors(embeddings, all_chunks, batch_size=100)

    print("\n" + "=" * 70)
    print("✅ DATA PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Configuration:")
    print(f"  - Chunk size: {MAX_TOKENS} tokens")
    print(f"  - Overlap ratio: {OVERLAP_RATIO}")
=======
import pandas as pd
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# ============================================================================
# CONFIGURATION
# ============================================================================

LLMSTUDIO_API_KEY = "your-llmstudio-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"

EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
EMBEDDING_DIM = 1536

# RAG Hyperparameters
MAX_TOKENS = 1024
OVERLAP_RATIO = 0.2
TOP_K = 20

PINECONE_INDEX_NAME = "ted-talks-rag-full-talks"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


# ============================================================================
# 1. LOAD DATA
# ============================================================================

def load_data(filepath='ted_talks_en.csv'):
    """Load TED talks dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ File loaded successfully. Shape: {df.shape}")

        columns_to_keep = [
            'talk_id', 'title', 'all_speakers', 'about_speakers', 'occupations',
            'topics', 'related_talks', 'description', 'transcript',
        ]
        df = df[columns_to_keep]
        df = df.dropna(subset=['transcript'])
        print(f"After cleaning: {len(df)} talks")

        return df
    except FileNotFoundError:
        print("File not found. Please upload 'ted_talks_en.csv'")
        return None


# ============================================================================
# 2. IMPROVED CHUNKING WITH TEXT_SPLITTER
# ============================================================================

def chunk_text_with_splitter(talk, max_tokens=1024, overlap_ratio=0.2):
    """
    Chunk text using LangChain's RecursiveCharacterTextSplitter
    with token-based splitting for precise control
    """
    text = talk['transcript']
    title = talk['title']
    topics = talk.get('topics', '')
    description = talk.get('description', '')

    if not text or not text.strip():
        return []

    # Prepare metadata
    metadata = {
        'talk_id': talk.get('talk_id', ''),
        'title': talk.get('title', ''),
        'all_speakers': talk.get('all_speakers', ''),
        'about_speakers': talk.get('about_speakers', ''),
        'occupations': talk.get('occupations', ''),
        'topics': topics,
        'related_talks': talk.get('related_talks', ''),
        'description': description
    }

    # Calculate metadata token overhead
    encoding = tiktoken.get_encoding("cl100k_base")
    metadata_prefix = f"Topics: {topics}\nDescription: {description}\n\nTranscript: "
    metadata_tokens = len(encoding.encode(metadata_prefix))

    # Adjust chunk size to account for metadata
    effective_chunk_size = max_tokens - metadata_tokens
    if effective_chunk_size < 100:
        effective_chunk_size = 100  # Minimum viable chunk size

    # Calculate overlap in tokens
    overlap_tokens = int(effective_chunk_size * overlap_ratio)

    # Initialize text splitter with token-based splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=overlap_tokens,
        length_function=lambda t: len(encoding.encode(t)),
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )

    # Split the transcript
    chunks = text_splitter.split_text(text)

    # Create chunk objects with metadata
    chunk_objects = []
    for i, chunk in enumerate(chunks):
        # Combine metadata with chunk
        full_text = (
            f"Topics: {topics}\n"
            f"Description: {description}\n\n"
            f"Transcript: {chunk}"
        )

        # Verify token count
        final_tokens = len(encoding.encode(full_text))
        if final_tokens > max_tokens:
            print(f"⚠️ WARNING: Chunk exceeds {max_tokens} tokens ({final_tokens})")

        chunk_obj = {"text": full_text}
        chunk_obj.update(metadata)
        chunk_obj['chunk_index'] = i
        chunk_objects.append(chunk_obj)

    return chunk_objects


# ============================================================================
# 3. EMBEDDING
# ============================================================================

class LLMStudioEmbedding:
    """Embedding using OpenAI-compatible API"""

    def __init__(self, api_key, model=EMBEDDING_MODEL):
        self.embedding_model = OpenAIEmbeddings(
            api_key=api_key,
            base_url="https://api.llmod.ai/v1",
            model=model
        )

    def embed(self, texts, batch_size=50):
        """Embed texts using llmstudio API"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"  Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}...")

            try:
                batch_embeddings = self.embedding_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"❌ Error embedding batch: {e}")
                raise

        return np.array(all_embeddings)


# ============================================================================
# 4. PINECONE VECTOR STORE
# ============================================================================

class PineconeVectorStore:
    """Pinecone vector store for RAG"""

    def __init__(self, api_key, index_name, dimension=EMBEDDING_DIM):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.index = None

    def create_index(self):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Creating index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            print("✓ Index created")
        else:
            print(f"✓ Index '{self.index_name}' already exists")

        self.index = self.pc.Index(self.index_name)

    def upsert_vectors(self, embeddings, chunks, batch_size=100):
        """Upload vectors to Pinecone with full text storage"""
        vectors = []

        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            vector_id = f"chunk_{i}"

            full_text = str(chunk.get('text', ''))

            max_text_bytes = 35000
            text_to_store = full_text

            text_bytes = text_to_store.encode('utf-8')
            if len(text_bytes) > max_text_bytes:
                text_to_store = text_to_store[:max_text_bytes // 2]
                print(f"⚠️ Truncated text for chunk {i} (too large for Pinecone)")

            metadata = {
                'talk_id': str(chunk.get('talk_id', '')),
                'title': str(chunk.get('title', '')),
                'all_speakers': str(chunk.get('all_speakers', ''))[:500],
                'text': text_to_store,
            }

            vectors.append({
                'id': vector_id,
                'values': embedding.tolist(),
                'metadata': metadata
            })

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            print(f"✓ Uploaded batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}")

        print(f"✓ Total vectors uploaded: {len(vectors)}")

    def search(self, query_embedding, top_k=5):
        """Search for similar vectors"""
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return results['matches']


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TED TALKS RAG SYSTEM - DATA PROCESSING")
    print("=" * 70)

    # Load data
    print("\n1. LOADING DATA...")
    df = load_data('ted_talks_en.csv')

    # Process all talks or subset for testing
    # df = df.head(100)  # Uncomment for testing with subset
    print(f"✓ Processing {len(df)} talks")

    # Chunk talks with new splitter
    print("\n2. CHUNKING WITH TEXT_SPLITTER...")
    all_chunks = []
    for idx, talk in df.iterrows():
        chunks = chunk_text_with_splitter(talk, MAX_TOKENS, OVERLAP_RATIO)
        all_chunks.extend(chunks)
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} talks...")

    print(f"✓ Created {len(all_chunks)} chunks")

    # Initialize embedding client
    print("\n3. INITIALIZING EMBEDDING CLIENT...")
    embedding_client = LLMStudioEmbedding(LLMSTUDIO_API_KEY)

    # Embed chunks
    print("\n4. EMBEDDING CHUNKS...")
    chunk_texts = [c['text'] for c in all_chunks]
    embeddings = embedding_client.embed(chunk_texts, batch_size=50)
    print(f"✓ Created embeddings shape: {embeddings.shape}")

    # Initialize Pinecone and upload
    print("\n5. UPLOADING TO PINECONE...")
    vector_store = PineconeVectorStore(PINECONE_API_KEY, PINECONE_INDEX_NAME)
    vector_store.create_index()
    vector_store.upsert_vectors(embeddings, all_chunks, batch_size=100)

    print("\n" + "=" * 70)
    print("✅ DATA PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Configuration:")
    print(f"  - Chunk size: {MAX_TOKENS} tokens")
    print(f"  - Overlap ratio: {OVERLAP_RATIO}")
>>>>>>> 3bdfbd1 (Fix: use embed_query for user questions in RAG retrieval)
    print(f"  - Top-K: {TOP_K}")