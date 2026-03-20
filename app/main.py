"""
User Memory Service - Per-user Hash Sphere Memory Panel
Provides isolated memory storage with 3D visualization coordinates.

This service can either:
1. Use its own local SQLite storage with hash-based coordinates
2. Proxy to the main memory_service for real ResonanceHasher PCA coordinates
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx

from .models import (
    MemoryNode, MemoryEdge, MemoryCluster,
    MemoryType, ClusterType, LayerType,
    EmbedRequest, EmbedResponse,
    RetrieveRequest, RetrieveResponse,
    ClusterCreateRequest, UniverseStateResponse
)
from .storage import get_user_storage, UserMemoryStorage
from .embedding import embedding_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("user_memory")

# Configuration
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "50000"))
MAX_MEMORIES_DEVELOPER = int(os.getenv("MAX_MEMORIES_DEVELOPER", "100"))
MAX_MEMORIES_PLUS = int(os.getenv("MAX_MEMORIES_PLUS", "10000"))
MAX_MEMORIES_ENTERPRISE = int(os.getenv("MAX_MEMORIES_ENTERPRISE", "100000"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("User Memory Service starting...")
    yield
    logger.info("User Memory Service shutting down...")


app = FastAPI(
    title="User Memory Service",
    description="Per-user Hash Sphere Memory Panel - Store, visualize, and retrieve memories in 3D space",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend directory
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# Serve frontend at root
@app.get("/")
async def serve_frontend():
    """Serve the 3D Memory Panel frontend."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/visualizer/semantic-space")
async def serve_semantic_space_visualizer():
    """Serve the 3D semantic space visualizer."""
    visualizer_path = FRONTEND_DIR / "semantic_space_visualizer.html"
    if not visualizer_path.exists():
        raise HTTPException(status_code=404, detail="Semantic space visualizer not found")
    return FileResponse(visualizer_path)


# Dependency to get user context from headers
async def get_user_context(
    x_user_id: str = Header(None, alias="X-User-Id"),
    x_org_id: str = Header(None, alias="X-Org-Id"),
    authorization: str = Header(None)
):
    """Extract user context from request headers."""
    # For demo/development, use default values if not provided
    user_id = x_user_id or "demo-user"
    org_id = x_org_id or "demo-org"
    return {"user_id": user_id, "org_id": org_id}


def get_storage(ctx: dict = Depends(get_user_context)) -> UserMemoryStorage:
    """Get storage instance for current user."""
    return get_user_storage(ctx["user_id"], ctx["org_id"])


# ============================================================================
# HEALTH & STATUS
# ============================================================================

@app.get("/health")
async def health():
    return {"status": "ok", "service": "user_memory"}


@app.get("/status")
async def status():
    return {
        "service": "user_memory",
        "version": "1.0.0",
        "embedding_model": embedding_engine.model,
        "embedding_dim": embedding_engine.dim
    }


# ============================================================================
# MEMORY EMBEDDING API
# ============================================================================

@app.post("/memories/embed", response_model=EmbedResponse)
async def embed_memory(
    request: EmbedRequest,
    storage: UserMemoryStorage = Depends(get_storage),
    ctx: dict = Depends(get_user_context)
):
    """
    Embed and store a new memory in the user's Hash Sphere.
    The memory will be assigned 3D coordinates based on its semantic content.
    """
    start_time = time.time()
    
    # Validate content length
    if len(request.content) > MAX_CONTENT_LENGTH:
        raise HTTPException(
            status_code=400, 
            detail=f"Content too long. Max {MAX_CONTENT_LENGTH} characters."
        )
    
    # Check memory limits (would check subscription tier in production)
    stats = storage.get_stats()
    if stats["total_memories"] >= MAX_MEMORIES_FREE:
        raise HTTPException(
            status_code=402,
            detail="Memory limit reached. Upgrade to Pro for more storage."
        )
    
    # Generate embedding
    embedding = await embedding_engine.embed_text(request.content)
    
    # Get cluster center for coordinate calculation
    clusters = storage.get_clusters()
    cluster_center = (0, 0, 0)
    for c in clusters:
        if c.name == request.cluster:
            cluster_center = (c.center_x, c.center_y, c.center_z)
            break
    
    # Convert embedding to 3D coordinates
    x, y, z = embedding_engine.embedding_to_coordinates(embedding, cluster_center)
    
    # Create memory node
    content_hash = embedding_engine.content_hash(request.content)
    memory = MemoryNode(
        user_id=ctx["user_id"],
        org_id=ctx["org_id"],
        content=request.content,
        content_hash=content_hash,
        memory_type=request.memory_type,
        embedding=embedding,
        x=x,
        y=y,
        z=z,
        cluster=request.cluster,
        layer=request.layer,
        title=request.title,
        tags=request.tags,
        source=request.source,
        metadata=request.metadata
    )
    
    # Store memory
    if not storage.add_memory(memory):
        raise HTTPException(status_code=500, detail="Failed to store memory")
    
    # Find and create edges to similar memories
    all_embeddings = storage.get_all_embeddings()
    if len(all_embeddings) > 1:
        similar = embedding_engine.find_similar(
            embedding, 
            [(id_, emb) for id_, emb in all_embeddings if id_ != memory.id],
            top_k=5,
            min_similarity=0.7
        )
        for similar_id, similarity in similar:
            edge = MemoryEdge(
                source_id=memory.id,
                target_id=similar_id,
                relationship_type="similar",
                similarity=similarity
            )
            storage.add_edge(edge)
    
    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"Embedded memory {memory.id} in {duration_ms:.2f}ms")
    
    return EmbedResponse(
        memory_id=memory.id,
        content_hash=content_hash,
        coordinates={"x": x, "y": y, "z": z},
        cluster=request.cluster,
        layer=request.layer.value,
        embedding_dim=len(embedding),
        created_at=memory.created_at.isoformat()
    )


@app.post("/memories/retrieve", response_model=RetrieveResponse)
async def retrieve_memories(
    request: RetrieveRequest,
    storage: UserMemoryStorage = Depends(get_storage)
):
    """
    Retrieve memories similar to a query.
    Uses semantic search to find the most relevant memories.
    """
    start_time = time.time()
    
    # Generate query embedding
    query_embedding = await embedding_engine.embed_text(request.query)
    
    # Get all embeddings
    all_embeddings = storage.get_all_embeddings()
    
    # Find similar
    similar = embedding_engine.find_similar(
        query_embedding,
        all_embeddings,
        top_k=request.top_k,
        min_similarity=request.min_similarity
    )
    
    # Fetch full memory data for results
    results = []
    for memory_id, similarity in similar:
        memory = storage.get_memory(memory_id)
        if memory:
            # Filter by cluster/layer if specified
            if request.cluster and memory.cluster != request.cluster:
                continue
            if request.layer and memory.layer != request.layer:
                continue
            
            # Update access stats
            storage.update_access(memory_id)
            
            result = {
                "id": memory.id,
                "content": memory.content,
                "similarity": round(similarity, 4),
                "coordinates": {"x": memory.x, "y": memory.y, "z": memory.z},
                "cluster": memory.cluster,
                "layer": memory.layer.value,
                "title": memory.title,
                "tags": memory.tags,
                "created_at": memory.created_at.isoformat()
            }
            
            if request.include_embeddings:
                result["embedding"] = memory.embedding
            
            results.append(result)
    
    duration_ms = (time.time() - start_time) * 1000
    
    return RetrieveResponse(
        query=request.query,
        results=results,
        total_searched=len(all_embeddings),
        retrieval_time_ms=round(duration_ms, 2)
    )


# ============================================================================
# MEMORY CRUD
# ============================================================================

@app.get("/memories")
async def list_memories(
    cluster: Optional[str] = None,
    layer: Optional[str] = None,
    limit: int = Query(100, le=1000),
    storage: UserMemoryStorage = Depends(get_storage)
):
    """List all memories, optionally filtered by cluster or layer."""
    layer_enum = LayerType(layer) if layer else None
    memories = storage.get_all_memories(cluster=cluster, layer=layer_enum, limit=limit)
    
    return {
        "memories": [
            {
                "id": m.id,
                "content": m.content[:200] + "..." if len(m.content) > 200 else m.content,
                "title": m.title,
                "coordinates": {"x": m.x, "y": m.y, "z": m.z},
                "cluster": m.cluster,
                "layer": m.layer.value,
                "tags": m.tags,
                "importance": m.importance,
                "access_count": m.access_count,
                "created_at": m.created_at.isoformat()
            }
            for m in memories
        ],
        "total": len(memories)
    }


@app.get("/memories/{memory_id}")
async def get_memory(
    memory_id: str,
    storage: UserMemoryStorage = Depends(get_storage)
):
    """Get a specific memory by ID."""
    memory = storage.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    storage.update_access(memory_id)
    
    return {
        "id": memory.id,
        "content": memory.content,
        "content_hash": memory.content_hash,
        "memory_type": memory.memory_type.value,
        "coordinates": {"x": memory.x, "y": memory.y, "z": memory.z},
        "cluster": memory.cluster,
        "layer": memory.layer.value,
        "title": memory.title,
        "tags": memory.tags,
        "source": memory.source,
        "metadata": memory.metadata,
        "importance": memory.importance,
        "access_count": memory.access_count,
        "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
        "created_at": memory.created_at.isoformat()
    }


@app.patch("/memories/{memory_id}/archive")
async def archive_memory(
    memory_id: str,
    storage: UserMemoryStorage = Depends(get_storage)
):
    """
    Archive a memory - move to archive layer.
    
    IMPORTANT: Memories can NEVER be deleted because they are interconnected
    in the hash universe. Deleting one would break the integrity of the entire
    system. Instead, memories can only be archived (moved to archive layer).
    """
    memory = storage.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Update layer to archive
    if not storage.update_layer(memory_id, "archive"):
        raise HTTPException(status_code=500, detail="Failed to archive memory")
    
    return {
        "status": "archived",
        "memory_id": memory_id,
        "message": "Memory moved to archive layer. Memories cannot be deleted as they are interconnected in the hash universe."
    }


@app.patch("/memories/{memory_id}/restore")
async def restore_memory(
    memory_id: str,
    layer: str = "active",
    storage: UserMemoryStorage = Depends(get_storage)
):
    """Restore an archived memory to active or another layer."""
    memory = storage.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    valid_layers = ["core", "active", "peripheral"]
    if layer not in valid_layers:
        raise HTTPException(status_code=400, detail=f"Invalid layer. Must be one of: {valid_layers}")
    
    if not storage.update_layer(memory_id, layer):
        raise HTTPException(status_code=500, detail="Failed to restore memory")
    
    return {"status": "restored", "memory_id": memory_id, "layer": layer}


# ============================================================================
# CLUSTERS
# ============================================================================

@app.post("/clusters")
async def create_cluster(
    request: ClusterCreateRequest,
    storage: UserMemoryStorage = Depends(get_storage),
    ctx: dict = Depends(get_user_context)
):
    """Create a new memory cluster/universe."""
    # Calculate cluster center position (spread clusters apart)
    existing = storage.get_clusters()
    center_x = len(existing) * 150  # Spread horizontally
    center_y = 0
    center_z = 0
    
    cluster = MemoryCluster(
        user_id=ctx["user_id"],
        org_id=ctx["org_id"],
        name=request.name,
        description=request.description,
        cluster_type=request.cluster_type,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        color=request.color
    )
    
    if not storage.add_cluster(cluster):
        raise HTTPException(status_code=400, detail="Cluster already exists")
    
    return {
        "id": cluster.id,
        "name": cluster.name,
        "center": {"x": center_x, "y": center_y, "z": center_z},
        "color": cluster.color
    }


@app.get("/clusters")
async def list_clusters(storage: UserMemoryStorage = Depends(get_storage)):
    """List all clusters."""
    clusters = storage.get_clusters()
    return {
        "clusters": [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "cluster_type": c.cluster_type.value,
                "center": {"x": c.center_x, "y": c.center_y, "z": c.center_z},
                "radius": c.radius,
                "color": c.color,
                "memory_count": c.memory_count
            }
            for c in clusters
        ]
    }


# ============================================================================
# UNIVERSE VISUALIZATION
# ============================================================================

# Memory service URL for proxying to real ResonanceHasher coordinates
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://memory_service:8000")


@app.get("/universe")
async def get_universe_state(
    source: str = Query("local", description="'local' for local storage, 'memory_service' for real ResonanceHasher coordinates"),
    storage: UserMemoryStorage = Depends(get_storage),
    ctx: dict = Depends(get_user_context)
):
    """
    Get the complete universe state for 3D visualization.
    
    Args:
        source: 'local' uses local SQLite storage with hash-based coordinates.
                'memory_service' proxies to the real memory_service with 
                ResonanceHasher PCA-based coordinates.
    """
    if source == "memory_service" and ctx["user_id"] != "demo-user":
        # Proxy to real memory_service for proper ResonanceHasher coordinates
        # Only works with real UUID user_id/org_id from authenticated requests
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{MEMORY_SERVICE_URL}/rag/universe",
                    headers={
                        "X-User-Id": ctx["user_id"],
                        "X-Org-Id": ctx["org_id"]
                    },
                    timeout=30.0
                )
                if resp.status_code == 200:
                    return resp.json()
                else:
                    logger.warning(f"Memory service returned {resp.status_code}, falling back to local")
        except Exception as e:
            logger.warning(f"Failed to reach memory_service: {e}, falling back to local")
    
    # Local storage with hash-based coordinates
    memories = storage.get_all_memories(limit=2000)
    edges = storage.get_edges()
    clusters = storage.get_clusters()
    stats = storage.get_stats()
    
    return {
        "user_id": ctx["user_id"],
        "nodes": [
            {
                "id": m.id,
                "x": m.x,
                "y": m.y,
                "z": m.z,
                "content": m.content[:100] + "..." if len(m.content) > 100 else m.content,
                "title": m.title,
                "cluster": m.cluster,
                "layer": m.layer.value,
                "importance": m.importance,
                "access_count": m.access_count,
                "memory_type": m.memory_type.value
            }
            for m in memories
        ],
        "edges": [
            {
                "id": e.id,
                "source": e.source_id,
                "target": e.target_id,
                "similarity": e.similarity,
                "type": e.relationship_type
            }
            for e in edges
        ],
        "clusters": [
            {
                "id": c.id,
                "name": c.name,
                "x": c.center_x,
                "y": c.center_y,
                "z": c.center_z,
                "radius": c.radius,
                "color": c.color,
                "memory_count": c.memory_count
            }
            for c in clusters
        ],
        "stats": stats
    }


# ============================================================================
# STATS & API KEYS
# ============================================================================

@app.get("/stats")
async def get_stats(storage: UserMemoryStorage = Depends(get_storage)):
    """Get storage statistics."""
    return storage.get_stats()


@app.post("/api-keys")
async def create_api_key(
    name: str = "default",
    storage: UserMemoryStorage = Depends(get_storage),
    ctx: dict = Depends(get_user_context)
):
    """
    Create an API key for external access.
    Users can use this key to embed/retrieve memories from their own applications.
    """
    import secrets
    import hashlib
    
    # Generate API key
    api_key = f"rgm_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    # Store key hash (not the actual key)
    conn = __import__('sqlite3').connect(storage.db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO api_keys (id, key_hash, name, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        secrets.token_urlsafe(8),
        key_hash,
        name,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()
    
    return {
        "api_key": api_key,
        "name": name,
        "message": "Save this key securely. It won't be shown again.",
        "usage": {
            "embed": "POST /memories/embed with X-API-Key header",
            "retrieve": "POST /memories/retrieve with X-API-Key header"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8094)
