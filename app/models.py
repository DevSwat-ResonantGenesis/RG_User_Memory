"""
User Memory Service Models
Per-user isolated memory storage with Hash Sphere visualization
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class MemoryType(str, Enum):
    """Types of memories that can be stored"""
    TEXT = "text"
    CODE = "code"
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    IMAGE_DESCRIPTION = "image_description"
    CUSTOM = "custom"


class ClusterType(str, Enum):
    """Memory cluster/universe types"""
    DEFAULT = "default"
    WORK = "work"
    PERSONAL = "personal"
    PROJECT = "project"
    KNOWLEDGE = "knowledge"
    CUSTOM = "custom"


class LayerType(str, Enum):
    """Memory layers in the hash universe"""
    CORE = "core"           # Most important, center of universe
    ACTIVE = "active"       # Recently accessed
    ARCHIVE = "archive"     # Older memories
    PERIPHERAL = "peripheral"  # Less important


class MemoryNode(BaseModel):
    """A single memory node in the Hash Sphere"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    org_id: str
    
    # Content
    content: str
    content_hash: str
    memory_type: MemoryType = MemoryType.TEXT
    
    # Embedding vector (stored as list for JSON serialization)
    embedding: List[float] = []
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    
    # Hash Sphere coordinates (computed from embedding)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    # Universe/Cluster/Layer classification
    cluster: str = "default"
    cluster_type: ClusterType = ClusterType.DEFAULT
    layer: LayerType = LayerType.ACTIVE
    universe_id: str = "main"
    
    # Metadata
    title: Optional[str] = None
    tags: List[str] = []
    source: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    # Metrics
    importance: float = 0.5  # 0-1 scale
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryEdge(BaseModel):
    """Connection between memory nodes"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str
    target_id: str
    
    # Relationship
    relationship_type: str = "related"
    similarity: float = 0.0  # Cosine similarity
    weight: float = 1.0
    
    # Metadata
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryCluster(BaseModel):
    """A cluster/universe of related memories"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    org_id: str
    
    name: str
    description: Optional[str] = None
    cluster_type: ClusterType = ClusterType.CUSTOM
    
    # Cluster center in hash space
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    radius: float = 50.0
    
    # Color for visualization
    color: str = "#6366f1"
    
    # Stats
    memory_count: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UserMemoryUniverse(BaseModel):
    """Complete memory universe for a user"""
    user_id: str
    org_id: str
    
    # All nodes and edges
    nodes: Dict[str, MemoryNode] = {}
    edges: Dict[str, MemoryEdge] = {}
    clusters: Dict[str, MemoryCluster] = {}
    
    # Stats
    total_memories: int = 0
    total_embeddings: int = 0
    storage_used_bytes: int = 0
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# API Request/Response Models

class EmbedRequest(BaseModel):
    """Request to embed and store a memory"""
    content: str
    memory_type: MemoryType = MemoryType.TEXT
    title: Optional[str] = None
    tags: List[str] = []
    cluster: str = "default"
    layer: LayerType = LayerType.ACTIVE
    metadata: Dict[str, Any] = {}
    source: Optional[str] = None


class EmbedResponse(BaseModel):
    """Response after embedding a memory"""
    memory_id: str
    content_hash: str
    coordinates: Dict[str, float]  # x, y, z
    cluster: str
    layer: str
    embedding_dim: int
    created_at: str


class RetrieveRequest(BaseModel):
    """Request to retrieve similar memories"""
    query: str
    top_k: int = 10
    cluster: Optional[str] = None
    layer: Optional[LayerType] = None
    min_similarity: float = 0.0
    include_embeddings: bool = False


class RetrieveResponse(BaseModel):
    """Response with retrieved memories"""
    query: str
    results: List[Dict[str, Any]]
    total_searched: int
    retrieval_time_ms: float


class ClusterCreateRequest(BaseModel):
    """Request to create a new cluster"""
    name: str
    description: Optional[str] = None
    cluster_type: ClusterType = ClusterType.CUSTOM
    color: str = "#6366f1"


class UniverseStateResponse(BaseModel):
    """Full universe state for visualization"""
    user_id: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    clusters: List[Dict[str, Any]]
    stats: Dict[str, Any]


class APIKeyResponse(BaseModel):
    """API key for external access"""
    api_key: str
    created_at: str
    expires_at: Optional[str] = None
    scopes: List[str] = ["embed", "retrieve"]
