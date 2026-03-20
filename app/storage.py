"""
Storage Engine for User Memory Service
Per-user isolated memory storage with SQLite backend
"""

import os
import json
import sqlite3
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging

from .models import (
    MemoryNode, MemoryEdge, MemoryCluster, UserMemoryUniverse,
    MemoryType, ClusterType, LayerType
)

logger = logging.getLogger(__name__)

# Storage configuration
DATA_DIR = os.getenv("USER_MEMORY_DATA_DIR", "/data/user_memories")
MAX_MEMORIES_PER_USER = int(os.getenv("MAX_MEMORIES_PER_USER", "10000"))
MAX_STORAGE_MB_PER_USER = int(os.getenv("MAX_STORAGE_MB_PER_USER", "100"))


class UserMemoryStorage:
    """
    Per-user isolated memory storage.
    Each user gets their own SQLite database file.
    """
    
    def __init__(self, user_id: str, org_id: str):
        self.user_id = user_id
        self.org_id = org_id
        self.db_path = self._get_db_path()
        self._ensure_db()
    
    def _get_db_path(self) -> str:
        """Get path to user's database file."""
        # Create directory structure: /data/user_memories/{org_id}/{user_id}.db
        org_dir = Path(DATA_DIR) / self.org_id
        org_dir.mkdir(parents=True, exist_ok=True)
        return str(org_dir / f"{self.user_id}.db")
    
    def _ensure_db(self):
        """Ensure database and tables exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                memory_type TEXT DEFAULT 'text',
                embedding TEXT,
                embedding_model TEXT,
                embedding_dim INTEGER DEFAULT 1536,
                x REAL DEFAULT 0,
                y REAL DEFAULT 0,
                z REAL DEFAULT 0,
                cluster TEXT DEFAULT 'default',
                cluster_type TEXT DEFAULT 'default',
                layer TEXT DEFAULT 'active',
                universe_id TEXT DEFAULT 'main',
                title TEXT,
                tags TEXT,
                source TEXT,
                metadata TEXT,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relationship_type TEXT DEFAULT 'related',
                similarity REAL DEFAULT 0,
                weight REAL DEFAULT 1,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES memories(id),
                FOREIGN KEY (target_id) REFERENCES memories(id)
            )
        """)
        
        # Clusters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                cluster_type TEXT DEFAULT 'custom',
                center_x REAL DEFAULT 0,
                center_y REAL DEFAULT 0,
                center_z REAL DEFAULT 0,
                radius REAL DEFAULT 50,
                color TEXT DEFAULT '#6366f1',
                memory_count INTEGER DEFAULT 0,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # API Keys table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                key_hash TEXT NOT NULL UNIQUE,
                name TEXT,
                scopes TEXT DEFAULT '["embed","retrieve"]',
                created_at TEXT NOT NULL,
                expires_at TEXT,
                last_used TEXT,
                is_active INTEGER DEFAULT 1
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_cluster ON memories(cluster)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_layer ON memories(layer)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
        
        conn.commit()
        conn.close()
    
    def _serialize_list(self, lst: List) -> str:
        """Serialize list to JSON string."""
        return json.dumps(lst)
    
    def _deserialize_list(self, s: str) -> List:
        """Deserialize JSON string to list."""
        if not s:
            return []
        try:
            return json.loads(s)
        except:
            return []
    
    def _serialize_dict(self, d: Dict) -> str:
        """Serialize dict to JSON string."""
        return json.dumps(d)
    
    def _deserialize_dict(self, s: str) -> Dict:
        """Deserialize JSON string to dict."""
        if not s:
            return {}
        try:
            return json.loads(s)
        except:
            return {}
    
    # Memory CRUD operations
    
    def add_memory(self, memory: MemoryNode) -> bool:
        """Add a memory node to storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO memories (
                    id, content, content_hash, memory_type, embedding, embedding_model,
                    embedding_dim, x, y, z, cluster, cluster_type, layer, universe_id,
                    title, tags, source, metadata, importance, access_count,
                    last_accessed, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id, memory.content, memory.content_hash, memory.memory_type.value,
                self._serialize_list(memory.embedding), memory.embedding_model,
                memory.embedding_dim, memory.x, memory.y, memory.z,
                memory.cluster, memory.cluster_type.value, memory.layer.value,
                memory.universe_id, memory.title, self._serialize_list(memory.tags),
                memory.source, self._serialize_dict(memory.metadata),
                memory.importance, memory.access_count,
                memory.last_accessed.isoformat() if memory.last_accessed else None,
                memory.created_at.isoformat(), memory.updated_at.isoformat()
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Memory {memory.id} already exists")
            return False
        finally:
            conn.close()
    
    def get_memory(self, memory_id: str) -> Optional[MemoryNode]:
        """Get a memory by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return self._row_to_memory(row)
    
    def _row_to_memory(self, row) -> MemoryNode:
        """Convert database row to MemoryNode."""
        return MemoryNode(
            id=row[0],
            user_id=self.user_id,
            org_id=self.org_id,
            content=row[1],
            content_hash=row[2],
            memory_type=MemoryType(row[3]),
            embedding=self._deserialize_list(row[4]),
            embedding_model=row[5],
            embedding_dim=row[6],
            x=row[7],
            y=row[8],
            z=row[9],
            cluster=row[10],
            cluster_type=ClusterType(row[11]),
            layer=LayerType(row[12]),
            universe_id=row[13],
            title=row[14],
            tags=self._deserialize_list(row[15]),
            source=row[16],
            metadata=self._deserialize_dict(row[17]),
            importance=row[18],
            access_count=row[19],
            last_accessed=datetime.fromisoformat(row[20]) if row[20] else None,
            created_at=datetime.fromisoformat(row[21]),
            updated_at=datetime.fromisoformat(row[22])
        )
    
    def get_all_memories(
        self, 
        cluster: Optional[str] = None,
        layer: Optional[LayerType] = None,
        limit: int = 1000
    ) -> List[MemoryNode]:
        """Get all memories, optionally filtered."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM memories WHERE 1=1"
        params = []
        
        if cluster:
            query += " AND cluster = ?"
            params.append(cluster)
        
        if layer:
            query += " AND layer = ?"
            params.append(layer.value)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_memory(row) for row in rows]
    
    def get_all_embeddings(self) -> List[tuple]:
        """Get all memory IDs and embeddings for similarity search."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()
        conn.close()
        
        return [(row[0], self._deserialize_list(row[1])) for row in rows if row[1]]
    
    def update_access(self, memory_id: str):
        """Update access count and timestamp for a memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE memories 
            SET access_count = access_count + 1, last_accessed = ?
            WHERE id = ?
        """, (datetime.utcnow().isoformat(), memory_id))
        
        conn.commit()
        conn.close()
    
    def update_layer(self, memory_id: str, layer: str) -> bool:
        """
        Update the layer of a memory.
        
        IMPORTANT: Memories can NEVER be deleted because they are interconnected
        in the hash universe. Deleting one would break the integrity of the entire
        system. Instead, memories can only be moved between layers (archive, active, etc).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE memories 
            SET layer = ?, updated_at = ?
            WHERE id = ?
        """, (layer, datetime.utcnow().isoformat(), memory_id))
        
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated
    
    def update_importance(self, memory_id: str, importance: float) -> bool:
        """Update the importance score of a memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clamp importance between 0 and 1
        importance = max(0.0, min(1.0, importance))
        
        cursor.execute("""
            UPDATE memories 
            SET importance = ?, updated_at = ?
            WHERE id = ?
        """, (importance, datetime.utcnow().isoformat(), memory_id))
        
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated
    
    # Cluster operations
    
    def add_cluster(self, cluster: MemoryCluster) -> bool:
        """Add a cluster."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO clusters (
                    id, name, description, cluster_type, center_x, center_y, center_z,
                    radius, color, memory_count, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cluster.id, cluster.name, cluster.description, cluster.cluster_type.value,
                cluster.center_x, cluster.center_y, cluster.center_z,
                cluster.radius, cluster.color, cluster.memory_count,
                self._serialize_dict(cluster.metadata),
                cluster.created_at.isoformat(), cluster.updated_at.isoformat()
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def get_clusters(self) -> List[MemoryCluster]:
        """Get all clusters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM clusters")
        rows = cursor.fetchall()
        conn.close()
        
        clusters = []
        for row in rows:
            clusters.append(MemoryCluster(
                id=row[0],
                user_id=self.user_id,
                org_id=self.org_id,
                name=row[1],
                description=row[2],
                cluster_type=ClusterType(row[3]),
                center_x=row[4],
                center_y=row[5],
                center_z=row[6],
                radius=row[7],
                color=row[8],
                memory_count=row[9],
                metadata=self._deserialize_dict(row[10]),
                created_at=datetime.fromisoformat(row[11]),
                updated_at=datetime.fromisoformat(row[12])
            ))
        return clusters
    
    # Edge operations
    
    def add_edge(self, edge: MemoryEdge) -> bool:
        """Add an edge between memories."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO edges (
                    id, source_id, target_id, relationship_type, similarity,
                    weight, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                edge.id, edge.source_id, edge.target_id, edge.relationship_type,
                edge.similarity, edge.weight, self._serialize_dict(edge.metadata),
                edge.created_at.isoformat()
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def get_edges(self) -> List[MemoryEdge]:
        """Get all edges."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM edges")
        rows = cursor.fetchall()
        conn.close()
        
        edges = []
        for row in rows:
            edges.append(MemoryEdge(
                id=row[0],
                source_id=row[1],
                target_id=row[2],
                relationship_type=row[3],
                similarity=row[4],
                weight=row[5],
                metadata=self._deserialize_dict(row[6]),
                created_at=datetime.fromisoformat(row[7])
            ))
        return edges
    
    # Stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM memories")
        memory_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM edges")
        edge_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM clusters")
        cluster_count = cursor.fetchone()[0]
        
        conn.close()
        
        # Get file size
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        return {
            "total_memories": memory_count,
            "total_edges": edge_count,
            "total_clusters": cluster_count,
            "storage_bytes": db_size,
            "storage_mb": round(db_size / (1024 * 1024), 2)
        }


def get_user_storage(user_id: str, org_id: str) -> UserMemoryStorage:
    """Get storage instance for a user."""
    return UserMemoryStorage(user_id, org_id)
