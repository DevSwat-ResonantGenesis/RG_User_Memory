# RG User Memory

> **Part of the [ResonantGenesis](https://resonant.dev-swat.com) platform** — Per-user Hash Sphere memory panel with 3D visualization.

[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()
[![Port: 8094](https://img.shields.io/badge/Port-8094-orange.svg)]()
[![License: RG Source Available](https://img.shields.io/badge/License-RG%20Source%20Available-blue.svg)](LICENSE.txt)

Per-user isolated memory storage with 3D Hash Sphere visualization coordinates. Can use local SQLite storage with hash-based coordinates or proxy to the main memory_service for real ResonanceHasher PCA coordinates. Includes a built-in frontend for 3D memory visualization.

## Features

- **Isolated memory storage** — Per-user memory panels
- **3D coordinates** — Hash-based or PCA-derived 3D positions for visualization
- **Frontend included** — Built-in 3D memory visualization UI
- **Dual mode** — Local SQLite or proxy to main memory service

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8094 --reload
```

## Deployment Status

- **Extracted from**: `genesis2026_production_backend/user_memory_service/`
- **Server path**: `/home/deploy/RG_User_Memory`
- **Docker service**: `user_memory_service`

---
**Organization**: [DevSwat-ResonantGenesis](https://github.com/DevSwat-ResonantGenesis) | **Platform**: [resonant.dev-swat.com](https://resonant.dev-swat.com)
