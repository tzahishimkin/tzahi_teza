# Crocs RTB Relevance System Architecture

```mermaid
flowchart LR
    %% Input
    A[Request<br/>{url, snippet}] --> B[Brand Safety<br/>Filter]
    
    %% Processing pipeline
    B --> C[Cache<br/>Lookup]
    C --> D[Text<br/>Embedding]
    D --> E[Logistic<br/>Regression]
    E --> F[Threshold<br/>Check]
    F --> G[CPM<br/>Mapping]
    
    %% Output
    G --> H[Response<br/>{bid, price, score}]
    
    %% Offline artifacts (single box)
    I[Offline Artifacts<br/>• Brief embedding<br/>• Trained model<br/>• Feature scalers] 
    I -.-> D
    I -.-> E
    
    %% Monitoring (single box to the side)
    J[Monitoring & Metrics<br/>• Latency tracking<br/>• Bid rate monitoring<br/>• Model drift detection]
    
    %% Connect monitoring
    E -.-> J
    H -.-> J
    
    %% Styling
    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef processing fill:#fff8e1,stroke:#f57c00,stroke-width:2px
    classDef model fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef artifacts fill:#fffde7,stroke:#fbc02d,stroke-width:2px
    classDef monitoring fill:#fce4ec,stroke:#ad1457,stroke-width:2px
    
    class A,H input
    class B,C,D,F,G processing
    class E model
    class I artifacts
    class J monitoring
```

**Tech Notes:**
- Sub-100ms latency target
- Supports up to 20k QPS  
- MiniLM sentence transformer encoder
- CPM range: $0.50 - $3.00
- In-memory LRU cache (1000 items)
- 145+ brand safety keywords