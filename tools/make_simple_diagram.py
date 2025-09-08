#!/usr/bin/env python3
"""
Generate a clean, interview-friendly architecture diagram in Mermaid syntax.
Outputs both Mermaid code and a simple PNG version.
"""

def generate_mermaid_diagram():
    """Generate Mermaid flowchart syntax for the RTB system."""
    
    mermaid_code = """```mermaid
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
- 145+ brand safety keywords"""

    return mermaid_code

def generate_simple_png():
    """Generate a simple PNG version using matplotlib with hand-drawn style."""
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    
    # Set up figure with simple styling
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Simple color scheme
    colors = {
        'input': '#e3f2fd',      # Light blue
        'processing': '#fff8e1',  # Light beige
        'model': '#e8f5e8',      # Light green
        'artifacts': '#fffde7',   # Light yellow
        'monitoring': '#fce4ec'   # Light pink
    }
    
    # Title
    ax.text(7, 7.5, 'Crocs RTB Relevance System', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Main flow boxes
    boxes = [
        (1, 5.5, 'Request\n{url, snippet}', colors['input']),
        (2.5, 5.5, 'Brand Safety\nFilter', colors['processing']),
        (4, 5.5, 'Cache\nLookup', colors['processing']),
        (5.5, 5.5, 'Text\nEmbedding', colors['processing']),
        (7, 5.5, 'Logistic\nRegression', colors['model']),
        (8.5, 5.5, 'Threshold\nCheck', colors['processing']),
        (10, 5.5, 'CPM\nMapping', colors['processing']),
        (11.5, 5.5, 'Response\n{bid, price, score}', colors['input'])
    ]
    
    # Draw main flow boxes
    for x, y, text, color in boxes:
        box = FancyBboxPatch(
            (x - 0.6, y - 0.4), 1.2, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='#666',
            linewidth=1
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    # Draw arrows between main boxes
    for i in range(len(boxes) - 1):
        x1, y1 = boxes[i][0] + 0.6, boxes[i][1]
        x2, y2 = boxes[i+1][0] - 0.6, boxes[i+1][1]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), 
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))
    
    # Offline artifacts box
    artifacts_text = "Offline Artifacts\n• Brief embedding\n• Trained model\n• Feature scalers"
    artifacts_box = FancyBboxPatch(
        (2, 2.5), 3, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['artifacts'],
        edgecolor='#666',
        linewidth=1
    )
    ax.add_patch(artifacts_box)
    ax.text(3.5, 3.25, artifacts_text, ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Connect artifacts to embedding and model
    ax.annotate('', xy=(5.5, 5.1), xytext=(4.5, 4), 
               arrowprops=dict(arrowstyle='->', lw=1, color='#666', linestyle='--'))
    ax.annotate('', xy=(7, 5.1), xytext=(4.5, 4), 
               arrowprops=dict(arrowstyle='->', lw=1, color='#666', linestyle='--'))
    
    # Monitoring box
    monitoring_text = "Monitoring & Metrics\n• Latency tracking\n• Bid rate monitoring\n• Model drift detection"
    monitoring_box = FancyBboxPatch(
        (9, 2.5), 3.5, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['monitoring'],
        edgecolor='#666',
        linewidth=1
    )
    ax.add_patch(monitoring_box)
    ax.text(10.75, 3.25, monitoring_text, ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Connect monitoring
    ax.annotate('', xy=(10.75, 4), xytext=(7, 5.1), 
               arrowprops=dict(arrowstyle='->', lw=1, color='#666', linestyle='--'))
    
    # Tech notes
    tech_notes = [
        "Tech Notes:",
        "• Sub-100ms latency target",
        "• Supports up to 20k QPS",
        "• MiniLM sentence transformer encoder", 
        "• CPM range: $0.50 - $3.00",
        "• In-memory LRU cache (1000 items)",
        "• 145+ brand safety keywords"
    ]
    
    for i, note in enumerate(tech_notes):
        weight = 'bold' if i == 0 else 'normal'
        ax.text(1, 1.5 - i*0.15, note, fontsize=9, fontweight=weight, ha='left')
    
    plt.tight_layout()
    plt.savefig('diagram_simple.png', dpi=100, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ Simple diagram saved as diagram_simple.png")

if __name__ == '__main__':
    print("🎯 Generating clean, interview-friendly architecture diagram...")
    
    # Generate Mermaid code
    mermaid_code = generate_mermaid_diagram()
    
    # Save Mermaid code to file
    with open('diagram_mermaid.md', 'w') as f:
        f.write("# Crocs RTB Relevance System Architecture\n\n")
        f.write(mermaid_code)
    
    print("✅ Mermaid diagram saved to diagram_mermaid.md")
    print("\n" + "="*60)
    print("MERMAID CODE (copy to draw.io or Markdown):")
    print("="*60)
    print(mermaid_code)
    
    # Also generate simple PNG
    try:
        generate_simple_png()
    except ImportError:
        print("⚠️  Matplotlib not available for PNG generation")
    
    print("\n✅ Done! Use the Mermaid code above in draw.io or any Markdown viewer.")
