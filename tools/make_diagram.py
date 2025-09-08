#!/usr/bin/env python3
"""
Generate architecture diagram for Crocs RTB Relevance System.
Outputs a clean 1200x700 PNG showing the complete request flow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_architecture_diagram():
    """Create a comprehensive architecture diagram for the RTB system."""
    
    # Set up the figure with high DPI for crisp output
    fig, ax = plt.subplots(figsize=(16, 9.33))  # 16:9 aspect ratio for 1200x700
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E3F2FD',      # Light blue
        'processing': '#FFF3E0',  # Light orange
        'model': '#E8F5E8',      # Light green
        'output': '#F3E5F5',     # Light purple
        'storage': '#FFF9C4',    # Light yellow
        'arrow': '#424242',      # Dark gray
        'text': '#212121'        # Almost black
    }
    
    # Title
    ax.text(6, 7.5, 'Crocs RTB Relevance System Architecture', 
            fontsize=24, fontweight='bold', ha='center', color=colors['text'])
    
    # Main request flow (top section)
    ax.text(6, 6.8, 'Real-time Request Flow', 
            fontsize=16, fontweight='bold', ha='center', color=colors['text'])
    
    # Step boxes with improved styling
    steps = [
        (1, 6, 'Request\n{url, snippet}', colors['input']),
        (2.5, 6, 'Brand Safety\nFilter', colors['processing']),
        (4, 6, 'Cache\nLookup', colors['processing']),
        (5.5, 6, 'Text Features\n(Embeddings)', colors['processing']),
        (7, 6, 'Logistic\nRegression', colors['model']),
        (8.5, 6, 'Threshold\nCheck', colors['processing']),
        (10, 6, 'CPM\nMapping', colors['processing']),
        (11.5, 6, 'Response\n{bid, price, score}', colors['output'])
    ]
    
    # Draw step boxes
    box_width = 1.2
    box_height = 0.8
    
    for x, y, text, color in steps:
        # Create rounded rectangle
        box = FancyBboxPatch(
            (x - box_width/2, y - box_height/2), 
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='#666666',
            linewidth=1.5
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=9, fontweight='bold', color=colors['text'])
    
    # Draw arrows between steps
    arrow_props = dict(arrowstyle='->', lw=2.5, color=colors['arrow'])
    
    for i in range(len(steps) - 1):
        x1, y1 = steps[i][0] + box_width/2, steps[i][1]
        x2, y2 = steps[i+1][0] - box_width/2, steps[i+1][1]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_props)
    
    # Offline artifacts section (bottom left)
    ax.text(2.5, 4.5, 'Offline Artifacts', 
            fontsize=14, fontweight='bold', ha='center', color=colors['text'])
    
    artifacts = [
        (1, 3.8, 'Brief\nEmbedding', colors['storage']),
        (2.5, 3.8, 'Trained\nModel', colors['storage']),
        (4, 3.8, 'Vectorizers\n& Scalers', colors['storage'])
    ]
    
    for x, y, text, color in artifacts:
        box = FancyBboxPatch(
            (x - 0.6, y - 0.4), 1.2, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='#666666',
            linewidth=1
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=9, fontweight='bold', color=colors['text'])
    
    # Monitoring section (bottom right)
    ax.text(9, 4.5, 'Monitoring & Metrics', 
            fontsize=14, fontweight='bold', ha='center', color=colors['text'])
    
    metrics = [
        (7.5, 3.8, 'Latency\nTracking', colors['output']),
        (9, 3.8, 'Bid Rate\nMonitoring', colors['output']),
        (10.5, 3.8, 'Model\nPerformance', colors['output'])
    ]
    
    for x, y, text, color in metrics:
        box = FancyBboxPatch(
            (x - 0.6, y - 0.4), 1.2, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='#666666',
            linewidth=1
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=9, fontweight='bold', color=colors['text'])
    
    # Connect artifacts to main flow
    # Brief embedding to text features
    ax.annotate('', xy=(5.5, 5.5), xytext=(1, 4.2), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#666666', 
                               connectionstyle="arc3,rad=0.3", linestyle='--'))
    
    # Model to logistic regression
    ax.annotate('', xy=(7, 5.5), xytext=(2.5, 4.2), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#666666', 
                               connectionstyle="arc3,rad=0.3", linestyle='--'))
    
    # Vectorizers to text features
    ax.annotate('', xy=(5.5, 5.5), xytext=(4, 4.2), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#666666', 
                               connectionstyle="arc3,rad=0.2", linestyle='--'))
    
    # Connect monitoring to main flow
    ax.annotate('', xy=(9, 4.2), xytext=(9, 5.5), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#666666', 
                               linestyle='--'))
    
    # Add technical details
    ax.text(6, 2.5, 'Technical Specifications', 
            fontsize=14, fontweight='bold', ha='center', color=colors['text'])
    
    tech_details = [
        '• Sub-100ms latency target',
        '• Supports up to 20k QPS',
        '• Sentence transformer embeddings (paraphrase-MiniLM-L3-v2)',
        '• Feature engineering: cosine similarity + element-wise operations',
        '• F1-optimized decision threshold',
        '• CPM range: $0.50 - $3.00',
        '• In-memory LRU caching for performance'
    ]
    
    for i, detail in enumerate(tech_details):
        ax.text(1, 2.1 - i*0.2, detail, fontsize=9, ha='left', 
                color=colors['text'], fontfamily='monospace')
    
    # Add cache details
    ax.text(4, 5.2, 'LRU Cache\n(1000 items)', ha='center', va='center', 
            fontsize=8, style='italic', color='#666666')
    
    # Add brand safety note
    ax.text(2.5, 5.2, '145+ blocked\nkeywords', ha='center', va='center', 
            fontsize=8, style='italic', color='#666666')
    
    # Add performance indicators
    ax.text(11.5, 5.2, 'JSON Response\n~50-80ms', ha='center', va='center', 
            fontsize=8, style='italic', color='#666666')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Input'),
        mpatches.Patch(color=colors['processing'], label='Processing'),
        mpatches.Patch(color=colors['model'], label='ML Model'),
        mpatches.Patch(color=colors['storage'], label='Artifacts'),
        mpatches.Patch(color=colors['output'], label='Output/Monitoring')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    # Add watermark/footer
    ax.text(6, 0.3, 'Built with sentence-transformers + scikit-learn | FastAPI + Uvicorn | Deployed on Render', 
            ha='center', va='center', fontsize=9, style='italic', color='#888888')
    
    plt.tight_layout()
    
    # Save with high DPI for crisp output
    plt.savefig('diagram.png', dpi=75, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ Architecture diagram saved as diagram.png (1200x700)")
    print("📊 Diagram includes:")
    print("   - Complete request flow from input to response")
    print("   - Offline artifacts and their connections")
    print("   - Monitoring and metrics components")
    print("   - Technical specifications and performance details")
    print("   - Color-coded component categories")

if __name__ == '__main__':
    create_architecture_diagram()
