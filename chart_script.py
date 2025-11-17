import plotly.graph_objects as go

# Create a flowchart-style diagram using Plotly with better layout
fig = go.Figure()

# Define positions for nodes (x, y coordinates) - more spread out to use space better
positions = {
    "RGB Input": (2, 7),
    "Layer 1": (2, 5.5),
    "Fruit Type": (2, 4),
    "Layer 2": (2, 2.5),
    "Ripeness": (2, 1),
    "LED Control": (2, -0.5),
    "LED Legend": (5, -0.5)  # Moved closer to LED Control
}

# Define node colors based on system layers
colors = {
    "RGB Input": "#FFEB8A",      # Light yellow for input
    "Layer 1": "#B3E5EC",        # Light cyan for layer 1
    "Fruit Type": "#B3E5EC",     # Light cyan for layer 1
    "Layer 2": "#A5D6A7",        # Light green for layer 2
    "Ripeness": "#A5D6A7",       # Light green for layer 2
    "LED Control": "#FFCDD2",    # Light red for output
    "LED Legend": "#9FA8B0"      # Light blue-gray for legend
}

# Add nodes as scatter points with text - larger markers
for node, (x, y) in positions.items():
    marker_size = 120 if node != "LED Legend" else 100
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers+text',
        marker=dict(size=marker_size, color=colors[node], line=dict(width=2, color='black')),
        text=node,
        textposition="middle center",
        textfont=dict(size=12, color='black', family="Arial Black"),
        showlegend=False,
        name=node
    ))

# Add detailed text for each node with larger fonts
details = {
    "RGB Input": "R:150, G:120, B:85",
    "Layer 1": "Fruit Classifier<br>(1 Model)",
    "Fruit Type": "Output: 1<br>(Banana)",
    "Layer 2": "Ripeness Classifier<br>(5 Fruit-Specific Models)",
    "Ripeness": "Output: 2<br>(Ripe)",
    "LED Control": "Arduino Control<br>White LED Active",
}

# Add detail text next to each node with better positioning and larger fonts
for node, (x, y) in positions.items():
    if node in details:
        fig.add_trace(go.Scatter(
            x=[x + 1.8], y=[y],
            mode='text',
            text=details[node],
            textposition="middle left",
            textfont=dict(size=11, color='#333333'),
            showlegend=False
        ))

# Add LED Legend as separate text box with better formatting
led_legend_text = "LED Codes:<br>0 = Green (Early Ripe)<br>1 = Yellow (Partially Ripe)<br>2 = White (Ripe)<br>3 = Red (Decay)"
fig.add_trace(go.Scatter(
    x=[5], y=[-1.2],
    mode='text',
    text=led_legend_text,
    textposition="middle center",
    textfont=dict(size=10, color='#333333'),
    showlegend=False
))

# Add arrows between nodes with better positioning
arrows = [
    ("RGB Input", "Layer 1"),
    ("Layer 1", "Fruit Type"),
    ("Fruit Type", "Layer 2"),
    ("Layer 2", "Ripeness"),
    ("Ripeness", "LED Control")
]

for start, end in arrows:
    x0, y0 = positions[start]
    x1, y1 = positions[end]
    
    # Add arrow annotation
    fig.add_annotation(
        x=x1, y=y1+0.4,
        ax=x0, ay=y0-0.4,
        xref='x', yref='y',
        axref='x', ayref='y',
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor='black',
        showarrow=True
    )

# Add connection line from LED Control to LED Legend
fig.add_annotation(
    x=5, y=-0.1,
    ax=2.8, ay=-0.5,
    xref='x', yref='y',
    axref='x', ayref='y',
    arrowhead=0,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor='gray',
    showarrow=True,
    opacity=0.6
)

# Update layout to better use space
fig.update_layout(
    title="Fruit Ripeness Detection System",
    xaxis=dict(
        range=[-1, 7],
        showgrid=False,
        showticklabels=False,
        zeroline=False
    ),
    yaxis=dict(
        range=[-2.5, 8],
        showgrid=False,
        showticklabels=False,
        zeroline=False
    ),
    plot_bgcolor='white',
    showlegend=False
)

# Save the chart
fig.write_image("fruit_detection_flowchart.png")
fig.write_image("fruit_detection_flowchart.svg", format="svg")

print("Improved flowchart created successfully using Plotly")