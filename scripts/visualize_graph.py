import sys
import webbrowser
from pathlib import Path
import networkx as nx

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import settings

def visualize():
    graph_path = settings.GRAPH_DIR / "knowledge_graph.graphml"
    if not graph_path.exists():
        print(f"Graph file not found at {graph_path}. Please run the ingestion pipeline first.")
        sys.exit(1)
        
    print(f"Loading graph from {graph_path}...")
    try:
        from pyvis.network import Network
    except ImportError:
        print("The 'pyvis' library is required for visualization.")
        print("Please install it by running: venv/bin/pip install pyvis")
        sys.exit(1)

    nx_graph = nx.read_graphml(str(graph_path))
    
    print(f"Graph loaded: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges.")
    
    # Create a Pyvis network
    net = Network(
        height="750px", 
        width="100%", 
        bgcolor="#ffffff", 
        font_color="black", 
        directed=False
    )
    
    # Populate the network from NetworkX graph
    net.from_nx(nx_graph)
    
    # Customize node appearance based on entity type if available
    for node in net.nodes:
        entity_type = nx_graph.nodes[node["id"]].get("type", "UNKNOWN")
        node["title"] = f"Type: {entity_type}"
        
        # Simple color mapping based on common entity types
        color_map = {
            "ORG": "#ff9999",
            "PERSON": "#99ccff",
            "PRODUCT": "#99ff99",
            "GPE": "#ffcc99",
            "EVENT": "#cc99ff",
            "TECH": "#ffff99"
        }
        node["color"] = color_map.get(entity_type, "#cccccc")
        
    # Customize edge appearance to show relationship type
    for edge in net.edges:
        source = edge["from"]
        target = edge["to"]
        edge_data = nx_graph[source][target]
        
        rel_type = edge_data.get("relation_type", "related_to")
        weight = edge_data.get("weight", 1)
        
        edge["title"] = f"Relation: {rel_type} (Weight: {weight})"
        edge["label"] = rel_type  # Display the relation type on the edge
        edge["value"] = weight    # Edge thickness based on weight

    # Generate and save the visualization
    output_file = "knowledge_graph_viz.html"
    net.save_graph(output_file)
    print(f"Visualization saved to {output_file}")
    
    # Attempt to open it in the default web browser
    output_path = Path(output_file).absolute()
    webbrowser.open(f"file://{output_path}")
    print("Opened in your default web browser.")

if __name__ == "__main__":
    visualize()
