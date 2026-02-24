import networkx as nx
from kg.channel_mapping import CHANNEL_MAPPING

def build_adcs_graph():
    G = nx.DiGraph()

    # Subsystem
    G.add_node("ADCS", type="subsystem")

    # Components
    components = set(v["component"] for v in CHANNEL_MAPPING.values())
    for comp in components:
        G.add_node(comp, type="component")
        G.add_edge(comp, "ADCS", relation="PART_OF")

    # Channels
    for ch, meta in CHANNEL_MAPPING.items():
        G.add_node(ch, type="channel", **meta)
        G.add_edge(ch, meta["component"], relation="MEASURES")

    return G