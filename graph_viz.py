import itertools

import json
import pathlib
import networkx as nx
from pyvis.network import Network

from typing import Dict, List, Any

# This script reads the matches.json files from LiLoc and displays the results as a graph.

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Graph Visualization for LiLoc")
    parser.add_argument('files', nargs='+', help='list of JSON file paths')
    return parser.parse_args()

def read_json_file(file_path):
    data = {}
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing {file_path}: {e}")
    return data

def plot_graph_pyvis(nx_graph: nx.Graph):
    nt = Network('1080px', '1920px')
    # populates the nodes and edges data structures
    nt.from_nx(nx_graph)
    nt.toggle_physics(False)
    nt.show_buttons(filter_=['physics'])
    nt.show('nx.html', notebook=False)

def build_network(data_list) -> nx.Graph:

    g = nx.Graph()

    x = 0
    node_pos_stride_x = 1000
    node_pos_stride_y = 40

    group = 0

    for data in data_list:
        image_set = data.get("image_set", [])
        image_set_a = data.get("image_set_a", [])
        image_set_b = data.get("image_set_b", [])
        matches = data.get("matches", [])

        if len(image_set) > 0:
            y = 0
            for img in image_set:
                if img not in g.nodes:
                    g.add_node(img, pos=(x, y), x=x, y=y, title=f"<img src={img}></img>", group=group)
                    y += node_pos_stride_y
            x += node_pos_stride_x
            group += 1

        if len(image_set_a) > 0:
            y = 0
            for img in image_set_a:
                if img not in g.nodes:
                    g.add_node(img, pos=(x, y), x=x, y=y, title=f"<img src={img}></img>", group=group)
                    y += node_pos_stride_y
            x += node_pos_stride_x
            group += 1

        if len(image_set_b) > 0:
            y = 0
            for img in image_set_b:
                if img not in g.nodes:
                    g.add_node(img, pos=(x, y), x=x, y=y, title=f"<img src={img}></img>", group=group)
                    y += node_pos_stride_y
            x += node_pos_stride_x
            group += 1

        for match in matches:
            g.add_edge(match["image_a"] ,match["image_b"], matches=match["matches"], title=f"{match['image_a']} <-> {match['image_b']}")

    return g

def main():
    args = parse_args()
    data_list = [read_json_file(f) for f in args.files]
    g = build_network(data_list)
    plot_graph_pyvis(g)

if __name__ == '__main__':
    main()

# example usage:
# python graph_viz.py ./a6/matches_pano_512_current/matches.json ./a6/matches_bestand_512_current/matches.json ./a6/matches_512_current/matches.json ./a6/matches_512_bestand/matches.json
