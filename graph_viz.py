import itertools

import json
import os.path
import pathlib
import networkx as nx
from pyvis.network import Network
import re

from typing import Dict, List, Any

import graph
from graph import DataSource

# This script reads the matches.json files from LiLoc and displays the results as a graph.

scan_root = False  # Connect the scans to a root object
root_pano_re = re.compile(r"A6(.*)_(\d{1,2})")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Graph Visualization for LiLoc")
    parser.add_argument('files', nargs='+', help='list of JSON file paths')
    return parser.parse_args()

def plot_graph_pyvis(nx_graph: nx.Graph):
    nt = Network('720px', '1920px', filter_menu=True)
    nt.set_template_dir(os.path.dirname(os.path.realpath(__file__)), "graphviz_template.html")
    # populates the nodes and edges data structures
    nt.from_nx(nx_graph)
    nt.toggle_physics(False)
    nt.show_buttons(filter_=['physics'])
    nt.write_html('nx.html', notebook=False)




def main():
    args = parse_args()
    data_list = []
    for f in args.files:
        data_list.append(DataSource(f))
    g = graph.LocalizationGraph(data_list)
    plot_graph_pyvis(g.get_graph())

if __name__ == '__main__':
    main()

# example usage:
# python graph_viz.py ./a6/matches_pano_512_current/matches.json ./a6/matches_bestand_512_current/matches.json ./a6/matches_512_current/matches.json ./a6/matches_512_bestand/matches.json
