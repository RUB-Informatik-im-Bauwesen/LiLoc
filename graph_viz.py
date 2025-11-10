import itertools

import json
import os.path
import pathlib
import networkx as nx
from pyvis.network import Network
import re

from typing import Dict, List, Any

# This script reads the matches.json files from LiLoc and displays the results as a graph.

scan_root = False  # Connect the scans to a root object
root_pano_re = re.compile(r"A6(.*)_(\d{1,2})")

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
    nt = Network('720px', '1920px', filter_menu=True)
    nt.set_template_dir(os.path.dirname(os.path.realpath(__file__)), "graphviz_template.html")
    # populates the nodes and edges data structures
    nt.from_nx(nx_graph)
    nt.toggle_physics(False)
    nt.show_buttons(filter_=['physics'])
    nt.write_html('nx.html', notebook=False)


def build_network(data_list: dict) -> nx.Graph:

    g = nx.Graph()
    if scan_root:
        g.add_node("root", title="Scan Root", x=-2000, y=0, group="root")

    x = 0
    node_pos_stride_x = 1000
    node_pos_stride_y = 40

    group = 0

    current_path = os.path.dirname(os.path.realpath(__file__))

    for data_name, data in data_list.items():
        image_set: dict = data.get("image_set", {})
        image_set_a: dict = data.get("image_set_a", {})
        image_set_b: dict = data.get("image_set_b", {})
        matches: list = data.get("matches", [])

        base_dir = os.path.relpath(os.path.dirname(data_name), current_path)


        if len(image_set) > 0:
            y = 0
            for img, img_data in image_set.items():
                if img not in g.nodes:
                    insert_node(g, img, img_data, group, x, y)
                    y += node_pos_stride_y
            x += node_pos_stride_x
            group += 1

        if len(image_set_a) > 0:
            y = 0
            for img, img_data in image_set_a.items():
                if img not in g.nodes:

                    insert_node(g, img, img_data, group, x, y)
                    y += node_pos_stride_y
            x += node_pos_stride_x
            group += 1

        if len(image_set_b) > 0:
            y = 0
            for img, img_data in image_set_b.items():
                if img not in g.nodes:

                    insert_node(g, img, img_data, group, x, y)
                    y += node_pos_stride_y
            x += node_pos_stride_x
            group += 1

        for match in matches:
            match_img = base_dir.replace("\\", "/") + "/" + match["match_id"] + "_matches.jpg"
            g.add_edge(match["image_a"], match["image_b"],
                       matches=match["matches"],
                       title=f"{match['image_a']} <-> {match['image_b']}",
                       match_img=match_img,
                       data_src=data_name)

    return g

pano_x = -1000
pano_y = 0

def insert_node(g: nx.Graph, img, img_data, group: int | Any, x: int | Any, y: int):
    g.add_node(img, pos=(x, y), x=x, y=y, title=f"{img}", group=group, img=img_data.get("filepath", ""))

    if scan_root:
        if res := root_pano_re.match(img):
            pano_name = res.group(1)
            pano_id = res.group(2)
            scan_name = f"scan_{pano_name}_{int(pano_id) // 6}"
            if scan_name not in g.nodes:
                global pano_y
                g.add_node(scan_name, group="scan", pos=(pano_x, pano_y), x=pano_x, y=pano_y)
                pano_y += 40*6
                g.add_edge("root", scan_name)
            g.add_edge(img, scan_name)


def main():
    args = parse_args()
    data_list = {}
    for f in args.files:
        data_list[f] = read_json_file(f)
    g = build_network(data_list)
    plot_graph_pyvis(g)

if __name__ == '__main__':
    main()

# example usage:
# python graph_viz.py ./a6/matches_pano_512_current/matches.json ./a6/matches_bestand_512_current/matches.json ./a6/matches_512_current/matches.json ./a6/matches_512_bestand/matches.json
