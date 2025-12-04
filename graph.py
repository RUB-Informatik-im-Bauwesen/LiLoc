import pathlib
from os import PathLike
from typing import Any

import networkx as nx
import os
import re
import pathlib
import json

class DataSource(dict):
    def __init__(self, matches_file: (str | PathLike), image_type="Localized", name=""):
        super().__init__()
        self.matches_file = matches_file
        self.update(self._load_data())
        self.image_type = image_type
        if name:
            self.name = name
        else:
            self.name = str(matches_file)
        self.data = None

    def _load_data(self):
        data = {}
        with open(self.matches_file, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error parsing {self.matches_file}: {e}")
        return data


class LocalizationGraph:
    def __init__(self, data_list:list[DataSource]=None, scan_root_re=""):
        if data_list is None:
            data_list = list()
        if scan_root_re:
            self.scan_root = True
            self.root_pano_re = re.compile(scan_root_re)
        else:
            self.scan_root = False
        self.data_list = data_list

        self._pano_x = -1000
        self._pano_y = 0
        self.graph = None

    def build_network(self):
        g = nx.Graph()
        if self.scan_root:
            g.add_node("root", title="Scan Root", x=-2000, y=0, group="root")

        x = 0
        node_pos_stride_x = 1000
        node_pos_stride_y = 40

        group = 0

        current_path = os.path.dirname(os.path.realpath(__file__))

        for data in self.data_list:
            data_name = data.name
            image_set: dict = data.get("image_set", {})
            image_set_a: dict = data.get("image_set_a", {})
            image_set_b: dict = data.get("image_set_b", {})
            matches: list = data.get("matches", [])

            base_dir = os.path.relpath(os.path.dirname(data_name), current_path)


            if len(image_set) > 0:
                y = 0
                for img, img_data in image_set.items():
                    if img not in g.nodes:
                        self.insert_node(g, img, img_data, group, data.image_type, x, y)
                        y += node_pos_stride_y
                x += node_pos_stride_x
                group += 1

            if len(image_set_a) > 0:
                y = 0
                for img, img_data in image_set_a.items():
                    if img not in g.nodes:

                        self.insert_node(g, img, img_data, group, data.image_type, x, y)
                        y += node_pos_stride_y
                x += node_pos_stride_x
                group += 1

            if len(image_set_b) > 0:
                y = 0
                for img, img_data in image_set_b.items():
                    if img not in g.nodes:

                        self.insert_node(g, img, img_data, group, data.image_type, x, y)
                        y += node_pos_stride_y
                x += node_pos_stride_x
                group += 1

            for match in matches:
                match_img = base_dir.replace("\\", "/") + "/" + match["match_id"] + "_matches.jpg"
                g.add_edge(match["image_a"], match["image_b"],
                           matches=match["matches"],
                           weight=match["matches"],
                           matrix=match["matrix"],
                           title=f"{match['image_a']} <-> {match['image_b']}",
                           match_img=match_img,
                           data_src=data_name)

        self.graph = g

    def get_graph(self) -> nx.Graph:
        if self.graph is None:
            self.build_network()
        return self.graph

    def insert_node(self, g: nx.Graph, img, img_data, group: int | Any, image_type: str, x: int | Any, y: int):
        g.add_node(img, pos=(x, y), x=x, y=y, title=f"{img}", group=group, image_type="", img=img_data.get("filepath", ""))

        if self.scan_root:
            if res := self.root_pano_re.match(img):
                pano_name = res.group(1)
                pano_id = res.group(2)
                scan_name = f"scan_{pano_name}_{int(pano_id) // 6}"
                if scan_name not in g.nodes:
                    g.add_node(scan_name, group="scan", pos=(self._pano_x, self._pano_y), x=self._pano_x, y=self._pano_y)
                    self._pano_y += 40 * 6
                    g.add_edge("root", scan_name)
                g.add_edge(img, scan_name)

    def get_all_node_localizations(self) -> list:
        g = self.get_graph()
        # Find nodes to localize
        unlocalized = filter(lambda n: n.image_type == "Unlocalized", g.nodes)
        localized = list(filter(lambda n: n.image_type == "Localized", g.nodes))

        loc_paths = []


        for unloc in unlocalized:
            possible_paths = nx.multi_source_dijkstra_path(g, localized, unloc)

            closest_path = min(possible_paths.items(), key=lambda p: len(p))  # TODO sort by weight
            loc_paths.append(closest_path)

        return loc_paths
