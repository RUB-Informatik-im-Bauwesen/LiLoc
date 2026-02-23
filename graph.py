import pathlib
from functools import reduce
from os import PathLike
from typing import Any

import cv2
import networkx as nx
import os
import re
import pathlib
import json
import itertools
import numpy as np

from image_tools import transform_and_overlay, read_image, resize_image, write_image


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
        g = nx.DiGraph()
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

                        self.insert_node(g, img, img_data, group, data.image_type[0], x, y)
                        y += node_pos_stride_y
                x += node_pos_stride_x
                group += 1

            if len(image_set_b) > 0:
                y = 0
                for img, img_data in image_set_b.items():
                    if img not in g.nodes:

                        self.insert_node(g, img, img_data, group, data.image_type[1], x, y)
                        y += node_pos_stride_y
                x += node_pos_stride_x
                group += 1

            for match in matches:
                match_img = base_dir.replace("\\", "/") + "/" + match["match_id"] + "_matches.jpg"
                g.add_edge(match["image_a"], match["image_b"],
                           matches=match["matches"],
                           weight=(1000+np.clip(int(match["matches"]), 100, 1000))/1000,
                           matrix=match["matrix"],
                           title=f"{match['image_a']} <-> {match['image_b']}",
                           match_img=match_img,
                           data_src=data_name)
                g.add_edge(match["image_b"], match["image_a"],
                           matches=match["matches"],
                           weight=(1000+np.clip(int(match["matches"]), 100, 1000))/1000,
                           matrix=np.linalg.inv(match["matrix"]).tolist(),
                           title=f"{match['image_a']} <-> {match['image_b']}",
                           match_img=match_img,
                           data_src=data_name)

        self.graph = g

    def get_graph(self) -> nx.Graph:
        if self.graph is None:
            self.build_network()
        return self.graph

    def insert_node(self, g: nx.Graph, img, img_data, group: int | Any, image_type: str, x: int | Any, y: int, **kwargs):
        g.add_node(img, pos=(x, y), x=x, y=y, title=f"{img}", group=group, image_type=image_type, img=img_data.get("filepath", ""), **kwargs)

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
        unlocalized = list(filter(lambda n: g.nodes[n]["image_type"] == "Unlocalized", g.nodes))
        print(list(unlocalized))
        localized = list(filter(lambda n: g.nodes[n]["image_type"] == "Localized", g.nodes))
        print(localized)

        loc_paths = []

        possible_paths = nx.multi_source_dijkstra_path(g, localized)

        for unloc in unlocalized:
            if unloc in possible_paths:
                loc_paths.append(possible_paths[unloc])

        return loc_paths

    def get_transform_from_path(self, path):
        matrices = list([(np.array(self.graph.edges[n1,n2]["matrix"])) for n1, n2 in itertools.pairwise(path)])
        matrix = np.identity(4)
        for m in reversed(matrices):
            m4 = np.identity(4)
            m4[:3, :3] = m
            matrix = m4 @ matrix
        matrix = matrix[:3, :3] / matrix[2,2]
        return matrix

def main():
    data = [
        DataSource("./a6/matches_pano_512_current/matches.json", image_type=["Localized", ""]),
        DataSource("./a6/matches_bestand_512_current/matches.json", image_type=["Unlocalized", ""]),
        DataSource("./a6/matches_512_current/matches.json", image_type=""),
        DataSource("./a6/matches_512_bestand/matches.json", image_type=["Unlocalized"]),
    ]
    locgraph = LocalizationGraph(data)
    locgraph.build_network()
    paths = locgraph.get_all_node_localizations()
    paths = sorted(paths, key=lambda pa: len(pa))
    for path in paths[2:]:
        print(path)
        matrix = locgraph.get_transform_from_path(path)
        matrices = list([np.array(locgraph.graph.edges[n1,n2]["matrix"]) for n1, n2 in itertools.pairwise(reversed(path))])
        print(matrix)
        loc_node = path[0]
        unloc_node = path[1]
        loc_img = read_image(locgraph.graph.nodes[loc_node]["img"])
        unloc_image = read_image(locgraph.graph.nodes[unloc_node]["img"])
        overlay = transform_and_overlay(loc_img, unloc_image, np.linalg.inv(matrix))
        #for p in path:
        #    img = read_image(locgraph.graph.nodes[p]["img"])
        #    cv2.imshow(p, img)
        im = read_image(locgraph.graph.nodes[path[-1]]["img"], max_size=2048)
        cv2.imshow("overlay", resize_image(im,1024))
        cv2.waitKey(0)

        for n1,n2 in itertools.pairwise(reversed(path)):
            nim = read_image(locgraph.graph.nodes[n2]["img"], max_size=2048)
            m = np.array(locgraph.graph.edges[n1,n2]["matrix"])
            im = transform_and_overlay(nim, im, m)
            print(n2, m)
            cv2.imshow("overlay", resize_image(im, 1024))
            write_image(f"overlay_{n1}_{n2}.png", im)
            cv2.waitKey(0)
        #cv2.imshow("overlay", overlay)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
