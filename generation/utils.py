from collections import deque
import numpy as np

class bfs_node(object):
    def __init__(self, idx, links):
        self.idx = idx
        self.links = links
        self.depth = None

def forfor(a):
        return [item for sublist in a for item in sublist]

def get_bfs_order(edges, n_nodes):
    edges = list(zip(*edges))
    bfs_links = [[] for _ in range(n_nodes)]
    for i in range(n_nodes):
        for link_parent, link_son in edges:
            if link_parent == i:
                bfs_links[i].append(link_son)
            elif link_son == i:
                bfs_links[i].append(link_parent)
    bfs_nodes = [bfs_node(idx, links) for idx, links in enumerate(bfs_links)]
    queue = deque([bfs_nodes[0]])
    visited = set([bfs_nodes[0].idx])
    bfs_nodes[0].depth = 0
    order1,order2 = [],[]
    bfs_order = [0,]
    while len(queue) > 0:
        x = queue.popleft()
        for y in x.links:
            y = bfs_nodes[y]
            if y.idx not in visited:
                queue.append(y)
                visited.add(y.idx)
                y.depth = x.depth + 1
                if y.depth > len(order1):
                    order1.append([])
                    order2.append([])
                order1[y.depth-1].append( (x.idx, y.idx) )
                order2[y.depth-1].append( (y.idx, x.idx) )
                bfs_order.append(y.idx)
    return bfs_order, forfor(order1)
