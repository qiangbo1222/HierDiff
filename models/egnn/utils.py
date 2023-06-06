import torch
import itertools as it


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def edge_to_angle_index(edge_index):
    edge_attr_bidirection = edge_index
    for edge in edge_index:
        if (edge[1], edge[0]) not in edge_index:
            edge_index_bidirection.append((edge[1], edge[0]))
    angle_index, collect_edges = [], {}
    for edge in edge_index_bidirection:
        if edge[0] not in collect_edges:
            collect_edges[edge[0]] = []
        collect_edges[edge[0]].append(edge[1])
    for k in collect_edges:
        iter_neighbors = list(it.combinations(collect_edges[k], 2))
        angle = [[k, i[0], i[1]] for i in iter_neighbors]
        angle_index.extend(angle)
    return angle_index


def angle_from_index(angle_index, pos):
    c = [a[0] for a in angle_index]
    a1 = [a[1] for a in angle_index]
    a2 = [a[2] for a in angle_index]
    angle_pos = [pos[c], pos[a1], pos[a2]]
    pos_ij = - angle_pos[0] + angle_pos[1]
    pos_ik = - angle_pos[0] + angle_pos[2]

    a = (pos_ij * pos_ik).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ij, pos_ik).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)
    return angle