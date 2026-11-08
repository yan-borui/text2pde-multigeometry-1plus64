"""Recover original triangular faces from the released planar edge graph."""

from __future__ import annotations

import numpy as np


def recover_mesh(mesh: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keep source node indices and edges; discard exterior and obstacle faces."""
    points = np.column_stack((mesh["x"], mesh["y"])).astype(np.float64)
    raw_edges = np.asarray(mesh["edges"])
    edges = raw_edges.astype(np.int64)
    if (
        not np.isfinite(points).all()
        or edges.ndim != 2
        or edges.shape[1] != 2
        or not np.array_equal(raw_edges, edges)
        or edges.min() < 0
        or edges.max() >= len(points)
        or np.any(edges[:, 0] == edges[:, 1])
    ):
        raise ValueError("invalid original high-resolution mesh")
    original = {(int(a), int(b)) for a, b in edges}
    if len(original) != len(edges) or any((b, a) not in original for a, b in original):
        raise ValueError("expected unique bidirectional original edges")
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    if not np.allclose(lengths, mesh["edge_properties"], rtol=1e-10, atol=1e-12):
        raise ValueError("original edge lengths disagree with coordinates")
    neighbors: list[list[int]] = [[] for _ in points]
    for a, b in edges:
        neighbors[a].append(int(b))
    successor = {}
    for vertex, adjacent in enumerate(neighbors):
        if len(adjacent) < 2:
            raise ValueError("isolated or dangling mesh node")
        adjacent.sort(
            key=lambda other: np.arctan2(
                points[other, 1] - points[vertex, 1],
                points[other, 0] - points[vertex, 0],
            )
        )
        for index, previous in enumerate(adjacent):
            successor[previous, vertex] = (vertex, adjacent[index - 1])
    visited: set[tuple[int, int]] = set()
    cells, outside, holes = [], [], []
    for first in successor:
        if first in visited:
            continue
        edge, face = first, []
        while edge not in visited:
            visited.add(edge)
            face.append(edge[0])
            edge = successor[edge]
        if edge != first or len(set(face)) != len(face):
            raise ValueError("non-simple mesh face")
        polygon = points[face]
        area = float(
            0.5
            * np.sum(
                polygon[:, 0] * np.roll(polygon[:, 1], -1)
                - polygon[:, 1] * np.roll(polygon[:, 0], -1)
            )
        )
        if area < 0:
            outside.append(face)
        elif area > 0 and len(face) == 3:
            cells.append(face)
        elif area > 0:
            holes.append(face)
        else:
            raise ValueError("zero-area mesh face")
    if len(outside) != 1 or len(holes) != 1:
        raise ValueError("expected one exterior boundary and one ellipse hole")
    cells = np.asarray(cells, dtype=np.int64)
    recovered = {
        (int(a), int(b)) for cell in cells for a, b in zip(cell, np.roll(cell, -1))
    }
    recovered |= {(b, a) for a, b in recovered.copy()}
    if recovered != original or len(np.unique(cells)) != len(points):
        raise ValueError("recovered cells do not preserve every original node/edge")
    if len(points) - len(edges) // 2 + len(cells) != 0:
        raise ValueError("mesh topology is inconsistent with one obstacle")
    # Geometry-derived labels, not labels claimed to have been released upstream.
    labels = np.zeros(len(points), dtype=np.int32)
    labels[holes[0]] = 2
    boundary = np.asarray(outside[0])
    x, y = points[:, 0], points[:, 1]
    tolerance = 1e-10 * max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
    left = np.isclose(x[boundary], x.min(), rtol=0, atol=tolerance)
    right = np.isclose(x[boundary], x.max(), rtol=0, atol=tolerance)
    wall = np.isclose(y[boundary], y.min(), rtol=0, atol=tolerance) | np.isclose(
        y[boundary], y.max(), rtol=0, atol=tolerance
    )
    if not np.all(left | right | wall):
        raise ValueError("exterior boundary is not the expected rectangle")
    labels[boundary[left]], labels[boundary[right]] = 4, 5
    labels[boundary[wall]] = 6  # Wall precedence at the four corners.
    if not np.any(labels == 4) or not np.any(labels == 5):
        raise ValueError("missing inlet or outlet")
    return {
        "mesh_pos": points,
        "cells": cells,
        "node_type": labels,
        "edges": edges,
        "edge_properties": lengths,
    }
