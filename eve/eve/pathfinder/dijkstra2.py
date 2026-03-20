"""
dijkstra2.py - Dijkstra-based pathfinder with branch tracking

This is an enhanced version of bruteforcebfs.py that:
1. Implements weighted Dijkstra's algorithm (not unweighted BFS)
2. Tracks which branch each edge belongs to
3. Provides access to the path branches (B_path) for reward shaping

Changes from bruteforcebfs.py:
- BPConnection now stores the branch that created the connection
- Uses Dijkstra's algorithm with proper arc-length weighting
- Exposes path_branches: list of branches on the shortest path
- Exposes is_branch_on_path(branch): check if a branch is on the path
"""

from typing import Dict, Generator, List, Tuple, NamedTuple, Set, Optional
from copy import deepcopy
from math import inf
import heapq
import numpy as np

from .pathfinder import Pathfinder
from ..intervention.vesseltree import (
    Branch,
    BranchingPoint,
    find_nearest_branch_to_point,
)

from ..intervention import Intervention
from ..util.coordtransform import tracking3d_to_vessel_cs, vessel_cs_to_tracking3d


def get_length(path: np.ndarray) -> float:
    """Calculate arc length of a polyline."""
    if len(path) < 2:
        return 0.0
    return np.sum(np.linalg.norm(path[:-1] - path[1:], axis=1))


class BPConnection(NamedTuple):
    """Connection between two branching points.
    
    Attributes:
        length: Arc length of the path segment
        points: (N, 3) array of points along the path
        branch: The Branch object that this connection belongs to
    """
    length: float
    points: np.ndarray
    branch: Branch  # NEW: Track which branch this connection belongs to


class DijkstraPathfinder(Pathfinder):
    """
    Pathfinder using Dijkstra's algorithm for weighted shortest path.
    
    Key differences from BruteForceBFS:
    1. Uses arc-length weighted Dijkstra instead of unweighted BFS
    2. Tracks branch identity for each edge
    3. Exposes path_branches for reward shaping
    
    Attributes:
        path_length: Length of shortest path from current position to target
        path_points3d: (N, 3) array of points along the shortest path
        path_branches: List of Branch objects on the shortest path (root→target order)
        path_branch_set: Set of branches on the path (for fast lookup)
    """
    
    def __init__(self, intervention: Intervention):
        self.intervention = intervention
        self.path_length: float = 0.0
        self.path_points3d: np.ndarray = np.empty((0, 3))
        self.path_branching_points3d: np.ndarray = np.empty((0, 3))
        
        # NEW: Track branches on the path
        self.path_branches: List[Branch] = []
        self.path_branch_set: Set[Branch] = set()
        
        # Internal state
        self._branches = None
        self._node_connections: Dict[BranchingPoint, Dict[BranchingPoint, BPConnection]] = {}
        self._search_graph_base = None
        
        # Root branch (branches[0] is the main trunk)
        self._root_branch: Optional[Branch] = None

    def reset(self, episode_nr=0) -> None:
        if self._branches != self.intervention.vessel_tree.branches:
            self._init_vessel_tree()
            self.path_length = 0.0
            self.path_points3d = np.empty((0, 3))
            self.path_branching_points3d = np.empty((0, 3))
            self.path_branches = []
            self.path_branch_set = set()
            self._branches = self.intervention.vessel_tree.branches
            
            # Root branch is the first branch (main trunk/aorta)
            if len(self._branches) > 0:
                self._root_branch = self._branches[0]
        self.step()

    def step(self) -> None:
        fluoro = self.intervention.fluoroscopy
        position = fluoro.tracking3d[0]
        position_vessel_cs = tracking3d_to_vessel_cs(
            position, fluoro.image_rot_zx, fluoro.image_center
        )
        target = self.intervention.target.coordinates3d
        target_vessel_cs = tracking3d_to_vessel_cs(
            target, fluoro.image_rot_zx, fluoro.image_center
        )
        position_branch = find_nearest_branch_to_point(
            position_vessel_cs, self.intervention.vessel_tree
        )
        target_branch = find_nearest_branch_to_point(
            target_vessel_cs, self.intervention.vessel_tree
        )

        (
            path_branching_points,
            self.path_length,
            path_points,
            self.path_branches,  # NEW: Get branches along path
        ) = self._get_shortest_path_dijkstra(
            position_branch, target_branch, position_vessel_cs, target_vessel_cs
        )
        
        # Update path branch set for fast lookup
        self.path_branch_set = set(self.path_branches)
        
        if path_branching_points is not None:
            path_branching_points = [
                branching_point.coordinates for branching_point in path_branching_points
            ]
            path_branching_points = np.array(path_branching_points)
            self.path_branching_points3d = vessel_cs_to_tracking3d(
                path_branching_points,
                fluoro.image_rot_zx,
                fluoro.image_center,
                fluoro.field_of_view,
            )
        else:
            self.path_branching_points3d = None
            
        self.path_points3d = vessel_cs_to_tracking3d(
            path_points,
            fluoro.image_rot_zx,
            fluoro.image_center,
            fluoro.field_of_view,
        )

    def is_branch_on_path(self, branch: Branch) -> bool:
        """Check if a branch is on the current shortest path."""
        return branch in self.path_branch_set
    
    def get_root_branch(self) -> Optional[Branch]:
        """Get the root/main trunk branch (branches[0])."""
        return self._root_branch

    def _init_vessel_tree(self) -> None:
        self._node_connections = self._initialize_node_connections(
            self.intervention.vessel_tree.branching_points
        )
        self._search_graph_base = self._initialize_search_graph_base()

    def _initialize_node_connections(
        self, branching_points: Tuple[BranchingPoint]
    ) -> Dict[BranchingPoint, Dict[BranchingPoint, BPConnection]]:
        """
        Build the graph of connections between branching points.
        
        Each edge stores:
        - length: arc length of the path segment
        - points: polyline points along the path
        - branch: which branch this connection belongs to (NEW)
        """
        node_connections = {}
        for branching_point in branching_points:
            node_connections[branching_point] = {}
            for connection in branching_point.connections:
                for target_branching_point in branching_points:
                    if branching_point == target_branching_point:
                        continue
                    if connection in target_branching_point.connections:
                        points = connection.get_path_along_branch(
                            branching_point.coordinates,
                            target_branching_point.coordinates,
                        )
                        length = get_length(points)

                        # NEW: Store the branch that this connection belongs to
                        node_connections[branching_point][
                            target_branching_point
                        ] = BPConnection(length, points, connection)
        return node_connections

    def _initialize_search_graph_base(
        self,
    ) -> Dict[BranchingPoint, List[BranchingPoint]]:
        _search_graph_base = {}
        for node in self._node_connections:
            _search_graph_base[node] = list(self._node_connections[node].keys())
        return _search_graph_base

    def _get_shortest_path_dijkstra(
        self,
        start_branch: Branch,
        target_branch: Branch,
        start: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[Optional[List[BranchingPoint]], float, np.ndarray, List[Branch]]:
        """
        Find shortest path using Dijkstra's algorithm.
        
        Returns:
            path_branching_points: List of branching points along the path (or None)
            path_length: Arc length of the path
            path_points: (N, 3) array of points along the path
            path_branches: List of Branch objects along the path
        """
        # Special case: start and target on same branch
        if start_branch == target_branch:
            path_points = start_branch.get_path_along_branch(start, target)
            path_length = get_length(path_points)
            return None, path_length, path_points, [start_branch]
        
        # Build search graph with special start/target nodes
        search_graph, start_edges, target_edges = self._create_dijkstra_graph(
            start_branch, target_branch, start, target
        )
        
        # Run Dijkstra's algorithm
        path_nodes, path_length, edge_branches = self._dijkstra(
            search_graph, start_edges, target_edges
        )
        
        if path_nodes is None:
            # No path found
            return None, 0.0, np.empty((1, 3)), []
        
        # Reconstruct full path points and branches
        path_points, path_branches = self._reconstruct_path(
            path_nodes, start, target, start_branch, target_branch, edge_branches
        )
        
        return path_nodes[1:-1] if len(path_nodes) > 2 else None, path_length, path_points, path_branches

    def _create_dijkstra_graph(
        self, start_branch: Branch, target_branch: Branch, start: np.ndarray, target: np.ndarray
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Create graph for Dijkstra with virtual start/target nodes.
        
        Returns:
            search_graph: Dict mapping node -> Dict[neighbor -> (distance, branch)]
            start_edges: Dict[branching_point -> (distance, branch)] from start
            target_edges: Dict[branching_point -> (distance, branch)] to target
        """
        # Build weighted graph: node -> {neighbor: (distance, branch)}
        search_graph = {}
        for node, neighbors in self._node_connections.items():
            search_graph[node] = {}
            for neighbor, conn in neighbors.items():
                search_graph[node][neighbor] = (conn.length, conn.branch)
        
        # Find connections from start position to branching points
        start_edges = {}
        for branching_point in self.intervention.vessel_tree.branching_points:
            if start_branch in branching_point.connections:
                points = start_branch.get_path_along_branch(start, branching_point.coordinates)
                length = get_length(points)
                start_edges[branching_point] = (length, start_branch)
        
        # Find connections from branching points to target
        target_edges = {}
        for branching_point in self.intervention.vessel_tree.branching_points:
            if target_branch in branching_point.connections:
                points = target_branch.get_path_along_branch(branching_point.coordinates, target)
                length = get_length(points)
                target_edges[branching_point] = (length, target_branch)
        
        return search_graph, start_edges, target_edges

    def _dijkstra(
        self,
        graph: Dict[BranchingPoint, Dict[BranchingPoint, Tuple[float, Branch]]],
        start_edges: Dict[BranchingPoint, Tuple[float, Branch]],
        target_edges: Dict[BranchingPoint, Tuple[float, Branch]],
    ) -> Tuple[Optional[List], float, Dict]:
        """
        Run Dijkstra's algorithm.
        
        Returns:
            path: List of nodes from start to target (including virtual nodes)
            total_distance: Total path length
            edge_branches: Dict mapping (from_node, to_node) -> branch
        """
        # Distance from start
        dist = {"start": 0.0}
        for node in graph:
            dist[node] = inf
        dist["target"] = inf
        
        # Predecessor for path reconstruction
        pred = {}
        
        # Edge branches for tracking which branch each edge uses
        edge_branches = {}
        
        # Priority queue: (distance, node)
        pq = [(0.0, "start")]
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            visited.add(u)
            
            if u == "target":
                break
            
            # Get neighbors
            if u == "start":
                neighbors = {bp: (length, branch) for bp, (length, branch) in start_edges.items()}
            else:
                neighbors = graph.get(u, {})
                # Also check if we can reach target from this node
                if u in target_edges:
                    length, branch = target_edges[u]
                    neighbors = dict(neighbors)  # Copy to avoid mutation
                    neighbors["target"] = (length, branch)
            
            for v, (edge_length, branch) in neighbors.items():
                if v in visited:
                    continue
                    
                new_dist = d + edge_length
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    pred[v] = u
                    edge_branches[(u, v)] = branch
                    heapq.heappush(pq, (new_dist, v))
        
        # Reconstruct path
        if "target" not in pred:
            return None, 0.0, {}
        
        path = []
        node = "target"
        while node is not None:
            path.append(node)
            node = pred.get(node)
        path.reverse()
        
        return path, dist["target"], edge_branches

    def _reconstruct_path(
        self,
        path_nodes: List,
        start: np.ndarray,
        target: np.ndarray,
        start_branch: Branch,
        target_branch: Branch,
        edge_branches: Dict,
    ) -> Tuple[np.ndarray, List[Branch]]:
        """
        Reconstruct the full path points and branch list.
        
        Returns:
            path_points: (N, 3) array of all points along the path
            path_branches: List of branches in order from start to target
        """
        if len(path_nodes) < 2:
            return np.array([start]), [start_branch]
        
        path_points = [start]
        path_branches = []
        
        # Add start branch
        path_branches.append(start_branch)
        
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            
            if u == "start":
                # Edge from start to first branching point
                bp = v
                points = start_branch.get_path_along_branch(start, bp.coordinates)
                path_points.extend(points[1:])  # Skip start (already added)
                
            elif v == "target":
                # Edge from last branching point to target
                bp = u
                points = target_branch.get_path_along_branch(bp.coordinates, target)
                path_points.extend(points[1:])
                if target_branch not in path_branches:
                    path_branches.append(target_branch)
                    
            else:
                # Edge between branching points
                conn = self._node_connections[u][v]
                path_points.extend(conn.points[1:])
                if conn.branch not in path_branches:
                    path_branches.append(conn.branch)
        
        return np.array(path_points), path_branches
