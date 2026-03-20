"""
fixedpath.py - Fixed path pathfinder (computes path once at reset)

This pathfinder computes the path from INSERTION POINT to TARGET once at reset,
then keeps it fixed for the entire episode. The step() method is a no-op.

This is efficient for waypoint-based rewards where:
- The path root→target is fixed (target doesn't change during episode)
- We only need to track which waypoint the tip is nearest to
- No need to recompute Dijkstra every step

Use this instead of DijkstraPathfinder when using CenterlineWaypointProgress reward.
"""

from typing import Dict, List, Tuple, NamedTuple, Set, Optional
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
    """Connection between two branching points."""
    length: float
    points: np.ndarray
    branch: Branch


class FixedPathfinder(Pathfinder):
    """
    Pathfinder that computes the path ONCE at reset (from insertion point to target).
    
    Unlike DijkstraPathfinder which recomputes every step, this:
    1. Computes path from insertion point → target at reset()
    2. step() is a no-op (path is fixed)
    3. Much more efficient for waypoint-based rewards
    
    Attributes:
        path_length: Length of the fixed path
        path_points3d: (N, 3) array of points along the path (in tracking3d coords)
        path_points_vessel_cs: (N, 3) array of points in vessel coordinate system
        path_branches: List of Branch objects on the path
        path_branch_set: Set of branches for fast lookup
    """
    
    def __init__(self, intervention: Intervention):
        self.intervention = intervention
        self.path_length: float = 0.0
        self.path_points3d: np.ndarray = np.empty((0, 3))
        self.path_points_vessel_cs: np.ndarray = np.empty((0, 3))
        self.path_branching_points3d: np.ndarray = np.empty((0, 3))
        
        # Track branches on the path
        self.path_branches: List[Branch] = []
        self.path_branch_set: Set[Branch] = set()
        
        # Internal state
        self._branches = None
        self._node_connections: Dict[BranchingPoint, Dict[BranchingPoint, BPConnection]] = {}
        self._search_graph_base = None
        self._root_branch: Optional[Branch] = None
        
        # Flag to track if path is computed
        self._path_computed = False

    def reset(self, episode_nr=0) -> None:
        """
        Compute the fixed path from insertion point to target.
        
        This is called once per episode when the target is set.
        """
        # Initialize vessel tree graph if needed
        if self._branches != self.intervention.vessel_tree.branches:
            self._init_vessel_tree()
            self._branches = self.intervention.vessel_tree.branches
            if len(self._branches) > 0:
                self._root_branch = self._branches[0]
        
        # Compute path from insertion point to target
        self._compute_fixed_path()
        self._path_computed = True

    def step(self) -> None:
        """
        No-op. The path is fixed for the episode.
        
        The path was computed at reset() and doesn't change.
        """
        # Nothing to do - path is fixed
        pass

    def _compute_fixed_path(self) -> None:
        """
        Compute the path from insertion point to target.
        
        Uses the INSERTION POINT (start of root branch) as the starting point,
        not the current tip position.
        """
        fluoro = self.intervention.fluoroscopy
        vessel_tree = self.intervention.vessel_tree
        
        # Get insertion point (start of the vessel tree / root branch)
        # This is where the device starts, not where the tip currently is
        insertion_point = np.array(vessel_tree.insertion.position)  # Ensure it's a numpy array
        insertion_vessel_cs = insertion_point  # Already in vessel CS
        
        # Get target position
        target = self.intervention.target.coordinates3d
        target_vessel_cs = tracking3d_to_vessel_cs(
            target, fluoro.image_rot_zx, fluoro.image_center
        )
        
        # Find branches for start and target
        start_branch = find_nearest_branch_to_point(insertion_vessel_cs, vessel_tree)
        target_branch = find_nearest_branch_to_point(target_vessel_cs, vessel_tree)
        
        # Compute shortest path
        (
            path_branching_points,
            self.path_length,
            path_points,
            self.path_branches,
        ) = self._get_shortest_path_dijkstra(
            start_branch, target_branch, insertion_vessel_cs, target_vessel_cs
        )
        
        # Update path branch set
        self.path_branch_set = set(self.path_branches)
        
        # Store path points in vessel CS
        self.path_points_vessel_cs = path_points
        
        # Convert to tracking3d
        if path_branching_points is not None:
            bp_coords = np.array([bp.coordinates for bp in path_branching_points])
            self.path_branching_points3d = vessel_cs_to_tracking3d(
                bp_coords,
                fluoro.image_rot_zx,
                fluoro.image_center,
                fluoro.field_of_view,
            )
        else:
            self.path_branching_points3d = np.empty((0, 3))
        
        self.path_points3d = vessel_cs_to_tracking3d(
            path_points,
            fluoro.image_rot_zx,
            fluoro.image_center,
            fluoro.field_of_view,
        )

    def is_branch_on_path(self, branch: Branch) -> bool:
        """Check if a branch is on the fixed path."""
        return branch in self.path_branch_set

    def get_root_branch(self) -> Optional[Branch]:
        """Get the root/main trunk branch."""
        return self._root_branch

    # ========== Internal methods (same as DijkstraPathfinder) ==========

    def _init_vessel_tree(self) -> None:
        self._node_connections = self._initialize_node_connections(
            self.intervention.vessel_tree.branching_points
        )
        self._search_graph_base = self._initialize_search_graph_base()

    def _initialize_node_connections(
        self, branching_points: Tuple[BranchingPoint]
    ) -> Dict[BranchingPoint, Dict[BranchingPoint, BPConnection]]:
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
                        node_connections[branching_point][
                            target_branching_point
                        ] = BPConnection(length, points, connection)
        return node_connections

    def _initialize_search_graph_base(self) -> Dict[BranchingPoint, List[BranchingPoint]]:
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
        """Find shortest path using Dijkstra's algorithm."""
        
        # Special case: start and target on same branch
        if start_branch == target_branch:
            path_points = start_branch.get_path_along_branch(start, target)
            path_length = get_length(path_points)
            return None, path_length, path_points, [start_branch]
        
        # Build search graph
        search_graph, start_edges, target_edges = self._create_dijkstra_graph(
            start_branch, target_branch, start, target
        )
        
        # Run Dijkstra
        path_nodes, path_length, edge_branches = self._dijkstra(
            search_graph, start_edges, target_edges
        )
        
        if path_nodes is None:
            return None, 0.0, np.empty((1, 3)), []
        
        # Reconstruct path
        path_points, path_branches = self._reconstruct_path(
            path_nodes, start, target, start_branch, target_branch, edge_branches
        )
        
        bp_list = path_nodes[1:-1] if len(path_nodes) > 2 else None
        return bp_list, path_length, path_points, path_branches

    def _create_dijkstra_graph(
        self, start_branch: Branch, target_branch: Branch, start: np.ndarray, target: np.ndarray
    ) -> Tuple[Dict, Dict, Dict]:
        search_graph = {}
        for node, neighbors in self._node_connections.items():
            search_graph[node] = {}
            for neighbor, conn in neighbors.items():
                search_graph[node][neighbor] = (conn.length, conn.branch)
        
        start_edges = {}
        for branching_point in self.intervention.vessel_tree.branching_points:
            if start_branch in branching_point.connections:
                points = start_branch.get_path_along_branch(start, branching_point.coordinates)
                length = get_length(points)
                start_edges[branching_point] = (length, start_branch)
        
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
        dist = {"start": 0.0}
        for node in graph:
            dist[node] = inf
        dist["target"] = inf
        
        pred = {}
        edge_branches = {}
        pq = [(0.0, "start")]
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            visited.add(u)
            
            if u == "target":
                break
            
            if u == "start":
                neighbors = {bp: (length, branch) for bp, (length, branch) in start_edges.items()}
            else:
                neighbors = graph.get(u, {})
                if u in target_edges:
                    length, branch = target_edges[u]
                    neighbors = dict(neighbors)
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
        if len(path_nodes) < 2:
            return np.array([start]), [start_branch]
        
        path_points = [start]
        path_branches = [start_branch]
        
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            
            if u == "start":
                bp = v
                points = start_branch.get_path_along_branch(start, bp.coordinates)
                path_points.extend(points[1:])
                
            elif v == "target":
                bp = u
                points = target_branch.get_path_along_branch(bp.coordinates, target)
                path_points.extend(points[1:])
                if target_branch not in path_branches:
                    path_branches.append(target_branch)
                    
            else:
                conn = self._node_connections[u][v]
                path_points.extend(conn.points[1:])
                if conn.branch not in path_branches:
                    path_branches.append(conn.branch)
        
        return np.array(path_points), path_branches
