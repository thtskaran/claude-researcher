"""Knowledge graph visualization using Pyvis and Mermaid."""

import json
from pathlib import Path
from typing import Optional

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

from .store import HybridKnowledgeGraphStore


class KnowledgeGraphVisualizer:
    """Visualize knowledge graph using Pyvis and Mermaid diagrams."""

    # Color scheme by entity type
    ENTITY_COLORS = {
        'CONCEPT': '#6366f1',      # Indigo
        'CLAIM': '#f59e0b',        # Amber
        'EVIDENCE': '#10b981',     # Emerald
        'METHOD': '#8b5cf6',       # Violet
        'METRIC': '#ef4444',       # Red
        'TECHNOLOGY': '#3b82f6',   # Blue
        'SOURCE': '#6b7280',       # Gray
        'PERSON': '#ec4899',       # Pink
        'ORGANIZATION': '#14b8a6', # Teal
        'AUTHOR': '#f97316',       # Orange
        'LOCATION': '#84cc16',     # Lime
        'DATE': '#a855f7',         # Purple
        'DEFAULT': '#9ca3af',      # Gray
    }

    def __init__(self, kg_store: HybridKnowledgeGraphStore):
        self.store = kg_store

    def create_interactive_graph(
        self,
        output_path: str = "knowledge_graph.html",
        height: str = "800px",
        width: str = "100%",
        show_physics_controls: bool = True
    ) -> Optional[str]:
        """Create interactive Pyvis visualization with fast stabilization.

        Uses Barnes-Hut physics (same as before) but with stabilization settings
        that compute the layout quickly on load and then disable physics for
        smooth interaction.

        Args:
            output_path: Path to save HTML file
            height: Height of the visualization
            width: Width of the visualization
            show_physics_controls: Whether to show physics controls

        Returns:
            Path to the generated file, or None if Pyvis not available
        """
        if not HAS_PYVIS or not HAS_NETWORKX:
            return None

        if self.store.graph is None or len(self.store.graph) == 0:
            return None

        # Create Pyvis network
        net = Network(
            height=height,
            width=width,
            directed=True,
            notebook=False,
            bgcolor="#ffffff",
            font_color="#333333"
        )

        # Configure physics - same Barnes-Hut as before
        net.barnes_hut(
            gravity=-80000,
            central_gravity=0.3,
            spring_length=200,
            spring_strength=0.001,
            damping=0.09,
            overlap=0.1
        )

        if show_physics_controls:
            net.show_buttons(filter_=['physics'])

        # Add nodes
        for node_id, data in self.store.graph.nodes(data=True):
            entity_type = data.get('entity_type', 'DEFAULT')
            color = self.ENTITY_COLORS.get(entity_type, self.ENTITY_COLORS['DEFAULT'])

            # Size based on degree centrality
            size = 10 + (self.store.graph.degree(node_id) * 3)

            # Build hover title
            title = self._build_node_title(node_id, data)

            net.add_node(
                node_id,
                label=data.get('name', node_id)[:30],
                title=title,
                color=color,
                size=min(size, 50),  # Cap size
                shape='dot' if entity_type != 'CLAIM' else 'diamond'
            )

        # Add edges
        for u, v, data in self.store.graph.edges(data=True):
            predicate = data.get('predicate', '')
            confidence = data.get('confidence', 1.0)

            # Color edges by type
            edge_color = self._get_edge_color(predicate)

            net.add_edge(
                u, v,
                title=predicate,
                label=predicate[:20] if len(predicate) <= 20 else predicate[:17] + '...',
                color=edge_color,
                width=1 + (confidence * 2),
                arrows='to'
            )

        # Save to file
        output_path = str(output_path)
        net.save_graph(output_path)

        # Post-process HTML to add stabilization and auto-disable physics
        num_nodes = len(self.store.graph)
        self._add_stabilization_handler(output_path, num_nodes)

        return output_path

    def _add_stabilization_handler(self, html_path: str, num_nodes: int) -> None:
        """Add JavaScript to configure stabilization and disable physics after completion."""
        try:
            with open(html_path, 'r') as f:
                html_content = f.read()

            # Adjust iterations based on graph size
            if num_nodes < 100:
                iterations = 200
            elif num_nodes < 500:
                iterations = 150
            elif num_nodes < 1000:
                iterations = 100
            else:
                iterations = 50

            # JavaScript to configure stabilization and disable physics after
            stabilization_script = f"""
    <script type="text/javascript">
        // Configure stabilization for faster loading
        network.setOptions({{
            physics: {{
                stabilization: {{
                    enabled: true,
                    iterations: {iterations},
                    updateInterval: 25,
                    fit: true
                }}
            }}
        }});

        // Disable physics after stabilization completes for smooth interaction
        network.once('stabilizationIterationsDone', function() {{
            network.setOptions({{ physics: {{ enabled: false }} }});
            console.log('Physics disabled after stabilization');
        }});

        // Start stabilization
        network.stabilize({iterations});
    </script>
</body>"""

            # Insert before closing body tag
            html_content = html_content.replace('</body>', stabilization_script)

            with open(html_path, 'w') as f:
                f.write(html_content)
        except Exception:
            pass  # If post-processing fails, the graph still works with physics

    def create_mermaid_diagram(self, max_nodes: int = 30) -> str:
        """Generate Mermaid diagram syntax for embedding in reports.

        Args:
            max_nodes: Maximum number of nodes to include

        Returns:
            Mermaid diagram as a string
        """
        if not HAS_NETWORKX or self.store.graph is None or len(self.store.graph) == 0:
            return "```mermaid\ngraph TD\n    empty[No data yet]\n```"

        lines = ["```mermaid", "graph TD"]

        # Get most important nodes if too many
        if len(self.store.graph) > max_nodes:
            try:
                pagerank = nx.pagerank(self.store.graph)
                top_nodes = sorted(
                    pagerank.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:max_nodes]
                included_nodes = {n[0] for n in top_nodes}
            except Exception:
                included_nodes = set(list(self.store.graph.nodes())[:max_nodes])
        else:
            included_nodes = set(self.store.graph.nodes())

        # Add nodes with styling
        for node_id in included_nodes:
            data = self.store.graph.nodes[node_id]
            name = data.get('name', node_id).replace('"', "'").replace('\n', ' ')[:40]
            entity_type = data.get('entity_type', 'DEFAULT')

            # Make ID safe for Mermaid
            safe_id = self._make_safe_id(node_id)

            # Mermaid node syntax
            if entity_type == 'CLAIM':
                lines.append(f'    {safe_id}{{"{name}"}}')  # Diamond
            elif entity_type == 'EVIDENCE':
                lines.append(f'    {safe_id}(["{name}"])')  # Stadium
            elif entity_type == 'METHOD':
                lines.append(f'    {safe_id}[/"{name}"/]')  # Parallelogram
            else:
                lines.append(f'    {safe_id}["{name}"]')    # Rectangle

        # Add edges
        for u, v, data in self.store.graph.edges(data=True):
            if u in included_nodes and v in included_nodes:
                predicate = data.get('predicate', '').replace('"', "'")[:15]
                safe_u = self._make_safe_id(u)
                safe_v = self._make_safe_id(v)
                lines.append(f'    {safe_u} -->|{predicate}| {safe_v}')

        # Add styling
        lines.extend([
            "",
            "    classDef concept fill:#6366f1,color:white",
            "    classDef claim fill:#f59e0b,color:white",
            "    classDef evidence fill:#10b981,color:white",
            "    classDef method fill:#8b5cf6,color:white",
        ])

        lines.append("```")
        return "\n".join(lines)

    def create_summary_stats_card(self) -> str:
        """Generate a text summary card for reports."""
        stats = self.store.get_stats()

        # Count by type
        type_counts = stats.get('entity_types', {})

        card = f"""
+--------------------------------------------+
|          KNOWLEDGE GRAPH STATS             |
+--------------------------------------------+
|  Entities: {stats['num_entities']:<30}|
|  Relations: {stats['num_relations']:<29}|
|  Components: {stats.get('num_components', 0):<28}|
|  Density: {stats.get('density', 0):.3f}                           |
+--------------------------------------------+
|  By Type:                                  |"""

        for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            card += f"\n|    {etype}: {count:<33}|"

        card += "\n+--------------------------------------------+"
        return card

    def _build_node_title(self, node_id: str, data: dict) -> str:
        """Build HTML hover title for a node."""
        parts = [
            f"<b>{data.get('name', node_id)}</b>",
            f"Type: {data.get('entity_type', 'Unknown')}",
            f"Confidence: {data.get('confidence', 1.0):.2f}",
        ]

        aliases = data.get('aliases', [])
        if aliases:
            parts.append(f"Aliases: {', '.join(aliases[:3])}")

        sources = data.get('sources', [])
        if sources:
            parts.append(f"Sources: {len(sources)}")

        return "<br>".join(parts)

    def _get_edge_color(self, predicate: str) -> str:
        """Get edge color based on predicate type."""
        pred_lower = predicate.lower()

        if any(p in pred_lower for p in ['supports', 'evidence', 'confirms']):
            return '#10b981'  # Green
        elif any(p in pred_lower for p in ['contradicts', 'refutes', 'disputes']):
            return '#ef4444'  # Red
        elif any(p in pred_lower for p in ['causes', 'leads to', 'results']):
            return '#f59e0b'  # Amber
        elif any(p in pred_lower for p in ['is_a', 'type of', 'instance']):
            return '#6366f1'  # Indigo
        else:
            return '#9ca3af'  # Gray

    def _make_safe_id(self, node_id: str) -> str:
        """Make an ID safe for Mermaid."""
        # Replace problematic characters
        safe = node_id.replace('-', '_').replace(' ', '_').replace('.', '_')
        # Ensure it starts with a letter
        if safe and not safe[0].isalpha():
            safe = 'n_' + safe
        return safe[:20]  # Limit length

    def create_static_svg(
        self,
        output_path: str = "knowledge_graph.svg",
        max_nodes: int = 100,
        figsize: tuple[int, int] = (16, 12)
    ) -> Optional[str]:
        """Create a static SVG visualization using matplotlib.

        Useful for embedding in documents where JavaScript isn't available.

        Args:
            output_path: Path to save SVG file
            max_nodes: Maximum nodes to display (uses PageRank to select most important)
            figsize: Figure size in inches (width, height)

        Returns:
            Path to the generated file, or None if matplotlib not available
        """
        if not HAS_NETWORKX or self.store.graph is None or len(self.store.graph) == 0:
            return None

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        graph = self.store.graph

        # If too many nodes, select most important by PageRank
        if len(graph) > max_nodes:
            try:
                pagerank = nx.pagerank(graph)
                top_nodes = sorted(
                    pagerank.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:max_nodes]
                included_nodes = {n[0] for n in top_nodes}
                graph = graph.subgraph(included_nodes).copy()
            except Exception:
                # Fallback: just take first N nodes
                included_nodes = set(list(graph.nodes())[:max_nodes])
                graph = graph.subgraph(included_nodes).copy()

        # Compute layout using NetworkX
        num_nodes = len(graph)
        try:
            if num_nodes < 100:
                positions = nx.kamada_kawai_layout(graph)
            else:
                k = 2.0 / (num_nodes ** 0.5)
                positions = nx.spring_layout(graph, k=k, iterations=100, seed=42)
        except Exception:
            positions = nx.circular_layout(graph)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.axis('off')

        # Get node colors
        node_colors = []
        for node_id in graph.nodes():
            entity_type = graph.nodes[node_id].get('entity_type', 'DEFAULT')
            color = self.ENTITY_COLORS.get(entity_type, self.ENTITY_COLORS['DEFAULT'])
            node_colors.append(color)

        # Get node sizes based on degree
        node_sizes = []
        for node_id in graph.nodes():
            size = 100 + (graph.degree(node_id) * 50)
            node_sizes.append(min(size, 1000))

        # Draw the graph
        nx.draw_networkx_edges(
            graph, positions, ax=ax,
            alpha=0.5, edge_color='#9ca3af',
            arrows=True, arrowsize=15,
            connectionstyle="arc3,rad=0.1"
        )

        nx.draw_networkx_nodes(
            graph, positions, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9
        )

        # Add labels
        labels = {
            node_id: graph.nodes[node_id].get('name', node_id)[:20]
            for node_id in graph.nodes()
        }
        nx.draw_networkx_labels(
            graph, positions, labels, ax=ax,
            font_size=8, font_weight='bold'
        )

        # Save as SVG
        plt.tight_layout()
        plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
        plt.close()

        return output_path
