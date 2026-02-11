"use client";

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

/* ── Types ──────────────────────────────────────────────── */
interface KGEntity {
    id: string;
    name: string;
    entity_type: string;
    properties: Record<string, unknown>;
    created_at: string;
}

interface KGRelation {
    id: string;
    subject_id: string;
    predicate: string;
    object_id: string;
    confidence: number;
    subject_name?: string;
    subject_type?: string;
    object_name?: string;
    object_type?: string;
}

interface KGContradiction {
    id: string;
    relation1_id: string;
    relation2_id: string;
    contradiction_type: string;
    description: string;
    severity: string;
}

/* ── Entity style map ──────────────────────────────────── */
const entityTypeConfig: Record<string, { bg: string; border: string; dimBg: string; label: string }> = {
    CLAIM:        { bg: "#d49e6e", border: "#b87c4c", dimBg: "rgba(212,158,110,0.15)", label: "Claim" },
    CONCEPT:      { bg: "#a594be", border: "#8b7aa5", dimBg: "rgba(165,148,190,0.15)", label: "Concept" },
    SOURCE:       { bg: "#8b8578", border: "#6e695e", dimBg: "rgba(139,133,120,0.15)", label: "Source" },
    EVIDENCE:     { bg: "#82af78", border: "#5e8c54", dimBg: "rgba(130,175,120,0.15)", label: "Evidence" },
    TECHNOLOGY:   { bg: "#7ab0c4", border: "#5a95ab", dimBg: "rgba(122,176,196,0.15)", label: "Technology" },
    METHOD:       { bg: "#a594be", border: "#8b7aa5", dimBg: "rgba(165,148,190,0.15)", label: "Method" },
    METRIC:       { bg: "#d77369", border: "#c05a50", dimBg: "rgba(215,115,105,0.15)", label: "Metric" },
    PERSON:       { bg: "#c9917a", border: "#b87c4c", dimBg: "rgba(201,145,122,0.15)", label: "Person" },
    ORGANIZATION: { bg: "#a8bba3", border: "#8aa383", dimBg: "rgba(168,187,163,0.15)", label: "Organization" },
    QUOTE:        { bg: "#d7b464", border: "#c09a3e", dimBg: "rgba(215,180,100,0.15)", label: "Quote" },
};
const defaultEntityCfg = { bg: "#8b8578", border: "#6e695e", dimBg: "rgba(139,133,120,0.15)", label: "Other" };

const predicateColors: Record<string, string> = {
    supports: "#82af78",
    contradicts: "#d77369",
    related: "#7ab0c4",
    related_to: "#7ab0c4",
    is_a: "#a594be",
    part_of: "#a594be",
    causes: "#d7b464",
    cites: "#8b8578",
    implements: "#a594be",
    outperforms: "#82af78",
    similar_to: "#7ab0c4",
    alternative_to: "#c9917a",
    authored_by: "#8b8578",
    mentioned_in: "#8b8578",
};
const defaultEdgeColor = "#3c3a34";

/* ── Graph theme (dark-only) ───────────────────────────── */
const THEME = {
    fontColor: "#f0ece4",
    fontBg: "rgba(28, 26, 22, 0.85)",
    edgeFontColor: "#b4aa9e",
    pageBg: "#1c1a16",
    dimmedFont: "rgba(240,236,228,0.25)",
    dimmedEdge: "rgba(55,52,44,0.3)",
};

/* ── Degree centrality helper ──────────────────────────── */
function computeDegrees(entities: KGEntity[], relations: KGRelation[]): Map<string, number> {
    const deg = new Map<string, number>();
    entities.forEach(e => deg.set(e.id, 0));
    relations.forEach(r => {
        deg.set(r.subject_id, (deg.get(r.subject_id) || 0) + 1);
        deg.set(r.object_id, (deg.get(r.object_id) || 0) + 1);
    });
    return deg;
}

/* ── Main Component ────────────────────────────────────── */
export default function KnowledgeGraphPage() {
    const params = useParams();
    const sessionId = params.id as string;

    const [entities, setEntities] = useState<KGEntity[]>([]);
    const [relations, setRelations] = useState<KGRelation[]>([]);
    const [contradictions, setContradictions] = useState<KGContradiction[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [nodeTypeFilter, setNodeTypeFilter] = useState<string>("all");
    const [confidenceFilter, setConfidenceFilter] = useState(0);
    const [searchQuery, setSearchQuery] = useState("");
    const [searchActive, setSearchActive] = useState(false);
    const [zoomLevel, setZoomLevel] = useState(100);
    const [stabilizing, setStabilizing] = useState(false);

    const containerRef = useRef<HTMLDivElement>(null);
    const networkRef = useRef<any>(null);
    const nodesDatasetRef = useRef<any>(null);
    const edgesDatasetRef = useRef<any>(null);
    const allNodesRef = useRef<Record<string, any>>({});
    const allEdgesRef = useRef<Record<string, any>>({});
    const highlightActiveRef = useRef(false);

    /* ── Fetch data ─────────────────────────────────────── */
    const fetchGraph = async () => {
        setLoading(true);
        setError("");
        try {
            const res = await fetch(`/api/sessions/${sessionId}/knowledge/graph?limit=500`);
            if (!res.ok) throw new Error("Failed to load knowledge graph");
            const data = await res.json();
            setEntities(data.entities || []);
            setRelations(data.relations || []);
            setContradictions(data.contradictions || []);
        } catch {
            setError("Unable to load knowledge graph data");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchGraph(); }, [sessionId]);

    /* ── Filtering ──────────────────────────────────────── */
    const filteredEntities = useMemo(() => {
        return entities.filter(e => {
            if (nodeTypeFilter !== "all" && e.entity_type !== nodeTypeFilter) return false;
            if (confidenceFilter > 0) {
                const conf = (e.properties as Record<string, number>)?.confidence ?? 1;
                if (conf < confidenceFilter / 100) return false;
            }
            return true;
        });
    }, [entities, nodeTypeFilter, confidenceFilter]);

    const entityTypes = useMemo(() => {
        const types = new Set(entities.map(e => e.entity_type));
        return Array.from(types).sort();
    }, [entities]);

    const selectedNode = useMemo(() => {
        if (!selectedId) return null;
        return entities.find(e => e.id === selectedId) || null;
    }, [selectedId, entities]);

    const connectedEdges = useMemo(() => {
        if (!selectedId) return [];
        return relations.filter(e => e.subject_id === selectedId || e.object_id === selectedId);
    }, [selectedId, relations]);

    /* ── Search ─────────────────────────────────────────── */
    const searchResults = useMemo(() => {
        if (!searchQuery.trim()) return [];
        const q = searchQuery.toLowerCase();
        return filteredEntities
            .filter(n => n.name.toLowerCase().includes(q))
            .slice(0, 8);
    }, [searchQuery, filteredEntities]);

    const centerOnNode = useCallback((nodeId: string) => {
        if (networkRef.current) {
            const pos = networkRef.current.getPositions([nodeId]);
            if (pos[nodeId]) {
                networkRef.current.moveTo({
                    position: pos[nodeId],
                    scale: 1.8,
                    animation: { duration: 600, easingFunction: "easeInOutQuad" as any }
                });
            }
            networkRef.current.selectNodes([nodeId], true);
            setSelectedId(nodeId);
            setSearchActive(false);
            setSearchQuery("");
        }
    }, []);

    const fitToView = useCallback(() => {
        if (networkRef.current) {
            networkRef.current.fit({
                animation: { duration: 600, easingFunction: "easeInOutQuad" as any },
                maxZoomLevel: 1.5
            });
        }
    }, []);

    /* ── Hover-to-dim ──────────────────────────────────── */
    const highlightConnected = useCallback((hoveredNodeId: string) => {
        if (!networkRef.current || !nodesDatasetRef.current || !edgesDatasetRef.current) return;

        highlightActiveRef.current = true;
        const connectedNodes = new Set<string>(networkRef.current.getConnectedNodes(hoveredNodeId));
        connectedNodes.add(hoveredNodeId);

        const connectedEdgeIds = new Set<string>(networkRef.current.getConnectedEdges(hoveredNodeId));

        // Dim non-connected nodes
        const nodeUpdates: any[] = [];
        const allNodes = allNodesRef.current;
        for (const nodeId in allNodes) {
            const node = allNodes[nodeId];
            if (connectedNodes.has(nodeId)) {
                nodeUpdates.push({
                    id: nodeId,
                    opacity: 1.0,
                    font: { color: THEME.fontColor, background: THEME.fontBg },
                });
            } else {
                nodeUpdates.push({
                    id: nodeId,
                    opacity: 0.15,
                    font: { color: THEME.dimmedFont, background: "transparent" },
                });
            }
        }
        nodesDatasetRef.current.update(nodeUpdates);

        // Dim non-connected edges
        const edgeUpdates: any[] = [];
        const allEdges = allEdgesRef.current;
        for (const edgeId in allEdges) {
            const edge = allEdges[edgeId];
            if (connectedEdgeIds.has(edgeId)) {
                const color = edge._originalColor || defaultEdgeColor;
                edgeUpdates.push({
                    id: edgeId,
                    color: { color, highlight: color, hover: color, opacity: 1.0 },
                    width: edge._originalWidth || 1.5,
                });
            } else {
                edgeUpdates.push({
                    id: edgeId,
                    color: { color: THEME.dimmedEdge, highlight: THEME.dimmedEdge, hover: THEME.dimmedEdge, opacity: 0.1 },
                    width: 0.5,
                });
            }
        }
        edgesDatasetRef.current.update(edgeUpdates);
    }, []);

    const unhighlightAll = useCallback(() => {
        if (!highlightActiveRef.current || !nodesDatasetRef.current || !edgesDatasetRef.current) return;
        highlightActiveRef.current = false;

        const nodeUpdates: any[] = [];
        const allNodes = allNodesRef.current;
        for (const nodeId in allNodes) {
            nodeUpdates.push({
                id: nodeId,
                opacity: 1.0,
                font: { color: THEME.fontColor, background: THEME.fontBg },
            });
        }
        nodesDatasetRef.current.update(nodeUpdates);

        const edgeUpdates: any[] = [];
        const allEdges = allEdgesRef.current;
        for (const edgeId in allEdges) {
            const edge = allEdges[edgeId];
            const color = edge._originalColor || defaultEdgeColor;
            edgeUpdates.push({
                id: edgeId,
                color: { color: color + "80", highlight: color, hover: color, opacity: 1.0 },
                width: edge._originalWidth || 1.5,
            });
        }
        edgesDatasetRef.current.update(edgeUpdates);
    }, []);

    /* ── Initialize vis-network ─────────────────────────── */
    useEffect(() => {
        if (!containerRef.current || loading || entities.length === 0) return;

        import("vis-network").then(({ Network }) => {
            import("vis-data").then(({ DataSet }) => {
                if (!containerRef.current) return;

                setStabilizing(true);

                // Compute degree centrality
                const degrees = computeDegrees(filteredEntities, relations);
                const maxDegree = Math.max(1, ...Array.from(degrees.values()));

                // Build filtered entity ID set
                const filteredIds = new Set(filteredEntities.map(e => e.id));

                // Prepare nodes with degree-based sizing
                const nodes = filteredEntities.map(e => {
                    const cfg = entityTypeConfig[e.entity_type] || defaultEntityCfg;
                    const degree = degrees.get(e.id) || 0;

                    // Size: 8 (isolated) to 40 (highest degree), logarithmic scale
                    const normalizedDeg = maxDegree > 0 ? degree / maxDegree : 0;
                    const size = 8 + Math.log(1 + normalizedDeg * 9) / Math.log(10) * 32;

                    // Progressive labels: only show labels for nodes with enough connections
                    const showLabel = degree >= 1;

                    return {
                        id: e.id,
                        label: showLabel ? (e.name.length > 30 ? e.name.slice(0, 27) + "\u2026" : e.name) : undefined,
                        title: `${e.name}\n${e.entity_type} \u00b7 ${degree} connections`,
                        color: {
                            background: cfg.bg,
                            border: cfg.border,
                            highlight: { background: cfg.bg, border: "#f0ece4" },
                            hover: { background: cfg.bg, border: "#f0ece4" },
                        },
                        shape: "dot",
                        size,
                        mass: 1 + degree * 0.2,
                        font: {
                            color: THEME.fontColor,
                            size: Math.max(10, Math.min(16, 10 + degree * 0.5)),
                            face: "Figtree, system-ui, sans-serif",
                            background: THEME.fontBg,
                            strokeWidth: 0,
                        },
                        borderWidth: 2,
                        borderWidthSelected: 3,
                        opacity: 1.0,
                        _degree: degree,
                        _entityType: e.entity_type,
                        _fullName: e.name,
                    };
                });

                // Prepare edges
                const edges = relations
                    .filter(r => filteredIds.has(r.subject_id) && filteredIds.has(r.object_id))
                    .map(r => {
                        const color = predicateColors[r.predicate] || defaultEdgeColor;
                        const width = 1 + r.confidence * 1.5;
                        return {
                            id: r.id,
                            from: r.subject_id,
                            to: r.object_id,
                            title: r.predicate.replace(/_/g, " "),
                            color: {
                                color: color + "80",
                                highlight: color,
                                hover: color,
                            },
                            width,
                            arrows: { to: { enabled: true, scaleFactor: 0.5, type: "arrow" } },
                            arrowStrikethrough: false,
                            smooth: { enabled: true, type: "continuous", roundness: 0.5 },
                            dashes: r.predicate === "contradicts" ? [6, 4] : false,
                            _originalColor: color,
                            _originalWidth: width,
                        };
                    });

                const nodesDataset = new DataSet(nodes as any[]);
                const edgesDataset = new DataSet(edges as any[]);
                nodesDatasetRef.current = nodesDataset;
                edgesDatasetRef.current = edgesDataset;

                // Store references for hover manipulation
                const allNodesObj: Record<string, any> = {};
                nodes.forEach(n => { allNodesObj[n.id] = n; });
                allNodesRef.current = allNodesObj;

                const allEdgesObj: Record<string, any> = {};
                edges.forEach(e => { allEdgesObj[e.id] = e; });
                allEdgesRef.current = allEdgesObj;

                const options: any = {
                    nodes: {
                        borderWidth: 2,
                        borderWidthSelected: 3,
                        scaling: {
                            label: {
                                enabled: true,
                                min: 10,
                                max: 18,
                                maxVisible: 24,
                                drawThreshold: 6,
                            },
                        },
                    },
                    edges: {
                        font: {
                            size: 0,
                            color: "transparent",
                        },
                        chosen: true,
                        selectionWidth: 1,
                        hoverWidth: 0.5,
                    },
                    physics: {
                        enabled: true,
                        solver: "forceAtlas2Based",
                        forceAtlas2Based: {
                            theta: 0.5,
                            gravitationalConstant: -120,
                            centralGravity: 0.008,
                            springLength: 160,
                            springConstant: 0.06,
                            damping: 0.45,
                            avoidOverlap: 0.3,
                        },
                        stabilization: {
                            enabled: true,
                            iterations: 800,
                            updateInterval: 25,
                        },
                        maxVelocity: 50,
                        minVelocity: 0.75,
                    },
                    interaction: {
                        hover: true,
                        tooltipDelay: 200,
                        hideEdgesOnDrag: false,
                        hideEdgesOnZoom: false,
                        dragNodes: true,
                        dragView: true,
                        zoomView: true,
                        multiselect: false,
                        navigationButtons: false,
                    },
                    layout: {
                        improvedLayout: true,
                        randomSeed: 42,
                    },
                };

                // Destroy existing
                if (networkRef.current) {
                    networkRef.current.destroy();
                }

                const network = new Network(containerRef.current!, { nodes: nodesDataset, edges: edgesDataset }, options);
                networkRef.current = network;

                // ── Events ──

                // Node selection
                network.on("selectNode", (params: any) => {
                    if (params.nodes.length > 0) setSelectedId(params.nodes[0]);
                });
                network.on("deselectNode", () => setSelectedId(null));

                // Zoom tracking
                network.on("zoom", () => {
                    const scale = network.getScale();
                    setZoomLevel(Math.round(scale * 100));
                });

                // Hover-to-dim
                network.on("hoverNode", (params: any) => {
                    highlightConnected(params.node);
                });
                network.on("blurNode", () => {
                    unhighlightAll();
                });

                // Drag: re-enable physics briefly with high damping
                let dragTimeout: NodeJS.Timeout;
                network.on("dragStart", () => {
                    clearTimeout(dragTimeout);
                    network.setOptions({
                        physics: {
                            enabled: true,
                            solver: "forceAtlas2Based",
                            forceAtlas2Based: {
                                gravitationalConstant: -60,
                                centralGravity: 0.005,
                                springLength: 160,
                                springConstant: 0.06,
                                damping: 0.7,
                                avoidOverlap: 0.3,
                            },
                            maxVelocity: 30,
                            minVelocity: 0.75,
                        },
                    });
                });
                network.on("dragEnd", () => {
                    clearTimeout(dragTimeout);
                    dragTimeout = setTimeout(() => {
                        network.setOptions({ physics: { enabled: false } });
                    }, 600);
                });

                // After stabilization: disable physics, fit to view
                network.on("stabilizationIterationsDone", () => {
                    setStabilizing(false);
                    network.setOptions({ physics: { enabled: false } });
                    network.fit({
                        animation: { duration: 800, easingFunction: "easeInOutQuad" as any },
                        maxZoomLevel: 1.5,
                    });
                });
            });
        });

        return () => {
            if (networkRef.current) {
                networkRef.current.destroy();
                networkRef.current = null;
            }
        };
    }, [filteredEntities, relations, loading, highlightConnected, unhighlightAll]);

    /* ── Render ─────────────────────────────────────────── */
    return (
        <div className="min-h-screen flex flex-col bg-page">
            {/* Header */}
            <header className="border-b border-edge bg-card/50 backdrop-blur-sm sticky top-0 z-20">
                <div className="max-w-full mx-auto px-6 py-3">
                    <div className="flex items-center gap-2 text-xs text-ink-muted mb-1">
                        <Link href="/" className="hover:text-sage transition-colors">Sessions</Link>
                        <span className="material-symbols-outlined text-[12px]">chevron_right</span>
                        <Link href={`/session/${sessionId}`} className="hover:text-sage transition-colors">Session</Link>
                        <span className="material-symbols-outlined text-[12px]">chevron_right</span>
                        <span className="text-ink-secondary">Knowledge Graph</span>
                    </div>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <Link href={`/session/${sessionId}`} className="text-ink-secondary hover:text-sage transition-colors">
                                <span className="material-symbols-outlined">arrow_back</span>
                            </Link>
                            <h1 className="text-lg font-display">Knowledge Graph</h1>
                            <span className="badge badge-system">{filteredEntities.length} nodes</span>
                            <span className="badge badge-system">{relations.length} edges</span>
                            {contradictions.length > 0 && (
                                <span className="badge badge-error">{contradictions.length} contradictions</span>
                            )}
                        </div>
                        <div className="flex items-center gap-2">
                            {/* Search */}
                            <div className="relative">
                                <button
                                    onClick={() => setSearchActive(a => !a)}
                                    className="btn btn-ghost text-xs px-2 py-1"
                                    title="Search entities"
                                >
                                    <span className="material-symbols-outlined text-sm">search</span>
                                </button>
                                {searchActive && (
                                    <div className="absolute right-0 top-full mt-1 w-72 z-50">
                                        <input
                                            autoFocus
                                            value={searchQuery}
                                            onChange={(e) => setSearchQuery(e.target.value)}
                                            placeholder="Search entities..."
                                            className="input h-9 w-full text-sm"
                                            onKeyDown={(e) => {
                                                if (e.key === "Escape") { setSearchActive(false); setSearchQuery(""); }
                                                if (e.key === "Enter" && searchResults.length > 0) centerOnNode(searchResults[0].id);
                                            }}
                                        />
                                        {searchResults.length > 0 && (
                                            <div className="mt-1 bg-card border border-edge rounded-lg max-h-64 overflow-y-auto" style={{ boxShadow: "var(--shadow-lg)" }}>
                                                {searchResults.map(n => {
                                                    const cfg = entityTypeConfig[n.entity_type] || defaultEntityCfg;
                                                    return (
                                                        <button
                                                            key={n.id}
                                                            onClick={() => centerOnNode(n.id)}
                                                            className="w-full text-left px-3 py-2 hover:bg-card-hover transition-colors flex items-center gap-2 text-sm"
                                                        >
                                                            <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: cfg.bg }} />
                                                            <span className="text-ink truncate">{n.name}</span>
                                                            <span className="text-xs text-ink-muted ml-auto">{n.entity_type}</span>
                                                        </button>
                                                    );
                                                })}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>

                            <button onClick={fitToView} className="btn btn-ghost text-xs gap-1" title="Fit to view">
                                <span className="material-symbols-outlined text-sm">fit_screen</span>
                            </button>
                            <button onClick={fetchGraph} className="btn btn-ghost text-xs gap-1">
                                <span className="material-symbols-outlined text-sm">refresh</span>
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Area */}
            {loading ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="flex items-center gap-3 text-ink-secondary">
                        <span className="material-symbols-outlined animate-spin">progress_activity</span>
                        Loading knowledge graph...
                    </div>
                </div>
            ) : error ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <span className="material-symbols-outlined text-4xl text-ink-muted mb-3 block">error</span>
                        <p className="text-sm text-ink-secondary">{error}</p>
                        <button onClick={fetchGraph} className="btn btn-ghost text-xs mt-4">Retry</button>
                    </div>
                </div>
            ) : entities.length === 0 ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <span className="material-symbols-outlined text-4xl text-ink-muted mb-3 block">hub</span>
                        <p className="text-sm text-ink-secondary">No knowledge graph data yet.</p>
                        <p className="text-xs text-ink-muted mt-1">Entities will appear as the research processes findings.</p>
                    </div>
                </div>
            ) : (
                <div className="flex-1 relative overflow-hidden">
                    {/* Vis-Network Container */}
                    <div ref={containerRef} className="absolute inset-0" style={{ background: THEME.pageBg }} />

                    {/* Stabilization overlay */}
                    {stabilizing && (
                        <div className="absolute inset-0 z-10 flex items-center justify-center" style={{ background: "rgba(28,26,22,0.7)" }}>
                            <div className="flex items-center gap-3 text-ink-secondary text-sm">
                                <span className="material-symbols-outlined animate-spin text-sage">progress_activity</span>
                                Computing layout...
                            </div>
                        </div>
                    )}

                    {/* Detail Panel */}
                    {selectedNode && (
                        <div className="absolute right-4 top-4 w-80 bg-card/95 backdrop-blur-xl border border-edge rounded-xl p-5 z-20 max-h-[calc(100%-6rem)] overflow-y-auto animate-fade-up-in" style={{ boxShadow: "var(--shadow-lg)" }}>
                            <div className="flex items-start justify-between mb-3">
                                <div className="flex items-center gap-2.5 min-w-0">
                                    <span
                                        className="w-3 h-3 rounded-full shrink-0"
                                        style={{ backgroundColor: (entityTypeConfig[selectedNode.entity_type] || defaultEntityCfg).bg }}
                                    />
                                    <h3 className="font-semibold text-sm truncate">{selectedNode.name}</h3>
                                </div>
                                <button onClick={() => { setSelectedId(null); if (networkRef.current) networkRef.current.unselectAll(); }} className="text-ink-muted hover:text-ink transition-colors shrink-0">
                                    <span className="material-symbols-outlined text-lg">close</span>
                                </button>
                            </div>

                            <span
                                className="badge text-xs mb-4 inline-block"
                                style={{
                                    backgroundColor: (entityTypeConfig[selectedNode.entity_type] || defaultEntityCfg).dimBg,
                                    color: (entityTypeConfig[selectedNode.entity_type] || defaultEntityCfg).bg,
                                    borderColor: (entityTypeConfig[selectedNode.entity_type] || defaultEntityCfg).bg + "40",
                                }}
                            >
                                {selectedNode.entity_type}
                            </span>

                            {/* Properties */}
                            {selectedNode.properties && Object.keys(selectedNode.properties).length > 0 && (
                                <div className="mb-4">
                                    <label className="text-[10px] text-ink-muted uppercase tracking-widest font-bold block mb-2">Properties</label>
                                    <div className="space-y-1">
                                        {Object.entries(selectedNode.properties).map(([k, v]) => (
                                            <div key={k} className="text-xs flex justify-between gap-2">
                                                <span className="text-ink-muted font-mono shrink-0">{k}</span>
                                                <span className="text-ink-secondary truncate">{String(v)}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Connections */}
                            <div>
                                <label className="text-[10px] text-ink-muted uppercase tracking-widest font-bold block mb-2">Connections ({connectedEdges.length})</label>
                                <div className="space-y-1.5 max-h-60 overflow-y-auto">
                                    {connectedEdges.map((edge, i) => {
                                        const isSubject = edge.subject_id === selectedId;
                                        const otherName = isSubject ? edge.object_name : edge.subject_name;
                                        const color = predicateColors[edge.predicate] || defaultEdgeColor;
                                        return (
                                            <button
                                                key={i}
                                                onClick={() => {
                                                    const otherId = isSubject ? edge.object_id : edge.subject_id;
                                                    centerOnNode(otherId);
                                                }}
                                                className="flex items-center gap-2 text-xs w-full text-left hover:bg-card-hover p-1.5 rounded-md transition-colors"
                                            >
                                                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: color }} />
                                                <span className="text-ink-muted font-mono shrink-0">{edge.predicate.replace(/_/g, " ")}</span>
                                                <span className="text-ink-secondary truncate">{otherName || "Unknown"}</span>
                                                <span className="material-symbols-outlined text-ink-muted text-[12px] ml-auto shrink-0">arrow_forward</span>
                                            </button>
                                        );
                                    })}
                                    {connectedEdges.length === 0 && (
                                        <p className="text-xs text-ink-muted">No connections</p>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Bottom Controls Bar */}
                    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-card/90 backdrop-blur-xl border border-edge rounded-xl px-5 py-2.5 z-20 flex items-center gap-4 max-w-[92vw] overflow-x-auto" style={{ boxShadow: "var(--shadow-lg)" }}>
                        {/* Legend */}
                        <div className="flex items-center gap-3 text-[11px] shrink-0 border-r border-edge pr-4">
                            {Object.entries(entityTypeConfig).slice(0, 6).map(([key, cfg]) => (
                                <span key={key} className="flex items-center gap-1.5 text-ink-secondary">
                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: cfg.bg }} />
                                    <span>{cfg.label}</span>
                                </span>
                            ))}
                        </div>

                        {/* Node Type Filter */}
                        <div className="flex items-center gap-1.5 shrink-0 border-r border-edge pr-4">
                            <button
                                onClick={() => setNodeTypeFilter("all")}
                                className={`chip text-[11px] ${nodeTypeFilter === "all" ? "chip-active" : ""}`}
                            >
                                All
                            </button>
                            {entityTypes.map(t => (
                                <button
                                    key={t}
                                    onClick={() => setNodeTypeFilter(prev => prev === t ? "all" : t)}
                                    className={`chip text-[11px] ${nodeTypeFilter === t ? "chip-active" : ""}`}
                                >
                                    <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: (entityTypeConfig[t] || defaultEntityCfg).bg }} />
                                    {t.toLowerCase()}
                                </button>
                            ))}
                        </div>

                        {/* Confidence Slider */}
                        <div className="flex items-center gap-2 text-[11px] text-ink-secondary shrink-0">
                            <span>Confidence</span>
                            <input
                                type="range"
                                min="0"
                                max="100"
                                step="5"
                                value={confidenceFilter}
                                onChange={(e) => setConfidenceFilter(parseInt(e.target.value))}
                                className="w-16 accent-sage"
                            />
                            <span className="font-mono w-8">{confidenceFilter}%</span>
                        </div>

                        {/* Zoom Level Indicator */}
                        <div className="text-[10px] text-ink-secondary font-mono shrink-0 border-l border-edge pl-4">
                            {zoomLevel}%
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
