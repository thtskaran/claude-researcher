"use client";

import { useState, useEffect, useRef, useMemo } from "react";
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
const entityTypeConfig: Record<string, { color: string; icon: string; label: string }> = {
    CLAIM: { color: "#f59e0b", icon: "\u{1F4A1}", label: "Claim" },
    CONCEPT: { color: "#6366f1", icon: "\u{1F3F7}\uFE0F", label: "Concept" },
    SOURCE: { color: "#6b7280", icon: "\u{1F4C4}", label: "Source" },
    EVIDENCE: { color: "#10b981", icon: "\u{1F52C}", label: "Evidence" },
    TECHNOLOGY: { color: "#0ea5e9", icon: "\u2699\uFE0F", label: "Technology" },
    METHOD: { color: "#8b5cf6", icon: "\u{1F527}", label: "Method" },
    METRIC: { color: "#f43f5e", icon: "\u{1F4CA}", label: "Metric" },
    PERSON: { color: "#ec4899", icon: "\u{1F464}", label: "Person" },
    ORGANIZATION: { color: "#14b8a6", icon: "\u{1F3E2}", label: "Organization" },
    QUOTE: { color: "#f97316", icon: "\u{1F4AC}", label: "Quote" },
};
const defaultEntityCfg = { color: "#9ca3af", icon: "\u25CF", label: "Other" };

const predicateColors: Record<string, string> = {
    supports: "#10b981",
    contradicts: "#f43f5e",
    related: "#0ea5e9",
    related_to: "#0ea5e9",
    is_a: "#6366f1",
    part_of: "#8b5cf6",
    causes: "#f59e0b",
    cites: "#6b7280",
    implements: "#a78bfa",
    outperforms: "#10b981",
    similar_to: "#0ea5e9",
    alternative_to: "#ec4899",
    authored_by: "#6b7280",
    mentioned_in: "#6b7280",
};
const defaultEdgeColor = "#484f58";

/* ── Theme-aware colors for vis-network ───────────────── */
function getGraphTheme() {
    const isDark = typeof document !== "undefined" && document.documentElement.classList.contains("dark");
    return {
        fontColor: isDark ? "#f0ece4" : "#3a2c22",
        fontBg: isDark ? "rgba(28, 26, 22, 0.85)" : "rgba(247, 244, 234, 0.85)",
        edgeFontColor: isDark ? "#b4aa9e" : "#766252",
    };
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

    const containerRef = useRef<HTMLDivElement>(null);
    const networkRef = useRef<any>(null);

    /* ── Fetch data ─────────────────────────────────────── */
    const fetchGraph = async () => {
        setLoading(true);
        setError("");
        try {
            const res = await fetch(`/api/sessions/${sessionId}/knowledge/graph?limit=500`);
            if (!res.ok) throw new Error("Failed to load knowledge graph");
            const data = await res.json();
            const ents: KGEntity[] = data.entities || [];
            const rels: KGRelation[] = data.relations || [];
            setEntities(ents);
            setRelations(rels);
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

    const centerOnNode = (nodeId: string) => {
        if (networkRef.current) {
            networkRef.current.selectNodes([nodeId], true);
            networkRef.current.moveTo({
                position: { x: 0, y: 0 },
                scale: 1.5,
                animation: { duration: 500, easingFunction: "easeInOutQuad" as any }
            });
            setSelectedId(nodeId);
            setSearchActive(false);
            setSearchQuery("");
        }
    };

    const fitToView = () => {
        if (networkRef.current) {
            networkRef.current.fit({ animation: { duration: 500, easingFunction: "easeInOutQuad" as any } });
        }
    };

    /* ── Initialize vis-network ─────────────────────────── */
    useEffect(() => {
        if (!containerRef.current || loading || entities.length === 0) return;

        // Dynamically import vis-network (client-side only)
        import("vis-network").then(({ Network }) => {
            import("vis-data").then(({ DataSet }) => {
                if (!containerRef.current) return;

                const theme = getGraphTheme();

                // Create filtered node IDs set
                const filteredIds = new Set(filteredEntities.map(e => e.id));

                // Prepare nodes
                const nodes = filteredEntities.map(e => {
                    const cfg = entityTypeConfig[e.entity_type] || defaultEntityCfg;
                    const edgeCount = relations.filter(r =>
                        r.subject_id === e.id || r.object_id === e.id
                    ).length;
                    const size = Math.max(20, Math.min(50, 20 + edgeCount * 3));

                    return {
                        id: e.id,
                        label: e.name.length > 25 ? e.name.slice(0, 22) + "..." : e.name,
                        title: `<b>${e.name}</b><br/>Type: ${e.entity_type}<br/>Connections: ${edgeCount}`,
                        color: {
                            background: cfg.color,
                            border: cfg.color,
                            highlight: { background: cfg.color, border: "#ffffff" }
                        },
                        shape: e.entity_type === "CLAIM" ? "diamond" : "dot",
                        size: size,
                        font: {
                            color: theme.fontColor,
                            size: 14,
                            face: "Figtree, system-ui, sans-serif",
                            background: theme.fontBg,
                            strokeWidth: 0
                        }
                    };
                });

                // Prepare edges
                const edges = relations
                    .filter(r => filteredIds.has(r.subject_id) && filteredIds.has(r.object_id))
                    .map(r => {
                        const color = predicateColors[r.predicate] || defaultEdgeColor;
                        return {
                            id: r.id,
                            from: r.subject_id,
                            to: r.object_id,
                            label: "",  // Hide edge labels for cleaner look
                            title: r.predicate.replace(/_/g, " "),
                            color: {
                                color: color + "80",  // 50% opacity
                                highlight: color,
                                hover: color
                            },
                            width: 1.5 + (r.confidence * 1.5),
                            arrows: {
                                to: {
                                    enabled: true,
                                    scaleFactor: 0.8,
                                    type: "arrow"
                                }
                            },
                            arrowStrikethrough: false,
                            smooth: { enabled: true, type: "curvedCW", roundness: 0.2 },
                            dashes: r.predicate === "contradicts" ? [5, 5] : false
                        };
                    });

                const data = {
                    nodes: new DataSet(nodes),
                    edges: new DataSet(edges)
                };

                const options: any = {
                    nodes: {
                        borderWidth: 2,
                        borderWidthSelected: 4
                    },
                    edges: {
                        font: {
                            size: 11,
                            color: theme.edgeFontColor,
                            strokeWidth: 0,
                            align: "middle"
                        },
                        chosen: true,
                        hoverWidth: 0.5
                    },
                    physics: {
                        enabled: true,
                        barnesHut: {
                            gravitationalConstant: -80000,
                            centralGravity: 0.3,
                            springLength: 250,
                            springConstant: 0.001,
                            damping: 0.09,
                            avoidOverlap: 0.8
                        },
                        stabilization: {
                            enabled: true,
                            iterations: 400,
                            updateInterval: 25
                        }
                    },
                    interaction: {
                        hover: true,
                        tooltipDelay: 100,
                        hideEdgesOnDrag: false,
                        hideEdgesOnZoom: false,
                        dragNodes: true,
                        dragView: true,
                        zoomView: true
                    },
                    layout: {
                        improvedLayout: true,
                        randomSeed: 42
                    }
                };

                // Destroy existing network
                if (networkRef.current) {
                    networkRef.current.destroy();
                }

                // Create new network
                const network = new Network(containerRef.current, data, options);
                networkRef.current = network;

                // Event listeners
                network.on("selectNode", (params: any) => {
                    if (params.nodes.length > 0) {
                        setSelectedId(params.nodes[0]);
                    }
                });

                network.on("deselectNode", () => {
                    setSelectedId(null);
                });

                network.on("zoom", () => {
                    if (networkRef.current) {
                        const scale = networkRef.current.getScale();
                        setZoomLevel(Math.round(scale * 100));
                    }
                });

                // Drag events for elastic physics
                let dragTimeout: NodeJS.Timeout;
                network.on("dragStart", () => {
                    network.setOptions({
                        physics: {
                            enabled: true,
                            barnesHut: {
                                gravitationalConstant: -8000,
                                centralGravity: 0.05,
                                springLength: 150,
                                springConstant: 0.01,
                                damping: 0.4,
                                avoidOverlap: 0.3
                            },
                            maxVelocity: 50,
                            minVelocity: 0.1,
                            solver: 'barnesHut',
                            timestep: 0.35
                        }
                    });
                });

                network.on("dragEnd", () => {
                    clearTimeout(dragTimeout);
                    dragTimeout = setTimeout(() => {
                        network.setOptions({ physics: { enabled: false } });
                    }, 800);
                });

                // Disable physics after stabilization
                network.on("stabilizationIterationsDone", () => {
                    network.fit({ animation: { duration: 500, easingFunction: "easeInOutQuad" as any } });
                    network.setOptions({ physics: { enabled: false } });
                });
            });
        });

        return () => {
            if (networkRef.current) {
                networkRef.current.destroy();
                networkRef.current = null;
            }
        };
    }, [filteredEntities, relations, loading]);

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
                                <span className="badge badge-error">{contradictions.length} ⚡</span>
                            )}
                        </div>
                        <div className="flex items-center gap-2">
                            {/* Search */}
                            <div className="relative">
                                <div className="flex items-center">
                                    <button
                                        onClick={() => setSearchActive(a => !a)}
                                        className="btn btn-ghost text-xs px-2 py-1"
                                        title="Search entities"
                                    >
                                        <span className="material-symbols-outlined text-sm">search</span>
                                    </button>
                                </div>
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
                                                            <span style={{ color: cfg.color }}>{cfg.icon}</span>
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
                    <div ref={containerRef} className="absolute inset-0 bg-page" />

                    {/* Detail Panel */}
                    {selectedNode && (
                        <div className="absolute right-4 top-4 w-80 bg-card/95 backdrop-blur-xl border border-edge rounded-xl p-5 z-20 max-h-[calc(100%-6rem)] overflow-y-auto" style={{ boxShadow: "var(--shadow-lg)" }}>
                            <div className="flex items-start justify-between mb-3">
                                <div className="flex items-center gap-2 min-w-0">
                                    <span className="text-xl shrink-0">{(entityTypeConfig[selectedNode.entity_type] || defaultEntityCfg).icon}</span>
                                    <h3 className="font-semibold text-sm truncate">{selectedNode.name}</h3>
                                </div>
                                <button onClick={() => setSelectedId(null)} className="text-ink-muted hover:text-ink transition-colors shrink-0">
                                    <span className="material-symbols-outlined text-lg">close</span>
                                </button>
                            </div>

                            <span
                                className="badge text-xs mb-4 inline-block"
                                style={{
                                    backgroundColor: `${(entityTypeConfig[selectedNode.entity_type] || defaultEntityCfg).color}20`,
                                    color: (entityTypeConfig[selectedNode.entity_type] || defaultEntityCfg).color,
                                    borderColor: `${(entityTypeConfig[selectedNode.entity_type] || defaultEntityCfg).color}40`,
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
                                <span key={key} className="flex items-center gap-1 text-ink-secondary">
                                    <span className="text-xs">{cfg.icon}</span>
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
                                    {(entityTypeConfig[t] || defaultEntityCfg).icon} {t.toLowerCase()}
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
