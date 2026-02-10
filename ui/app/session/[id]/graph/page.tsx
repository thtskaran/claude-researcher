"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

/* ── types ──────────────────────────────────────────────── */
interface KGEntity {
    id: string;
    name: string;
    entity_type: string;
    properties: Record<string, unknown>;
    created_at: string;
    // Computed on the frontend for layout
    x?: number;
    y?: number;
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

/* ── style maps ────────────────────────────────────────── */
const entityTypeConfig: Record<string, { color: string; icon: string }> = {
    CLAIM: { color: "#2b7cee", icon: "lightbulb" },
    CONCEPT: { color: "#10b981", icon: "category" },
    SOURCE: { color: "#d29922", icon: "article" },
    EVIDENCE: { color: "#a78bfa", icon: "labs" },
    TECHNOLOGY: { color: "#06b6d4", icon: "memory" },
    METHOD: { color: "#f97316", icon: "build" },
    METRIC: { color: "#ec4899", icon: "monitoring" },
    PERSON: { color: "#8b5cf6", icon: "person" },
    ORGANIZATION: { color: "#14b8a6", icon: "corporate_fare" },
    QUOTE: { color: "#f59e0b", icon: "format_quote" },
};
const defaultEntityCfg = { color: "#6b7280", icon: "circle" };

const predicateConfig: Record<string, { color: string; label: string }> = {
    supports: { color: "#3fb950", label: "Supports" },
    contradicts: { color: "#f85149", label: "Contradicts" },
    related: { color: "#58a6ff", label: "Related" },
    is_a: { color: "#58a6ff", label: "Is A" },
    part_of: { color: "#58a6ff", label: "Part Of" },
    causes: { color: "#d29922", label: "Causes" },
    cites: { color: "#d29922", label: "Cites" },
    implements: { color: "#a78bfa", label: "Implements" },
    outperforms: { color: "#3fb950", label: "Outperforms" },
    similar_to: { color: "#58a6ff", label: "Similar To" },
    alternative_to: { color: "#ec4899", label: "Alternative" },
    authored_by: { color: "#6b7280", label: "Authored By" },
    mentioned_in: { color: "#6b7280", label: "Mentioned In" },
};
const defaultEdgeCfg = { color: "#484f58", label: "Link" };

/* ── force-directed positions ───────────────────────────── */
function assignPositions(entities: KGEntity[]): KGEntity[] {
    if (entities.length === 0) return [];
    const n = entities.length;
    // Place nodes in expanding spiral to avoid overlap
    return entities.map((e, i) => {
        const angle = i * 2.39996; // golden angle in radians
        const radius = 15 + Math.sqrt(i / n) * 32;
        return {
            ...e,
            x: Math.max(5, Math.min(95, 50 + radius * Math.cos(angle))),
            y: Math.max(5, Math.min(95, 50 + radius * Math.sin(angle))),
        };
    });
}

export default function KnowledgeGraphPage() {
    const params = useParams();
    const sessionId = params.id as string;
    const [entities, setEntities] = useState<KGEntity[]>([]);
    const [relations, setRelations] = useState<KGRelation[]>([]);
    const [contradictions, setContradictions] = useState<KGContradiction[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [selectedNode, setSelectedNode] = useState<string | null>(null);
    const [confidenceFilter, setConfidenceFilter] = useState(0);
    const [nodeTypeFilter, setNodeTypeFilter] = useState<string>("all");

    /* ── fetch graph data ─────────────────────────────── */
    const fetchGraph = useCallback(async () => {
        setLoading(true);
        setError("");
        try {
            const res = await fetch(
                `/api/sessions/${sessionId}/knowledge/graph?limit=500`
            );
            if (!res.ok) throw new Error("Failed to load knowledge graph");
            const data = await res.json();
            const positioned = assignPositions(data.entities || []);
            setEntities(positioned);
            setRelations(data.relations || []);
            setContradictions(data.contradictions || []);
        } catch {
            setError("Unable to load knowledge graph data");
        } finally {
            setLoading(false);
        }
    }, [sessionId]);

    useEffect(() => { fetchGraph(); }, [fetchGraph]);

    /* ── filtering ─────────────────────────────────────── */
    const filteredNodes = useMemo(() => {
        return entities.filter((n) => {
            if (confidenceFilter > 0) {
                const conf = (n.properties as Record<string, number>)?.confidence ?? 1;
                if (conf < confidenceFilter / 100) return false;
            }
            if (nodeTypeFilter !== "all" && n.entity_type !== nodeTypeFilter) return false;
            return true;
        });
    }, [entities, confidenceFilter, nodeTypeFilter]);

    const filteredNodeIds = new Set(filteredNodes.map((n) => n.id));
    const filteredEdges = relations.filter(
        (e) => filteredNodeIds.has(e.subject_id) && filteredNodeIds.has(e.object_id)
    );

    const selected = entities.find((n) => n.id === selectedNode);
    const connectedEdges = selected
        ? relations.filter((e) => e.subject_id === selected.id || e.object_id === selected.id)
        : [];

    const entityTypes = useMemo(() => {
        const types = new Set(entities.map((e) => e.entity_type));
        return Array.from(types).sort();
    }, [entities]);

    /* ── render ─────────────────────────────────────────── */
    return (
        <div className="min-h-screen flex flex-col">
            {/* Header */}
            <header className="border-b border-dark-border bg-dark-surface/50 backdrop-blur-sm sticky top-0 z-20">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center gap-2 text-xs text-gray-500 mb-2">
                        <Link href="/" className="hover:text-primary transition-colors">Sessions</Link>
                        <span className="material-symbols-outlined text-[12px]">chevron_right</span>
                        <Link href={`/session/${sessionId}`} className="hover:text-primary transition-colors">Session</Link>
                        <span className="material-symbols-outlined text-[12px]">chevron_right</span>
                        <span className="text-gray-300">Knowledge Graph</span>
                    </div>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <Link href={`/session/${sessionId}`} className="text-gray-400 hover:text-primary transition-colors">
                                <span className="material-symbols-outlined">arrow_back</span>
                            </Link>
                            <h1 className="text-xl font-bold">Knowledge Graph</h1>
                            <span className="badge badge-system">{filteredNodes.length} nodes</span>
                            <span className="badge badge-system">{filteredEdges.length} edges</span>
                            {contradictions.length > 0 && (
                                <span className="badge badge-error">{contradictions.length} contradictions</span>
                            )}
                        </div>
                        <button onClick={fetchGraph} className="btn btn-ghost text-xs gap-1">
                            <span className="material-symbols-outlined text-sm">refresh</span>
                            Refresh
                        </button>
                    </div>
                </div>
            </header>

            {/* Loading / Error */}
            {loading ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="flex items-center gap-3 text-gray-400">
                        <span className="material-symbols-outlined animate-spin">progress_activity</span>
                        Loading knowledge graph...
                    </div>
                </div>
            ) : error ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <span className="material-symbols-outlined text-4xl text-gray-600 mb-3 block">error</span>
                        <p className="text-sm text-gray-400">{error}</p>
                        <button onClick={fetchGraph} className="btn btn-ghost text-xs mt-4">Retry</button>
                    </div>
                </div>
            ) : entities.length === 0 ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <span className="material-symbols-outlined text-4xl text-gray-600 mb-3 block">hub</span>
                        <p className="text-sm text-gray-400">No knowledge graph data yet.</p>
                        <p className="text-xs text-gray-600 mt-1">Entities will appear as the research processes findings.</p>
                    </div>
                </div>
            ) : (
                // Graph Area
                <div className="flex-1 flex relative overflow-hidden bg-grid-pattern">
                    {/* SVG edges */}
                    <div className="flex-1 relative" style={{ minHeight: "600px" }}>
                        <svg className="absolute inset-0 w-full h-full pointer-events-none">
                            {filteredEdges.map((edge, i) => {
                                const fromNode = entities.find((n) => n.id === edge.subject_id);
                                const toNode = entities.find((n) => n.id === edge.object_id);
                                if (!fromNode?.x || !toNode?.x) return null;
                                const cfg = predicateConfig[edge.predicate] || defaultEdgeCfg;
                                return (
                                    <line
                                        key={i}
                                        x1={`${fromNode.x}%`}
                                        y1={`${fromNode.y}%`}
                                        x2={`${toNode.x}%`}
                                        y2={`${toNode.y}%`}
                                        stroke={cfg.color}
                                        strokeWidth="1.5"
                                        opacity="0.4"
                                        strokeDasharray={edge.predicate === "contradicts" ? "6,4" : undefined}
                                    />
                                );
                            })}
                        </svg>

                        {/* Nodes */}
                        {filteredNodes.map((node) => {
                            const cfg = entityTypeConfig[node.entity_type] || defaultEntityCfg;
                            const isSelected = selectedNode === node.id;
                            return (
                                <button
                                    key={node.id}
                                    onClick={() => setSelectedNode(isSelected ? null : node.id)}
                                    className={`absolute flex flex-col items-center gap-1.5 group cursor-pointer z-10 transition-transform hover:scale-110 ${isSelected ? "scale-110" : ""}`}
                                    style={{
                                        left: `${node.x}%`,
                                        top: `${node.y}%`,
                                        transform: `translate(-50%, -50%) ${isSelected ? "scale(1.1)" : ""}`,
                                    }}
                                >
                                    <div
                                        className={`w-10 h-10 rounded-full border-2 flex items-center justify-center transition-all ${isSelected ? "ring-2 ring-white/30" : ""}`}
                                        style={{ backgroundColor: cfg.color, borderColor: `${cfg.color}66` }}
                                    >
                                        <span className="material-symbols-outlined text-white text-base">{cfg.icon}</span>
                                    </div>
                                    <span className="font-mono text-[10px] text-gray-300 bg-black/60 px-2 py-0.5 rounded backdrop-blur-sm whitespace-nowrap max-w-24 truncate">
                                        {node.name}
                                    </span>
                                </button>
                            );
                        })}
                    </div>

                    {/* Detail Panel */}
                    {selected && (
                        <div className="absolute right-4 top-4 w-80 glass-panel rounded-xl p-6 z-20 max-h-[calc(100%-2rem)] overflow-y-auto">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="font-semibold text-sm">{selected.name}</h3>
                                <button onClick={() => setSelectedNode(null)} className="text-gray-500 hover:text-white transition-colors">
                                    <span className="material-symbols-outlined text-lg">close</span>
                                </button>
                            </div>
                            <div className="space-y-4">
                                <div className="flex items-center gap-2">
                                    <span
                                        className="badge"
                                        style={{
                                            backgroundColor: `${(entityTypeConfig[selected.entity_type] || defaultEntityCfg).color}20`,
                                            color: (entityTypeConfig[selected.entity_type] || defaultEntityCfg).color,
                                            borderColor: `${(entityTypeConfig[selected.entity_type] || defaultEntityCfg).color}40`,
                                        }}
                                    >
                                        {selected.entity_type}
                                    </span>
                                </div>

                                {/* Properties */}
                                {selected.properties && Object.keys(selected.properties).length > 0 && (
                                    <div>
                                        <label className="text-xs text-gray-500 uppercase tracking-wider">Properties</label>
                                        <div className="mt-1 space-y-1">
                                            {Object.entries(selected.properties).map(([k, v]) => (
                                                <div key={k} className="text-xs flex justify-between">
                                                    <span className="text-gray-500 font-mono">{k}</span>
                                                    <span className="text-gray-300 truncate ml-2">{String(v)}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Connections */}
                                <div>
                                    <label className="text-xs text-gray-500 uppercase tracking-wider">Connections ({connectedEdges.length})</label>
                                    <div className="mt-2 space-y-2 max-h-48 overflow-y-auto">
                                        {connectedEdges.map((edge, i) => {
                                            const isSubject = edge.subject_id === selected.id;
                                            const otherName = isSubject ? edge.object_name : edge.subject_name;
                                            const eCfg = predicateConfig[edge.predicate] || defaultEdgeCfg;
                                            return (
                                                <div key={i} className="flex items-center gap-2 text-xs">
                                                    <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: eCfg.color }} />
                                                    <span className="text-gray-400">{eCfg.label}</span>
                                                    <span className="text-gray-300 truncate">{otherName || "Unknown"}</span>
                                                </div>
                                            );
                                        })}
                                        {connectedEdges.length === 0 && (
                                            <p className="text-xs text-gray-600">No connections</p>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Bottom Controls Bar */}
                    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 glass-panel rounded-xl px-6 py-3 z-20 flex items-center gap-6 max-w-[90vw] overflow-x-auto">
                        {/* Legend */}
                        <div className="flex items-center gap-4 text-xs shrink-0">
                            {Object.entries(predicateConfig).slice(0, 4).map(([key, cfg]) => (
                                <span key={key} className="flex items-center gap-1.5 text-gray-400">
                                    <span className="w-3 h-0.5 rounded" style={{ backgroundColor: cfg.color, display: "inline-block" }} />
                                    {cfg.label}
                                </span>
                            ))}
                        </div>

                        <div className="w-px h-6 bg-dark-border shrink-0" />

                        {/* Node Type Filter */}
                        <div className="flex items-center gap-2 shrink-0">
                            <button
                                onClick={() => setNodeTypeFilter("all")}
                                className={`chip ${nodeTypeFilter === "all" ? "chip-active" : ""}`}
                            >
                                All
                            </button>
                            {entityTypes.map((t) => (
                                <button
                                    key={t}
                                    onClick={() => setNodeTypeFilter(t)}
                                    className={`chip ${nodeTypeFilter === t ? "chip-active" : ""}`}
                                >
                                    {t.toLowerCase()}
                                </button>
                            ))}
                        </div>

                        <div className="w-px h-6 bg-dark-border shrink-0" />

                        {/* Confidence Slider */}
                        <div className="flex items-center gap-2 text-xs text-gray-400 shrink-0">
                            <span>Min</span>
                            <input
                                type="range"
                                min="0"
                                max="100"
                                step="5"
                                value={confidenceFilter}
                                onChange={(e) => setConfidenceFilter(parseInt(e.target.value))}
                                className="w-16 accent-primary"
                            />
                            <span className="font-mono w-8">{confidenceFilter}%</span>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
