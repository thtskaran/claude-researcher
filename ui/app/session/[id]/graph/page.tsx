"use client";

import { useState, useMemo } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

// Mock data for the knowledge graph (will be wired to API later)
const mockNodes = [
    { id: "n1", label: "Solar Energy Growth", type: "claim", confidence: 0.92, x: 50, y: 50 },
    { id: "n2", label: "Government Subsidies", type: "concept", confidence: 0.85, x: 25, y: 35 },
    { id: "n3", label: "Panel Cost Reduction", type: "claim", confidence: 0.88, x: 70, y: 30 },
    { id: "n4", label: "Rural Electrification", type: "concept", confidence: 0.78, x: 30, y: 70 },
    { id: "n5", label: "Carbon Emissions Impact", type: "claim", confidence: 0.95, x: 75, y: 65 },
    { id: "n6", label: "IEA Report 2024", type: "source", confidence: 0.97, x: 15, y: 55 },
    { id: "n7", label: "Market Competition", type: "concept", confidence: 0.72, x: 60, y: 80 },
    { id: "n8", label: "Storage Technology", type: "claim", confidence: 0.65, x: 85, y: 45 },
    { id: "n9", label: "Grid Integration", type: "concept", confidence: 0.80, x: 40, y: 25 },
    { id: "n10", label: "Investment Trends", type: "claim", confidence: 0.91, x: 55, y: 60 },
];

const mockEdges = [
    { from: "n1", to: "n2", type: "supports" },
    { from: "n1", to: "n3", type: "related" },
    { from: "n2", to: "n4", type: "supports" },
    { from: "n3", to: "n8", type: "related" },
    { from: "n5", to: "n1", type: "supports" },
    { from: "n6", to: "n1", type: "supports" },
    { from: "n6", to: "n5", type: "supports" },
    { from: "n7", to: "n3", type: "contradicts" },
    { from: "n9", to: "n8", type: "related" },
    { from: "n10", to: "n2", type: "supports" },
    { from: "n10", to: "n7", type: "related" },
];

const nodeTypeConfig = {
    claim: { color: "#2b7cee", icon: "lightbulb" },
    concept: { color: "#10b981", icon: "category" },
    source: { color: "#d29922", icon: "article" },
};

const edgeTypeConfig = {
    supports: { color: "#3fb950", label: "Supports" },
    contradicts: { color: "#f85149", label: "Contradicts" },
    related: { color: "#58a6ff", label: "Related" },
};

export default function KnowledgeGraphPage() {
    const params = useParams();
    const sessionId = params.id as string;
    const [selectedNode, setSelectedNode] = useState<string | null>(null);
    const [confidenceFilter, setConfidenceFilter] = useState(0);
    const [nodeTypeFilter, setNodeTypeFilter] = useState<string>("all");

    const filteredNodes = useMemo(() => {
        return mockNodes.filter((n) => {
            if (n.confidence < confidenceFilter / 100) return false;
            if (nodeTypeFilter !== "all" && n.type !== nodeTypeFilter) return false;
            return true;
        });
    }, [confidenceFilter, nodeTypeFilter]);

    const filteredNodeIds = new Set(filteredNodes.map((n) => n.id));
    const filteredEdges = mockEdges.filter((e) => filteredNodeIds.has(e.from) && filteredNodeIds.has(e.to));

    const selected = mockNodes.find((n) => n.id === selectedNode);
    const connectedEdges = selected
        ? mockEdges.filter((e) => e.from === selected.id || e.to === selected.id)
        : [];

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
                        </div>
                    </div>
                </div>
            </header>

            {/* Graph Area */}
            <div className="flex-1 flex relative overflow-hidden bg-grid-pattern">
                {/* SVG Graph Canvas */}
                <div className="flex-1 relative" style={{ minHeight: "600px" }}>
                    <svg className="absolute inset-0 w-full h-full pointer-events-none">
                        {filteredEdges.map((edge, i) => {
                            const fromNode = mockNodes.find((n) => n.id === edge.from);
                            const toNode = mockNodes.find((n) => n.id === edge.to);
                            if (!fromNode || !toNode) return null;
                            const cfg = edgeTypeConfig[edge.type as keyof typeof edgeTypeConfig];
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
                                    strokeDasharray={edge.type === "contradicts" ? "6,4" : undefined}
                                />
                            );
                        })}
                    </svg>

                    {/* Nodes */}
                    {filteredNodes.map((node) => {
                        const cfg = nodeTypeConfig[node.type as keyof typeof nodeTypeConfig];
                        const isSelected = selectedNode === node.id;
                        const pct = Math.round(node.confidence * 100);
                        const glow = pct >= 80 ? "shadow-[0_0_16px_rgba(43,124,238,0.4)]" : pct >= 50 ? "shadow-[0_0_12px_rgba(43,124,238,0.2)]" : "";

                        return (
                            <button
                                key={node.id}
                                onClick={() => setSelectedNode(isSelected ? null : node.id)}
                                className={`absolute flex flex-col items-center gap-1.5 group cursor-pointer z-10 transition-transform hover:scale-110 ${isSelected ? "scale-110" : ""
                                    }`}
                                style={{
                                    left: `${node.x}%`,
                                    top: `${node.y}%`,
                                    transform: `translate(-50%, -50%) ${isSelected ? "scale(1.1)" : ""}`,
                                }}
                            >
                                <div
                                    className={`w-10 h-10 rounded-full border-2 flex items-center justify-center transition-all ${glow} ${isSelected ? "ring-2 ring-white/30" : ""
                                        }`}
                                    style={{
                                        backgroundColor: cfg.color,
                                        borderColor: `${cfg.color}66`,
                                    }}
                                >
                                    <span className="material-symbols-outlined text-white text-base">{cfg.icon}</span>
                                </div>
                                <span className="font-mono text-[10px] text-gray-300 bg-black/60 px-2 py-0.5 rounded backdrop-blur-sm whitespace-nowrap max-w-24 truncate">
                                    {node.label}
                                </span>
                            </button>
                        );
                    })}
                </div>

                {/* Detail Panel (floating) */}
                {selected && (
                    <div className="absolute right-4 top-4 w-80 glass-panel rounded-xl p-6 z-20 max-h-[calc(100%-2rem)] overflow-y-auto">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="font-semibold text-sm">{selected.label}</h3>
                            <button
                                onClick={() => setSelectedNode(null)}
                                className="text-gray-500 hover:text-white transition-colors"
                            >
                                <span className="material-symbols-outlined text-lg">close</span>
                            </button>
                        </div>

                        <div className="space-y-4">
                            <div className="flex items-center gap-2">
                                <span className="badge" style={{ backgroundColor: `${nodeTypeConfig[selected.type as keyof typeof nodeTypeConfig].color}20`, color: nodeTypeConfig[selected.type as keyof typeof nodeTypeConfig].color, borderColor: `${nodeTypeConfig[selected.type as keyof typeof nodeTypeConfig].color}40` }}>
                                    {selected.type}
                                </span>
                                <span className="text-xs font-mono text-gray-400">
                                    {Math.round(selected.confidence * 100)}% confidence
                                </span>
                            </div>

                            {/* Confidence Bar */}
                            <div>
                                <label className="text-xs text-gray-500 uppercase tracking-wider">Confidence</label>
                                <div className="mt-1 h-2 bg-dark-border rounded-full overflow-hidden">
                                    <div
                                        className="h-full rounded-full transition-all"
                                        style={{
                                            width: `${Math.round(selected.confidence * 100)}%`,
                                            backgroundColor: nodeTypeConfig[selected.type as keyof typeof nodeTypeConfig].color,
                                        }}
                                    />
                                </div>
                            </div>

                            {/* Connected Nodes */}
                            <div>
                                <label className="text-xs text-gray-500 uppercase tracking-wider">Connections</label>
                                <div className="mt-2 space-y-2">
                                    {connectedEdges.map((edge, i) => {
                                        const otherId = edge.from === selected.id ? edge.to : edge.from;
                                        const other = mockNodes.find((n) => n.id === otherId);
                                        const eCfg = edgeTypeConfig[edge.type as keyof typeof edgeTypeConfig];
                                        return (
                                            <div key={i} className="flex items-center gap-2 text-xs">
                                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: eCfg.color }} />
                                                <span className="text-gray-400">{eCfg.label}</span>
                                                <span className="text-gray-300">{other?.label}</span>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Bottom Controls Bar */}
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 glass-panel rounded-xl px-6 py-3 z-20 flex items-center gap-6">
                    {/* Legend */}
                    <div className="flex items-center gap-4 text-xs">
                        {Object.entries(edgeTypeConfig).map(([key, cfg]) => (
                            <span key={key} className="flex items-center gap-1.5 text-gray-400">
                                <span className="w-3 h-0.5 rounded" style={{ backgroundColor: cfg.color, display: "inline-block" }} />
                                {cfg.label}
                            </span>
                        ))}
                    </div>

                    <div className="w-px h-6 bg-dark-border" />

                    {/* Node Type Filter */}
                    <div className="flex items-center gap-2">
                        {["all", "claim", "concept", "source"].map((t) => (
                            <button
                                key={t}
                                onClick={() => setNodeTypeFilter(t)}
                                className={`chip ${nodeTypeFilter === t ? "chip-active" : ""}`}
                            >
                                {t === "all" ? "All" : t}
                            </button>
                        ))}
                    </div>

                    <div className="w-px h-6 bg-dark-border" />

                    {/* Confidence Slider */}
                    <div className="flex items-center gap-2 text-xs text-gray-400">
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
        </div>
    );
}
