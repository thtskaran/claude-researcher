"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

/* ── types ─────────────────────────────────────────────── */
interface VerificationResult {
    id: number;
    finding_id: number;
    original_confidence: number | null;
    verified_confidence: number | null;
    verification_status: string | null;
    verification_method: string | null;
    consistency_score: number;
    kg_support_score: number;
    kg_entity_matches: number;
    kg_supporting_relations: number;
    critic_iterations: number;
    corrections_made: unknown | null;
    external_verification_used: boolean;
    contradictions: unknown | null;
    verification_time_ms: number | null;
    created_at: string;
    error: string | null;
    finding_content: string | null;
    finding_type: string | null;
    source_url: string | null;
    current_confidence: number | null;
}

interface VerificationStats {
    total: number;
    verified: number;
    flagged: number;
    rejected: number;
    pending: number;
    avg_confidence: number | null;
    avg_time_ms: number | null;
}

/* ── pipeline stage config ──────────────────────────────── */
const pipelineStages = [
    { id: "finding", label: "Finding", icon: "lightbulb", description: "Original research finding extracted from source" },
    { id: "questions", label: "Verification Questions", icon: "quiz", description: "Generate targeted questions to verify the claim" },
    { id: "evidence", label: "Evidence Search", icon: "search", description: "Search for corroborating or contradicting evidence" },
    { id: "scoring", label: "Confidence Score", icon: "speed", description: "Calculate final confidence based on evidence" },
];

function getStageFromResult(result: VerificationResult): number {
    const status = result.verification_status?.toLowerCase();
    if (status === "verified" || status === "flagged" || status === "rejected") return 4; // all stages done
    if (status === "processing" || status === "in_progress") return 2;
    return 0;
}

function mapStatus(status: string | null): string {
    if (!status) return "pending";
    const s = status.toLowerCase();
    if (s === "verified" || s === "confirmed") return "verified";
    if (s === "rejected" || s === "contradicted") return "contradicted";
    if (s === "flagged") return "flagged";
    if (s === "processing" || s === "in_progress") return "processing";
    return "pending";
}

export default function VerificationPipelinePage() {
    const params = useParams();
    const sessionId = params.id as string;
    const [results, setResults] = useState<VerificationResult[]>([]);
    const [stats, setStats] = useState<VerificationStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [selectedIdx, setSelectedIdx] = useState(0);

    const fetchData = useCallback(async () => {
        setLoading(true);
        try {
            const [resResults, resStats] = await Promise.all([
                fetch(`/api/sessions/${sessionId}/verification/results?limit=200`),
                fetch(`/api/sessions/${sessionId}/verification/stats`),
            ]);
            if (resResults.ok) {
                const data = await resResults.json();
                setResults(data || []);
                setSelectedIdx(0);
            }
            if (resStats.ok) setStats(await resStats.json());
        } catch {
            // silently handle
        } finally {
            setLoading(false);
        }
    }, [sessionId]);

    useEffect(() => { fetchData(); }, [fetchData]);

    const selectedFinding = results[selectedIdx] || null;
    const activeStage = selectedFinding ? getStageFromResult(selectedFinding) : 0;

    const displayStats = stats || {
        total: results.length,
        verified: results.filter((r) => mapStatus(r.verification_status) === "verified").length,
        flagged: results.filter((r) => mapStatus(r.verification_status) === "flagged").length,
        rejected: results.filter((r) => mapStatus(r.verification_status) === "contradicted").length,
        pending: results.filter((r) => mapStatus(r.verification_status) === "pending").length,
        avg_confidence: null,
        avg_time_ms: null,
    };

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
                        <span className="text-gray-300">Verification</span>
                    </div>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <Link href={`/session/${sessionId}`} className="text-gray-400 hover:text-primary transition-colors">
                                <span className="material-symbols-outlined">arrow_back</span>
                            </Link>
                            <h1 className="text-xl font-bold">CoVe Verification Pipeline</h1>
                        </div>
                        <button onClick={fetchData} className="btn btn-ghost text-xs gap-1">
                            <span className="material-symbols-outlined text-sm">refresh</span>
                            Refresh
                        </button>
                    </div>
                </div>
            </header>

            {loading ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="flex items-center gap-3 text-gray-400">
                        <span className="material-symbols-outlined animate-spin">progress_activity</span>
                        Loading verification data...
                    </div>
                </div>
            ) : results.length === 0 ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <span className="material-symbols-outlined text-4xl text-gray-600 mb-3 block">verified</span>
                        <p className="text-sm text-gray-400">No verification results yet.</p>
                        <p className="text-xs text-gray-600 mt-1">Findings will be verified as they are discovered.</p>
                    </div>
                </div>
            ) : (
                <div className="flex-1 flex max-w-7xl mx-auto w-full">
                    {/* Left Sidebar: Queue */}
                    <aside className="w-72 shrink-0 border-r border-dark-border p-4 overflow-y-auto">
                        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
                            Results ({results.length})
                        </h3>
                        <div className="space-y-2">
                            {results.map((item, idx) => {
                                const isActive = selectedIdx === idx;
                                const status = mapStatus(item.verification_status);
                                const conf = item.verified_confidence ?? item.current_confidence;
                                return (
                                    <button
                                        key={item.id}
                                        onClick={() => setSelectedIdx(idx)}
                                        className={`w-full text-left p-3 rounded-xl transition-all border cursor-pointer ${isActive
                                            ? "bg-card-hover border-primary/40 ring-1 ring-primary/20"
                                            : "bg-dark-bg/60 border-dark-border hover:border-gray-600"
                                            }`}
                                    >
                                        <div className="flex items-center gap-2 mb-1.5">
                                            <StatusDot status={status} />
                                            <span className={`text-[10px] font-bold uppercase tracking-wider ${getStatusColor(status)}`}>
                                                {status}
                                            </span>
                                            {conf !== null && conf !== undefined && (
                                                <span className="text-[10px] font-mono text-gray-500 ml-auto">
                                                    {Math.round(conf * 100)}%
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-xs text-gray-300 line-clamp-2 leading-relaxed">
                                            {item.finding_content || `Finding #${item.finding_id}`}
                                        </p>
                                    </button>
                                );
                            })}
                        </div>
                    </aside>

                    {/* Main Content */}
                    <main className="flex-1 p-8 space-y-8 overflow-y-auto">
                        {/* Stats Dashboard */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <MiniStat icon="verified" label="Verified" value={displayStats.verified} color="text-accent-green" />
                            <MiniStat icon="flag" label="Flagged" value={displayStats.flagged} color="text-accent-yellow" />
                            <MiniStat icon="dangerous" label="Rejected" value={displayStats.rejected} color="text-accent-red" />
                            <MiniStat icon="hourglass_empty" label="Pending" value={displayStats.pending} color="text-gray-500" />
                        </div>

                        {/* Avg confidence & time */}
                        {(displayStats.avg_confidence !== null || displayStats.avg_time_ms !== null) && (
                            <div className="flex items-center gap-6 text-xs text-gray-500">
                                {displayStats.avg_confidence !== null && (
                                    <span className="flex items-center gap-1">
                                        <span className="material-symbols-outlined text-sm">speed</span>
                                        Avg confidence: <span className="font-mono text-gray-300">{Math.round(displayStats.avg_confidence * 100)}%</span>
                                    </span>
                                )}
                                {displayStats.avg_time_ms !== null && (
                                    <span className="flex items-center gap-1">
                                        <span className="material-symbols-outlined text-sm">timer</span>
                                        Avg time: <span className="font-mono text-gray-300">{Math.round(displayStats.avg_time_ms)}ms</span>
                                    </span>
                                )}
                            </div>
                        )}

                        {/* Selected Finding */}
                        {selectedFinding && (
                            <>
                                <div className="card">
                                    <div className="flex items-center gap-2 mb-2">
                                        <StatusDot status={mapStatus(selectedFinding.verification_status)} />
                                        <span className={`text-xs font-bold uppercase tracking-wider ${getStatusColor(mapStatus(selectedFinding.verification_status))}`}>
                                            {mapStatus(selectedFinding.verification_status)}
                                        </span>
                                        {selectedFinding.verification_method && (
                                            <span className="badge badge-system ml-2">{selectedFinding.verification_method}</span>
                                        )}
                                    </div>
                                    <p className="text-sm text-gray-200 leading-relaxed">
                                        {selectedFinding.finding_content || `Finding #${selectedFinding.finding_id}`}
                                    </p>
                                    {selectedFinding.source_url && (
                                        <a
                                            href={selectedFinding.source_url}
                                            target="_blank"
                                            rel="noreferrer"
                                            className="text-xs text-info hover:underline mt-2 block truncate"
                                        >
                                            {selectedFinding.source_url}
                                        </a>
                                    )}
                                </div>

                                {/* Pipeline Flow */}
                                <div>
                                    <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-6">
                                        Verification Pipeline
                                    </h3>
                                    <div className="flex items-start gap-0">
                                        {pipelineStages.map((stage, i) => {
                                            const isCompleted = i < activeStage;
                                            const isActive = i === activeStage && activeStage < 4;
                                            return (
                                                <div key={stage.id} className="flex items-start flex-1">
                                                    <div className={`flex-1 relative rounded-xl p-5 border transition-all ${isActive
                                                        ? "bg-primary/10 border-primary/40 ring-1 ring-primary/20"
                                                        : isCompleted
                                                            ? "bg-accent-green/5 border-accent-green/30"
                                                            : "bg-dark-bg/60 border-dark-border"
                                                        }`}>
                                                        {isActive && (
                                                            <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-primary to-transparent animate-flow-pulse" />
                                                        )}
                                                        <div className="flex items-center gap-2 mb-3">
                                                            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${isCompleted ? "bg-accent-green/10" : isActive ? "bg-primary/20" : "bg-dark-border"}`}>
                                                                <span className={`material-symbols-outlined text-base ${isCompleted ? "text-accent-green" : isActive ? "text-primary" : "text-gray-600"}`}>
                                                                    {isCompleted ? "check_circle" : stage.icon}
                                                                </span>
                                                            </div>
                                                            <div>
                                                                <p className={`text-xs font-semibold ${isCompleted ? "text-accent-green" : isActive ? "text-primary" : "text-gray-500"}`}>
                                                                    {stage.label}
                                                                </p>
                                                                {isActive && (
                                                                    <span className="text-[10px] text-primary animate-pulse">Processing...</span>
                                                                )}
                                                            </div>
                                                        </div>
                                                        <p className="text-xs text-gray-500 leading-relaxed">{stage.description}</p>

                                                        {isCompleted && stage.id === "finding" && (
                                                            <div className="mt-3 text-xs text-accent-green flex items-center gap-1">
                                                                <span className="material-symbols-outlined text-sm">check</span>
                                                                Finding captured
                                                            </div>
                                                        )}
                                                    </div>

                                                    {i < pipelineStages.length - 1 && (
                                                        <div className="flex items-center pt-8 mx-2">
                                                            <div className={`w-8 h-0.5 rounded ${isCompleted ? "bg-accent-green" : "bg-dark-border"}`} />
                                                            <span className={`material-symbols-outlined text-sm ${isCompleted ? "text-accent-green" : "text-gray-600"}`}>
                                                                chevron_right
                                                            </span>
                                                        </div>
                                                    )}
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>

                                {/* Detail Metrics */}
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <MetricCard label="KG Support" value={selectedFinding.kg_support_score} format="pct" />
                                    <MetricCard label="Consistency" value={selectedFinding.consistency_score} format="pct" />
                                    <MetricCard label="KG Entities" value={selectedFinding.kg_entity_matches} format="num" />
                                    <MetricCard label="CRITIC Iterations" value={selectedFinding.critic_iterations} format="num" />
                                </div>

                                {/* Confidence Ring */}
                                {(selectedFinding.verified_confidence ?? selectedFinding.current_confidence) !== null && (() => {
                                    const conf = (selectedFinding.verified_confidence ?? selectedFinding.current_confidence) as number;
                                    return (
                                        <div className="card flex items-center gap-8">
                                            <div className="relative w-24 h-24">
                                                <svg className="w-24 h-24" viewBox="0 0 100 100">
                                                    <circle cx="50" cy="50" r="42" fill="none" stroke="currentColor" strokeWidth="6" className="text-dark-border" />
                                                    <circle
                                                        cx="50" cy="50" r="42" fill="none" strokeWidth="6" strokeLinecap="round"
                                                        strokeDasharray={`${Math.round(conf * 264)} 264`}
                                                        transform="rotate(-90 50 50)"
                                                        className={conf >= 0.7 ? "text-accent-green" : conf >= 0.4 ? "text-accent-yellow" : "text-accent-red"}
                                                        stroke="currentColor"
                                                    />
                                                </svg>
                                                <div className="absolute inset-0 flex items-center justify-center">
                                                    <span className="text-xl font-bold font-mono">{Math.round(conf * 100)}%</span>
                                                </div>
                                            </div>
                                            <div>
                                                <h4 className="font-semibold mb-1">
                                                    {selectedFinding.verified_confidence !== null ? "Verified" : "Original"} Confidence
                                                </h4>
                                                <p className="text-sm text-gray-400">
                                                    {conf >= 0.7
                                                        ? "High confidence — finding is well-supported by evidence."
                                                        : conf >= 0.4
                                                            ? "Moderate confidence — some supporting evidence found."
                                                            : "Low confidence — conflicting evidence or insufficient data."
                                                    }
                                                </p>
                                                {selectedFinding.original_confidence !== null && selectedFinding.verified_confidence !== null && (
                                                    <p className="text-xs text-gray-500 mt-1">
                                                        Original: {Math.round(selectedFinding.original_confidence * 100)}% → Verified: {Math.round(selectedFinding.verified_confidence * 100)}%
                                                    </p>
                                                )}
                                            </div>
                                        </div>
                                    );
                                })()}

                                {/* Error */}
                                {selectedFinding.error && (
                                    <div className="card border-accent-red/30 bg-accent-red/5">
                                        <div className="flex items-center gap-2 text-accent-red text-xs font-semibold mb-1">
                                            <span className="material-symbols-outlined text-sm">error</span>
                                            Verification Error
                                        </div>
                                        <p className="text-xs text-gray-400">{selectedFinding.error}</p>
                                    </div>
                                )}
                            </>
                        )}
                    </main>
                </div>
            )}
        </div>
    );
}

/* ── sub-components ─────────────────────────────────────── */
function StatusDot({ status }: { status: string }) {
    const colors: Record<string, string> = {
        verified: "bg-accent-green",
        processing: "bg-primary animate-pulse",
        pending: "bg-gray-500",
        contradicted: "bg-accent-red",
        flagged: "bg-accent-yellow",
    };
    return <span className={`w-2 h-2 rounded-full ${colors[status] || "bg-gray-500"}`} />;
}

function MiniStat({ icon, label, value, color }: { icon: string; label: string; value: number; color: string }) {
    return (
        <div className="card py-4">
            <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-500">{label}</span>
                <span className={`material-symbols-outlined text-lg ${color}`}>{icon}</span>
            </div>
            <span className="font-mono text-2xl font-bold">{value}</span>
        </div>
    );
}

function MetricCard({ label, value, format }: { label: string; value: number | null; format: "pct" | "num" }) {
    const display = value === null || value === undefined
        ? "—"
        : format === "pct"
            ? `${Math.round(value * 100)}%`
            : String(value);

    return (
        <div className="card py-3 text-center">
            <p className="text-xs text-gray-500 mb-1">{label}</p>
            <p className="font-mono text-lg font-bold">{display}</p>
        </div>
    );
}

function getStatusColor(status: string): string {
    switch (status) {
        case "verified": return "text-accent-green";
        case "processing": return "text-primary";
        case "contradicted": return "text-accent-red";
        case "flagged": return "text-accent-yellow";
        default: return "text-gray-500";
    }
}
