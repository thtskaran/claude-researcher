"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

// Mock data for the verification pipeline (will be wired to API later)
const mockQueue = [
    { id: "f1", content: "Solar panel adoption grew 45% in Southeast Asia in 2024", status: "verified", confidence: 0.92 },
    { id: "f2", content: "Government subsidies account for 60% of renewable investment", status: "processing", confidence: null },
    { id: "f3", content: "Carbon emissions reduced by 12% due to solar adoption", status: "pending", confidence: null },
    { id: "f4", content: "Grid integration costs decreased 30% year-over-year", status: "verified", confidence: 0.78 },
    { id: "f5", content: "Rural electrification reached 90% in Thailand", status: "contradicted", confidence: 0.35 },
    { id: "f6", content: "Battery storage capacity doubled in the ASEAN region", status: "pending", confidence: null },
];

const pipelineStages = [
    { id: "finding", label: "Finding", icon: "lightbulb", description: "Original research finding extracted from source" },
    { id: "questions", label: "Verification Questions", icon: "quiz", description: "Generate targeted questions to verify the claim" },
    { id: "evidence", label: "Evidence Search", icon: "search", description: "Search for corroborating or contradicting evidence" },
    { id: "scoring", label: "Confidence Score", icon: "speed", description: "Calculate final confidence based on evidence" },
];

export default function VerificationPipelinePage() {
    const params = useParams();
    const sessionId = params.id as string;
    const [selectedFinding, setSelectedFinding] = useState(mockQueue[1]); // Select the "processing" one

    const stats = {
        total: mockQueue.length,
        verified: mockQueue.filter((f) => f.status === "verified").length,
        contradicted: mockQueue.filter((f) => f.status === "contradicted").length,
        processing: mockQueue.filter((f) => f.status === "processing").length,
        pending: mockQueue.filter((f) => f.status === "pending").length,
    };

    const getActiveStage = () => {
        if (selectedFinding.status === "verified" || selectedFinding.status === "contradicted") return 4;
        if (selectedFinding.status === "processing") return 2;
        return 0;
    };
    const activeStage = getActiveStage();

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
                    <div className="flex items-center gap-3">
                        <Link href={`/session/${sessionId}`} className="text-gray-400 hover:text-primary transition-colors">
                            <span className="material-symbols-outlined">arrow_back</span>
                        </Link>
                        <h1 className="text-xl font-bold">CoVe Verification Pipeline</h1>
                    </div>
                </div>
            </header>

            <div className="flex-1 flex max-w-7xl mx-auto w-full">
                {/* Left Sidebar: Queue */}
                <aside className="w-72 shrink-0 border-r border-dark-border p-4 overflow-y-auto">
                    <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
                        Pending Queue
                    </h3>
                    <div className="space-y-2">
                        {mockQueue.map((item) => {
                            const isActive = selectedFinding.id === item.id;
                            return (
                                <button
                                    key={item.id}
                                    onClick={() => setSelectedFinding(item)}
                                    className={`w-full text-left p-3 rounded-xl transition-all border cursor-pointer ${isActive
                                            ? "bg-card-hover border-primary/40 ring-1 ring-primary/20"
                                            : "bg-dark-bg/60 border-dark-border hover:border-gray-600"
                                        }`}
                                >
                                    <div className="flex items-center gap-2 mb-1.5">
                                        <StatusDot status={item.status} />
                                        <span className={`text-[10px] font-bold uppercase tracking-wider ${getStatusColor(item.status)}`}>
                                            {item.status}
                                        </span>
                                        {item.confidence !== null && (
                                            <span className="text-[10px] font-mono text-gray-500 ml-auto">
                                                {Math.round(item.confidence * 100)}%
                                            </span>
                                        )}
                                    </div>
                                    <p className="text-xs text-gray-300 line-clamp-2 leading-relaxed">{item.content}</p>
                                </button>
                            );
                        })}
                    </div>
                </aside>

                {/* Main Content */}
                <main className="flex-1 p-8 space-y-8 overflow-y-auto">
                    {/* Stats Dashboard */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <MiniStat icon="verified" label="Verified" value={stats.verified} color="text-accent-green" />
                        <MiniStat icon="dangerous" label="Contradicted" value={stats.contradicted} color="text-accent-red" />
                        <MiniStat icon="pending" label="Processing" value={stats.processing} color="text-primary" />
                        <MiniStat icon="hourglass_empty" label="Pending" value={stats.pending} color="text-gray-500" />
                    </div>

                    {/* Selected Finding */}
                    <div className="card">
                        <div className="flex items-center gap-2 mb-2">
                            <StatusDot status={selectedFinding.status} />
                            <span className={`text-xs font-bold uppercase tracking-wider ${getStatusColor(selectedFinding.status)}`}>
                                {selectedFinding.status}
                            </span>
                        </div>
                        <p className="text-sm text-gray-200 leading-relaxed">{selectedFinding.content}</p>
                    </div>

                    {/* Pipeline Flow */}
                    <div>
                        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-6">
                            Verification Pipeline
                        </h3>
                        <div className="flex items-start gap-0">
                            {pipelineStages.map((stage, i) => {
                                const isCompleted = i < activeStage;
                                const isActive = i === activeStage;
                                const isPending = i > activeStage;

                                return (
                                    <div key={stage.id} className="flex items-start flex-1">
                                        {/* Stage Card */}
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
                                                <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${isCompleted ? "bg-accent-green/10" : isActive ? "bg-primary/20" : "bg-dark-border"
                                                    }`}>
                                                    <span className={`material-symbols-outlined text-base ${isCompleted ? "text-accent-green" : isActive ? "text-primary" : "text-gray-600"
                                                        }`}>
                                                        {isCompleted ? "check_circle" : stage.icon}
                                                    </span>
                                                </div>
                                                <div>
                                                    <p className={`text-xs font-semibold ${isCompleted ? "text-accent-green" : isActive ? "text-primary" : "text-gray-500"
                                                        }`}>
                                                        {stage.label}
                                                    </p>
                                                    {isActive && (
                                                        <span className="text-[10px] text-primary animate-pulse">Processing...</span>
                                                    )}
                                                </div>
                                            </div>
                                            <p className="text-xs text-gray-500 leading-relaxed">{stage.description}</p>

                                            {/* Active stage shows mock content */}
                                            {isActive && stage.id === "questions" && (
                                                <div className="mt-3 space-y-1.5 bg-dark-bg/50 rounded-lg p-3">
                                                    <p className="text-xs text-gray-300">1. What are the primary data sources for this claim?</p>
                                                    <p className="text-xs text-gray-300">2. Are there contradicting studies or reports?</p>
                                                    <p className="text-xs text-gray-400 animate-pulse">3. Generating question...</p>
                                                </div>
                                            )}

                                            {isCompleted && stage.id === "finding" && (
                                                <div className="mt-3 text-xs text-accent-green flex items-center gap-1">
                                                    <span className="material-symbols-outlined text-sm">check</span>
                                                    Finding captured
                                                </div>
                                            )}
                                        </div>

                                        {/* Connector */}
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

                    {/* Confidence Ring (for completed items) */}
                    {selectedFinding.confidence !== null && (
                        <div className="card flex items-center gap-8">
                            <div className="relative w-24 h-24">
                                <svg className="w-24 h-24" viewBox="0 0 100 100">
                                    <circle cx="50" cy="50" r="42" fill="none" stroke="currentColor" strokeWidth="6" className="text-dark-border" />
                                    <circle
                                        cx="50"
                                        cy="50"
                                        r="42"
                                        fill="none"
                                        strokeWidth="6"
                                        strokeLinecap="round"
                                        strokeDasharray={`${Math.round(selectedFinding.confidence * 264)} 264`}
                                        transform="rotate(-90 50 50)"
                                        className={selectedFinding.confidence >= 0.7 ? "text-accent-green" : selectedFinding.confidence >= 0.4 ? "text-accent-yellow" : "text-accent-red"}
                                        stroke="currentColor"
                                    />
                                </svg>
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <span className="text-xl font-bold font-mono">{Math.round(selectedFinding.confidence * 100)}%</span>
                                </div>
                            </div>
                            <div>
                                <h4 className="font-semibold mb-1">Final Confidence Score</h4>
                                <p className="text-sm text-gray-400">
                                    {selectedFinding.confidence >= 0.7
                                        ? "High confidence — finding is well-supported by multiple sources."
                                        : selectedFinding.confidence >= 0.4
                                            ? "Moderate confidence — some supporting evidence found."
                                            : "Low confidence — conflicting evidence or insufficient data."
                                    }
                                </p>
                            </div>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
}

function StatusDot({ status }: { status: string }) {
    const colors: Record<string, string> = {
        verified: "bg-accent-green",
        processing: "bg-primary animate-pulse",
        pending: "bg-gray-500",
        contradicted: "bg-accent-red",
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

function getStatusColor(status: string): string {
    switch (status) {
        case "verified": return "text-accent-green";
        case "processing": return "text-primary";
        case "contradicted": return "text-accent-red";
        default: return "text-gray-500";
    }
}
