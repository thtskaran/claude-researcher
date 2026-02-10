"use client";

import { useState, useEffect, useRef } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { ResearchWebSocket, AgentEvent } from "@/lib/websocket";

interface AgentInfo {
    name: string;
    role: string;
    level: number;
    icon: string;
    status: "active" | "thinking" | "idle" | "searching" | "reading";
    goal: string;
    progress: number;
    lastAction: string;
    currentQuery?: string;
}

export default function AgentTransparencyPage() {
    const params = useParams();
    const sessionId = params.id as string;
    const [events, setEvents] = useState<AgentEvent[]>([]);
    const [connected, setConnected] = useState(false);
    const wsRef = useRef<ResearchWebSocket | null>(null);

    // Default agent state from design
    const [agents, setAgents] = useState<AgentInfo[]>([
        {
            name: "Director",
            role: "Level 0 — Strategic Planning",
            level: 0,
            icon: "military_tech",
            status: "active",
            goal: "Orchestrate research strategy and synthesize final output",
            progress: 45,
            lastAction: "Delegated sub-topics to Manager agents",
        },
        {
            name: "Manager",
            role: "Level 1 — Tactical Coordination",
            level: 1,
            icon: "psychology",
            status: "thinking",
            goal: "Coordinate intern search tasks and evaluate findings",
            progress: 62,
            lastAction: "Evaluating search results from Intern-1",
            currentQuery: "renewable energy adoption rates southeast asia 2024",
        },
        {
            name: "Intern-1",
            role: "Level 2 — Research Worker",
            level: 2,
            icon: "person_search",
            status: "searching",
            goal: "Search for primary sources on solar panel adoption",
            progress: 80,
            lastAction: "Extracted 3 findings from IEA report",
            currentQuery: "solar panel installation growth 2023-2024",
        },
        {
            name: "Intern-2",
            role: "Level 2 — Research Worker",
            level: 2,
            icon: "person_search",
            status: "reading",
            goal: "Analyze government subsidy programs in ASEAN",
            progress: 35,
            lastAction: "Reading government policy documents",
            currentQuery: "ASEAN renewable energy subsidies policy",
        },
        {
            name: "Intern-3",
            role: "Level 2 — Research Worker",
            level: 2,
            icon: "person_search",
            status: "idle",
            goal: "Awaiting next assignment from Manager",
            progress: 0,
            lastAction: "Completed market analysis task",
        },
    ]);

    useEffect(() => {
        const ws = new ResearchWebSocket(sessionId);

        ws.onEvent((event) => {
            setEvents((prev) => [event, ...prev].slice(0, 500));
            updateAgentFromEvent(event);
        });

        ws.connect();
        wsRef.current = ws;

        const checkConnection = setInterval(() => setConnected(ws.isConnected()), 1000);
        return () => { clearInterval(checkConnection); ws.disconnect(); };
    }, [sessionId]);

    const updateAgentFromEvent = (event: AgentEvent) => {
        setAgents((prev) => {
            const agent = event.agent?.toLowerCase() || "";
            return prev.map((a) => {
                if (!agent.includes(a.name.toLowerCase().replace("-", ""))) return a;
                const data = event.data || {};
                return {
                    ...a,
                    lastAction: data.message || data.action || data.thought || a.lastAction,
                    status: event.event_type === "thinking" ? "thinking" : event.event_type === "action" ? "searching" : a.status,
                    currentQuery: data.query || a.currentQuery,
                };
            });
        });
    };

    const director = agents.find((a) => a.level === 0);
    const manager = agents.find((a) => a.level === 1);
    const interns = agents.filter((a) => a.level === 2);

    const statusConfig: Record<string, { color: string; bg: string; label: string }> = {
        active: { color: "text-accent-green", bg: "bg-accent-green", label: "Active" },
        thinking: { color: "text-primary", bg: "bg-primary", label: "Thinking" },
        searching: { color: "text-accent-blue", bg: "bg-accent-blue", label: "Searching" },
        reading: { color: "text-accent-yellow", bg: "bg-accent-yellow", label: "Reading" },
        idle: { color: "text-gray-500", bg: "bg-gray-500", label: "Idle" },
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
                        <span className="text-gray-300">Agents</span>
                    </div>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <Link href={`/session/${sessionId}`} className="text-gray-400 hover:text-primary transition-colors">
                                <span className="material-symbols-outlined">arrow_back</span>
                            </Link>
                            <h1 className="text-xl font-bold">Agent Transparency</h1>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className={`status-dot ${connected ? "status-dot-active" : "status-dot-idle"}`} />
                            <span className="text-xs text-gray-400">{connected ? "Live" : "Disconnected"}</span>
                        </div>
                    </div>
                </div>
            </header>

            <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8 space-y-8">
                {/* Director */}
                {director && (
                    <AgentSection agent={director} statusConfig={statusConfig} level="director">
                        <div className="mt-4 space-y-3">
                            <div>
                                <div className="flex justify-between text-xs text-gray-500 mb-1">
                                    <span>Overall Progress</span>
                                    <span className="font-mono">{director.progress}%</span>
                                </div>
                                <div className="h-2 bg-dark-border rounded-full overflow-hidden">
                                    <div className="h-full bg-gradient-to-r from-primary to-primary-light rounded-full transition-all" style={{ width: `${director.progress}%` }} />
                                </div>
                            </div>
                            <div className="flex items-center gap-4 text-xs text-gray-500">
                                <span className="flex items-center gap-1"><span className="material-symbols-outlined text-sm">schedule</span>Phase: Research & Analysis</span>
                                <span className="flex items-center gap-1"><span className="material-symbols-outlined text-sm">group</span>{interns.length} interns active</span>
                            </div>
                        </div>
                    </AgentSection>
                )}

                {/* Connector line */}
                <div className="flex justify-center"><div className="w-px h-8 bg-dark-border" /></div>

                {/* Manager */}
                {manager && (
                    <AgentSection agent={manager} statusConfig={statusConfig} level="manager">
                        <div className="mt-4 space-y-3">
                            {manager.currentQuery && (
                                <div className="bg-terminal-black border border-terminal-border rounded-lg p-3 font-mono text-xs text-gray-300">
                                    <span className="text-primary mr-2">query:</span>{manager.currentQuery}
                                </div>
                            )}
                            <div className="bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                                <p className="text-gray-500 uppercase tracking-wider font-medium mb-2">Reasoning Trace</p>
                                <p className="text-gray-400">1. Received research goal from Director</p>
                                <p className="text-gray-400">2. Decomposed into 3 sub-tasks for interns</p>
                                <p className="text-gray-300">3. Evaluating results from Intern-1...</p>
                            </div>
                        </div>
                    </AgentSection>
                )}

                {/* Connector line */}
                <div className="flex justify-center"><div className="w-px h-8 bg-dark-border" /></div>

                {/* Intern Grid */}
                <div>
                    <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                        <span className="material-symbols-outlined text-sm">group</span>
                        Intern Pool
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {interns.map((intern) => {
                            const cfg = statusConfig[intern.status] || statusConfig.idle;
                            return (
                                <div key={intern.name} className="card card-hover">
                                    <div className="flex items-center justify-between mb-3">
                                        <div className="flex items-center gap-2">
                                            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                                                <span className="material-symbols-outlined text-primary text-base">{intern.icon}</span>
                                            </div>
                                            <div>
                                                <p className="text-sm font-medium">{intern.name}</p>
                                                <p className="text-xs text-gray-500">{intern.role}</p>
                                            </div>
                                        </div>
                                        <span className={`badge ${cfg.color} ${cfg.bg}/10 border-current/20`}>
                                            {cfg.label}
                                        </span>
                                    </div>

                                    <p className="text-xs text-gray-400 mb-3">{intern.goal}</p>

                                    {intern.progress > 0 && (
                                        <div className="mb-3">
                                            <div className="h-1 bg-dark-border rounded-full overflow-hidden">
                                                <div className={`h-full rounded-full ${cfg.bg}`} style={{ width: `${intern.progress}%` }} />
                                            </div>
                                        </div>
                                    )}

                                    {intern.currentQuery && (
                                        <div className="bg-dark-bg rounded p-2 text-xs font-mono text-gray-400 truncate mb-2">
                                            <span className="text-primary mr-1">›</span>{intern.currentQuery}
                                        </div>
                                    )}

                                    <p className="text-xs text-gray-500 truncate">
                                        <span className="material-symbols-outlined text-[12px] mr-1 align-middle">history</span>
                                        {intern.lastAction}
                                    </p>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Recent Activity */}
                {events.length > 0 && (
                    <div className="card">
                        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Recent Agent Events</h3>
                        <div className="space-y-2 max-h-60 overflow-y-auto scrollbar-hide">
                            {events.slice(0, 20).map((event, i) => (
                                <div key={i} className="flex items-start gap-3 text-xs py-1">
                                    <span className="font-mono text-gray-600 shrink-0 w-16">
                                        {new Date(event.timestamp).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                                    </span>
                                    <span className={`badge ${getBadgeClass(event.event_type)}`}>{event.event_type}</span>
                                    <span className="text-gray-500">{event.agent}</span>
                                    <span className="text-gray-400 truncate">{getEventText(event)}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </main>

            {/* Footer Status Bar */}
            <footer className="border-t border-dark-border bg-terminal-black px-6 py-2">
                <div className="max-w-7xl mx-auto flex items-center justify-between text-xs font-mono text-gray-500">
                    <span>Session: {sessionId.slice(0, 8)}</span>
                    <span>{events.length} events captured</span>
                    <span className="flex items-center gap-2">
                        <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-accent-green" : "bg-gray-600"}`} />
                        WebSocket {connected ? "active" : "inactive"}
                    </span>
                </div>
            </footer>
        </div>
    );
}

function AgentSection({
    agent,
    statusConfig,
    level,
    children,
}: {
    agent: AgentInfo;
    statusConfig: Record<string, { color: string; bg: string; label: string }>;
    level: string;
    children?: React.ReactNode;
}) {
    const cfg = statusConfig[agent.status] || statusConfig.idle;
    return (
        <section>
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-2">
                <span className="material-symbols-outlined text-sm">{agent.icon}</span>
                {level === "director" ? "Director" : "Manager"}
            </h2>
            <div className={`card relative overflow-hidden ${level === "manager" ? "border-primary/30" : ""}`}>
                {agent.status === "thinking" && (
                    <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-primary to-transparent animate-flow-pulse" />
                )}
                <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
                            <span className="material-symbols-outlined text-primary text-xl">{agent.icon}</span>
                        </div>
                        <div>
                            <p className="font-semibold">{agent.name}</p>
                            <p className="text-xs text-gray-500">{agent.role}</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className={`status-dot ${cfg.bg} ${agent.status === "thinking" ? "animate-pulse" : ""}`} />
                        <span className={`text-xs font-medium ${cfg.color}`}>{cfg.label}</span>
                    </div>
                </div>
                <p className="text-sm text-gray-300 mb-2">{agent.goal}</p>
                <div className="flex items-center gap-2 text-xs text-gray-500">
                    <span className="material-symbols-outlined text-[14px]">history</span>
                    {agent.lastAction}
                </div>
                {children}
            </div>
        </section>
    );
}

function getBadgeClass(eventType: string): string {
    switch (eventType) {
        case "thinking": return "badge-thinking";
        case "action": return "badge-action";
        case "finding": return "badge-finding";
        case "error": return "badge-error";
        default: return "badge-system";
    }
}

function getEventText(event: AgentEvent): string {
    const d = event.data || {};
    return d.message || d.thought || d.action || d.content || JSON.stringify(d).slice(0, 80);
}
