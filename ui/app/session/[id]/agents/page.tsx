"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { ResearchWebSocket, AgentEvent } from "@/lib/websocket";

/* ── types ─────────────────────────────────────────────── */
interface AgentDecision {
    id: number;
    agent_role: string;
    decision_type: string;
    decision_outcome: string;
    reasoning: string | null;
    inputs: Record<string, unknown> | null;
    metrics: Record<string, unknown> | null;
    iteration: number | null;
    created_at: string;
}

interface TopicItem {
    id: number;
    topic: string;
    parent_topic_id: number | null;
    depth: number;
    status: string;
    priority: number;
    findings_count: number;
    assigned_at: string | null;
    completed_at: string | null;
}

interface HierarchyData {
    roles: Record<string, {
        role: string;
        decision_count: number;
        last_action: string | null;
        last_action_at: string | null;
        recent_decisions: AgentDecision[];
    }>;
    topics: TopicItem[];
    progress: {
        total_topics: number;
        completed: number;
        in_progress: number;
        pending: number;
    };
}

/* ── role config ───────────────────────────────────────── */
const roleConfig: Record<string, { icon: string; label: string; sublabel: string; level: number }> = {
    director: { icon: "military_tech", label: "Director", sublabel: "Level 0 — Strategic Planning", level: 0 },
    manager: { icon: "psychology", label: "Manager", sublabel: "Level 1 — Tactical Coordination", level: 1 },
    intern: { icon: "person_search", label: "Intern", sublabel: "Level 2 — Research Worker", level: 2 },
    researcher: { icon: "person_search", label: "Researcher", sublabel: "Level 2 — Research Worker", level: 2 },
    research_intern: { icon: "person_search", label: "Research Intern", sublabel: "Level 2 — Research Worker", level: 2 },
};
const defaultRoleCfg = { icon: "smart_toy", label: "Agent", sublabel: "Unknown Role", level: 3 };

export default function AgentTransparencyPage() {
    const params = useParams();
    const sessionId = params.id as string;
    const [events, setEvents] = useState<AgentEvent[]>([]);
    const [connected, setConnected] = useState(false);
    const wsRef = useRef<ResearchWebSocket | null>(null);
    const [hierarchy, setHierarchy] = useState<HierarchyData | null>(null);
    const [decisions, setDecisions] = useState<AgentDecision[]>([]);
    const [loading, setLoading] = useState(true);

    /* ── fetch data ────────────────────────────────────── */
    const fetchData = useCallback(async () => {
        setLoading(true);
        try {
            const [hierRes, decRes] = await Promise.all([
                fetch(`/api/sessions/${sessionId}/agents/hierarchy`),
                fetch(`/api/sessions/${sessionId}/agents/decisions?limit=100`),
            ]);
            if (hierRes.ok) setHierarchy(await hierRes.json());
            if (decRes.ok) setDecisions(await decRes.json());
        } catch {
            // silently fall back to empty state
        } finally {
            setLoading(false);
        }
    }, [sessionId]);

    useEffect(() => { fetchData(); }, [fetchData]);

    /* ── WebSocket ─────────────────────────────────────── */
    useEffect(() => {
        const ws = new ResearchWebSocket(sessionId);

        ws.onEvent((event) => {
            setEvents((prev) => [event, ...prev].slice(0, 500));
        });

        ws.connect();
        wsRef.current = ws;

        const checkConnection = setInterval(() => setConnected(ws.isConnected()), 1000);
        return () => { clearInterval(checkConnection); ws.disconnect(); };
    }, [sessionId]);

    /* ── derived data ──────────────────────────────────── */
    const roles = hierarchy?.roles || {};
    const progress = hierarchy?.progress || { total_topics: 0, completed: 0, in_progress: 0, pending: 0 };
    const topics = hierarchy?.topics || [];
    const progressPct = progress.total_topics > 0 ? Math.round((progress.completed / progress.total_topics) * 100) : 0;

    // Sort roles by level for display
    const sortedRoles = Object.entries(roles).sort(([a], [b]) => {
        const la = (roleConfig[a.toLowerCase()] || defaultRoleCfg).level;
        const lb = (roleConfig[b.toLowerCase()] || defaultRoleCfg).level;
        return la - lb;
    });

    const directorRole = sortedRoles.find(([k]) => k.toLowerCase() === "director");
    const managerRole = sortedRoles.find(([k]) => k.toLowerCase() === "manager");
    const workerRoles = sortedRoles.filter(([k]) => {
        const l = k.toLowerCase();
        return l !== "director" && l !== "manager";
    });

    const statusConfig: Record<string, { color: string; bg: string; label: string }> = {
        active: { color: "text-accent-green", bg: "bg-accent-green", label: "Active" },
        thinking: { color: "text-primary", bg: "bg-primary", label: "Thinking" },
        searching: { color: "text-accent-blue", bg: "bg-accent-blue", label: "Searching" },
        reading: { color: "text-accent-yellow", bg: "bg-accent-yellow", label: "Reading" },
        idle: { color: "text-gray-500", bg: "bg-gray-500", label: "Idle" },
    };

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
                        <span className="text-gray-300">Agents</span>
                    </div>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <Link href={`/session/${sessionId}`} className="text-gray-400 hover:text-primary transition-colors">
                                <span className="material-symbols-outlined">arrow_back</span>
                            </Link>
                            <h1 className="text-xl font-bold">Agent Transparency</h1>
                        </div>
                        <div className="flex items-center gap-3">
                            <button onClick={fetchData} className="btn btn-ghost text-xs gap-1">
                                <span className="material-symbols-outlined text-sm">refresh</span>
                                Refresh
                            </button>
                            <div className="flex items-center gap-2">
                                <span className={`status-dot ${connected ? "status-dot-active" : "status-dot-idle"}`} />
                                <span className="text-xs text-gray-400">{connected ? "Live" : "Disconnected"}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            {loading ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="flex items-center gap-3 text-gray-400">
                        <span className="material-symbols-outlined animate-spin">progress_activity</span>
                        Loading agent data...
                    </div>
                </div>
            ) : (
                <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8 space-y-8">
                    {/* Progress Overview */}
                    {progress.total_topics > 0 && (
                        <div className="card">
                            <div className="flex justify-between text-xs text-gray-500 mb-1">
                                <span>Topic Progress</span>
                                <span className="font-mono">{progress.completed}/{progress.total_topics} completed ({progressPct}%)</span>
                            </div>
                            <div className="h-2 bg-dark-border rounded-full overflow-hidden">
                                <div className="h-full bg-gradient-to-r from-primary to-primary-light rounded-full transition-all" style={{ width: `${progressPct}%` }} />
                            </div>
                            <div className="flex items-center gap-4 text-xs text-gray-500 mt-2">
                                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-accent-green" />{progress.completed} completed</span>
                                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-primary" />{progress.in_progress} in progress</span>
                                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-gray-500" />{progress.pending} pending</span>
                            </div>
                        </div>
                    )}

                    {/* Director */}
                    {directorRole && (
                        <RoleSection
                            roleName={directorRole[0]}
                            roleData={directorRole[1]}
                            level="director"
                            statusConfig={statusConfig}
                            topicCount={progress.total_topics}
                            workerCount={workerRoles.length}
                        />
                    )}

                    {(directorRole || managerRole) && (
                        <div className="flex justify-center"><div className="w-px h-8 bg-dark-border" /></div>
                    )}

                    {/* Manager */}
                    {managerRole && (
                        <RoleSection
                            roleName={managerRole[0]}
                            roleData={managerRole[1]}
                            level="manager"
                            statusConfig={statusConfig}
                        />
                    )}

                    {managerRole && workerRoles.length > 0 && (
                        <div className="flex justify-center"><div className="w-px h-8 bg-dark-border" /></div>
                    )}

                    {/* Workers Grid */}
                    {workerRoles.length > 0 && (
                        <div>
                            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                                <span className="material-symbols-outlined text-sm">group</span>
                                Worker Pool ({workerRoles.length})
                            </h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                {workerRoles.map(([name, data]) => {
                                    const cfg = roleConfig[name.toLowerCase()] || defaultRoleCfg;
                                    return (
                                        <div key={name} className="card card-hover">
                                            <div className="flex items-center justify-between mb-3">
                                                <div className="flex items-center gap-2">
                                                    <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                                                        <span className="material-symbols-outlined text-primary text-base">{cfg.icon}</span>
                                                    </div>
                                                    <div>
                                                        <p className="text-sm font-medium">{name}</p>
                                                        <p className="text-xs text-gray-500">{data.decision_count} decisions</p>
                                                    </div>
                                                </div>
                                            </div>
                                            {data.last_action && (
                                                <p className="text-xs text-gray-400 truncate">
                                                    <span className="material-symbols-outlined text-[12px] mr-1 align-middle">history</span>
                                                    {data.last_action}
                                                </p>
                                            )}
                                            {data.recent_decisions.length > 0 && (
                                                <div className="mt-3 space-y-1">
                                                    {data.recent_decisions.slice(0, 3).map((d) => (
                                                        <div key={d.id} className="text-xs text-gray-500 truncate">
                                                            <span className="text-gray-600 font-mono mr-1">{d.decision_type}:</span>
                                                            {d.decision_outcome}
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Topics */}
                    {topics.length > 0 && (
                        <div className="card">
                            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Topic Assignments</h3>
                            <div className="space-y-2 max-h-60 overflow-y-auto scrollbar-hide">
                                {topics.map((topic) => (
                                    <div key={topic.id} className="flex items-center gap-3 text-xs py-1" style={{ paddingLeft: `${topic.depth * 16}px` }}>
                                        <span className={`w-2 h-2 rounded-full shrink-0 ${topic.status === "completed" ? "bg-accent-green" : topic.status === "in_progress" ? "bg-primary" : "bg-gray-600"}`} />
                                        <span className="text-gray-300 truncate flex-1">{topic.topic}</span>
                                        <span className="text-gray-600 font-mono shrink-0">{topic.findings_count} findings</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Decision Log */}
                    {decisions.length > 0 && (
                        <div className="card">
                            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Decision Log</h3>
                            <div className="space-y-2 max-h-60 overflow-y-auto scrollbar-hide">
                                {decisions.slice(0, 30).map((d) => (
                                    <div key={d.id} className="flex items-start gap-3 text-xs py-1.5 border-b border-dark-border/50 last:border-0">
                                        <span className="font-mono text-gray-600 shrink-0 w-16">
                                            {new Date(d.created_at).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })}
                                        </span>
                                        <span className="badge badge-system shrink-0">{d.agent_role}</span>
                                        <span className="text-gray-500 shrink-0">{d.decision_type}</span>
                                        <span className="text-gray-300 truncate flex-1">{d.decision_outcome}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Live WebSocket Events */}
                    {events.length > 0 && (
                        <div className="card">
                            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Live Agent Events</h3>
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

                    {/* Empty state */}
                    {Object.keys(roles).length === 0 && decisions.length === 0 && events.length === 0 && (
                        <div className="flex-1 flex items-center justify-center py-16">
                            <div className="text-center">
                                <span className="material-symbols-outlined text-4xl text-gray-600 mb-3 block">groups</span>
                                <p className="text-sm text-gray-400">No agent data yet.</p>
                                <p className="text-xs text-gray-600 mt-1">Agent decisions will appear as research progresses.</p>
                            </div>
                        </div>
                    )}
                </main>
            )}

            {/* Footer */}
            <footer className="border-t border-dark-border bg-terminal-black px-6 py-2">
                <div className="max-w-7xl mx-auto flex items-center justify-between text-xs font-mono text-gray-500">
                    <span>Session: {sessionId.slice(0, 8)}</span>
                    <span>{decisions.length} decisions · {events.length} live events</span>
                    <span className="flex items-center gap-2">
                        <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-accent-green" : "bg-gray-600"}`} />
                        WebSocket {connected ? "active" : "inactive"}
                    </span>
                </div>
            </footer>
        </div>
    );
}

/* ── sub-components ─────────────────────────────────────── */
function RoleSection({
    roleName,
    roleData,
    level,
    statusConfig,
    topicCount,
    workerCount,
}: {
    roleName: string;
    roleData: HierarchyData["roles"][string];
    level: string;
    statusConfig: Record<string, { color: string; bg: string; label: string }>;
    topicCount?: number;
    workerCount?: number;
}) {
    const cfg = roleConfig[roleName.toLowerCase()] || defaultRoleCfg;
    const hasActivity = roleData.decision_count > 0;
    const statusCfg = statusConfig[hasActivity ? "active" : "idle"];

    return (
        <section>
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-2">
                <span className="material-symbols-outlined text-sm">{cfg.icon}</span>
                {cfg.label}
            </h2>
            <div className={`card relative overflow-hidden ${level === "manager" ? "border-primary/30" : ""}`}>
                <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
                            <span className="material-symbols-outlined text-primary text-xl">{cfg.icon}</span>
                        </div>
                        <div>
                            <p className="font-semibold">{roleName}</p>
                            <p className="text-xs text-gray-500">{cfg.sublabel}</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className={`status-dot ${statusCfg.bg}`} />
                        <span className={`text-xs font-medium ${statusCfg.color}`}>{statusCfg.label}</span>
                    </div>
                </div>

                <div className="flex items-center gap-2 text-xs text-gray-500 mb-3">
                    <span className="material-symbols-outlined text-[14px]">history</span>
                    {roleData.last_action || "No actions recorded yet"}
                </div>

                <div className="flex items-center gap-4 text-xs text-gray-500">
                    <span className="flex items-center gap-1">
                        <span className="material-symbols-outlined text-sm">analytics</span>
                        {roleData.decision_count} decisions
                    </span>
                    {topicCount !== undefined && (
                        <span className="flex items-center gap-1">
                            <span className="material-symbols-outlined text-sm">topic</span>
                            {topicCount} topics
                        </span>
                    )}
                    {workerCount !== undefined && (
                        <span className="flex items-center gap-1">
                            <span className="material-symbols-outlined text-sm">group</span>
                            {workerCount} workers
                        </span>
                    )}
                </div>

                {/* Recent decisions as reasoning trace */}
                {roleData.recent_decisions.length > 0 && (
                    <div className="mt-4 bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                        <p className="text-gray-500 uppercase tracking-wider font-medium mb-2">Recent Reasoning</p>
                        {roleData.recent_decisions.slice(0, 5).map((d, i) => (
                            <p key={d.id} className="text-gray-400">
                                <span className="text-gray-600 mr-1">{i + 1}.</span>
                                <span className="text-gray-500 font-mono mr-1">[{d.decision_type}]</span>
                                {d.reasoning || d.decision_outcome}
                            </p>
                        ))}
                    </div>
                )}
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
