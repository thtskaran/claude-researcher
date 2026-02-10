"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

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

/* ── role config ───────────────────────────────────────── */
const roleConfig: Record<string, { icon: string; color: string; bgColor: string; label: string; description: string }> = {
    director: {
        icon: "military_tech",
        color: "text-iris",
        bgColor: "bg-iris/10",
        label: "Director",
        description: "Strategic planning and goal decomposition"
    },
    manager: {
        icon: "psychology",
        color: "text-sage",
        bgColor: "bg-sage/10",
        label: "Manager",
        description: "Tactical coordination and synthesis"
    },
    intern: {
        icon: "person_search",
        color: "text-olive",
        bgColor: "bg-olive/10",
        label: "Research Intern",
        description: "Web search and finding extraction"
    },
    researcher: {
        icon: "person_search",
        color: "text-olive",
        bgColor: "bg-olive/10",
        label: "Research Intern",
        description: "Web search and finding extraction"
    },
    research_intern: {
        icon: "person_search",
        color: "text-olive",
        bgColor: "bg-olive/10",
        label: "Research Intern",
        description: "Web search and finding extraction"
    },
};
const defaultRoleCfg = {
    icon: "smart_toy",
    color: "text-ink-secondary",
    bgColor: "bg-ink-secondary/10",
    label: "Agent",
    description: "System agent"
};

export default function AgentTransparencyPage() {
    const params = useParams();
    const sessionId = params.id as string;
    const [decisions, setDecisions] = useState<AgentDecision[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

    const fetchData = useCallback(async () => {
        setLoading(true);
        try {
            const response = await fetch(`/api/sessions/${sessionId}/agents/decisions?limit=500`);
            if (response.ok) {
                const data = await response.json();
                setDecisions(data);
            }
        } catch (err) {
            console.error("Failed to fetch agent data:", err);
        } finally {
            setLoading(false);
        }
    }, [sessionId]);

    useEffect(() => { fetchData(); }, [fetchData]);

    // Group decisions by agent role
    const agentGroups: Record<string, AgentDecision[]> = {};
    decisions.forEach((d) => {
        const role = d.agent_role;
        if (!agentGroups[role]) agentGroups[role] = [];
        agentGroups[role].push(d);
    });

    // Get agent roles in hierarchy order
    const orderedAgents = Object.keys(agentGroups).sort((a, b) => {
        const levels: Record<string, number> = { director: 0, manager: 1, intern: 2, researcher: 2, research_intern: 2 };
        return (levels[a.toLowerCase()] || 3) - (levels[b.toLowerCase()] || 3);
    });

    // Stats
    const totalActions = decisions.length;
    const uniqueActionTypes = new Set(decisions.map(d => d.decision_type)).size;

    // Get timeline for selected agent or all
    const filteredDecisions = selectedAgent
        ? decisions.filter(d => d.agent_role === selectedAgent)
        : decisions;

    return (
        <div className="min-h-screen flex flex-col">
            {/* Header */}
            <header className="border-b border-edge bg-card/50 backdrop-blur-sm sticky top-0 z-20">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center gap-2 text-xs text-ink-muted mb-2">
                        <Link href="/" className="hover:text-sage transition-colors">Sessions</Link>
                        <span className="material-symbols-outlined text-[12px]">chevron_right</span>
                        <Link href={`/session/${sessionId}`} className="hover:text-sage transition-colors">Session</Link>
                        <span className="material-symbols-outlined text-[12px]">chevron_right</span>
                        <span className="text-ink-secondary">Agent Activity</span>
                    </div>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <Link href={`/session/${sessionId}`} className="text-ink-secondary hover:text-sage transition-colors">
                                <span className="material-symbols-outlined">arrow_back</span>
                            </Link>
                            <div>
                                <h1 className="text-xl font-display">Agent Transparency</h1>
                                <p className="text-xs text-ink-muted">See what each agent did and why</p>
                            </div>
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
                    <div className="flex items-center gap-3 text-ink-secondary">
                        <span className="material-symbols-outlined animate-spin">progress_activity</span>
                        Loading agent activity...
                    </div>
                </div>
            ) : decisions.length === 0 ? (
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <span className="material-symbols-outlined text-5xl text-ink-muted mb-3 block">groups</span>
                        <p className="text-ink-secondary">No agent activity yet</p>
                        <p className="text-xs text-ink-muted mt-1">Actions will appear as research progresses</p>
                    </div>
                </div>
            ) : (
                <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8">
                    {/* Overview Stats */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                        <div className="card bg-gradient-to-br from-sage/5 to-transparent border-sage/20">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-xs text-ink-secondary uppercase tracking-wider">Total Actions</span>
                                <span className="material-symbols-outlined text-sage">analytics</span>
                            </div>
                            <p className="text-3xl font-bold font-mono text-ink">{totalActions}</p>
                            <p className="text-xs text-ink-muted mt-1">Actions taken by all agents</p>
                        </div>

                        <div className="card bg-gradient-to-br from-olive/5 to-transparent border-olive/20">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-xs text-ink-secondary uppercase tracking-wider">Active Agents</span>
                                <span className="material-symbols-outlined text-olive">groups</span>
                            </div>
                            <p className="text-3xl font-bold font-mono text-ink">{orderedAgents.length}</p>
                            <p className="text-xs text-ink-muted mt-1">Agents participating in research</p>
                        </div>

                        <div className="card bg-gradient-to-br from-gold/5 to-transparent border-gold/20">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-xs text-ink-secondary uppercase tracking-wider">Action Types</span>
                                <span className="material-symbols-outlined text-gold">category</span>
                            </div>
                            <p className="text-3xl font-bold font-mono text-ink">{uniqueActionTypes}</p>
                            <p className="text-xs text-ink-muted mt-1">Different types of actions performed</p>
                        </div>
                    </div>

                    {/* Agent Cards */}
                    <div className="mb-8">
                        <h2 className="text-sm font-display font-normal text-ink-secondary uppercase tracking-wider mb-4">
                            Research Team
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {orderedAgents.map((agentRole) => {
                                const cfg = roleConfig[agentRole.toLowerCase()] || defaultRoleCfg;
                                const agentDecisions = agentGroups[agentRole];
                                const isSelected = selectedAgent === agentRole;

                                return (
                                    <button
                                        key={agentRole}
                                        onClick={() => setSelectedAgent(isSelected ? null : agentRole)}
                                        className={`card text-left transition-all hover:shadow-lg ${
                                            isSelected ? `ring-2 ${cfg.color.replace("text-", "ring-")} ${cfg.bgColor}` : ""
                                        }`}
                                    >
                                        <div className="flex items-start gap-3 mb-3">
                                            <div className={`w-12 h-12 rounded-xl ${cfg.bgColor} flex items-center justify-center shrink-0`}>
                                                <span className={`material-symbols-outlined text-xl ${cfg.color}`}>
                                                    {cfg.icon}
                                                </span>
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <h3 className="font-semibold text-ink">{cfg.label}</h3>
                                                <p className="text-xs text-ink-muted leading-relaxed">{cfg.description}</p>
                                            </div>
                                        </div>

                                        <div className="flex items-center justify-between text-xs">
                                            <div className="flex items-center gap-4">
                                                <div>
                                                    <span className="text-ink-muted">Actions:</span>
                                                    <span className="ml-1 font-mono font-bold text-ink">{agentDecisions.length}</span>
                                                </div>
                                                {agentDecisions[0] && (
                                                    <div>
                                                        <span className="text-ink-muted">Last:</span>
                                                        <span className="ml-1 font-mono text-ink-secondary">
                                                            {new Date(agentDecisions[0].created_at).toLocaleTimeString("en-US", {
                                                                hour: "2-digit",
                                                                minute: "2-digit"
                                                            })}
                                                        </span>
                                                    </div>
                                                )}
                                            </div>
                                            {isSelected && (
                                                <span className={`badge ${cfg.color.replace("text-", "badge-")}`}>Selected</span>
                                            )}
                                        </div>
                                    </button>
                                );
                            })}
                        </div>
                        {selectedAgent && (
                            <p className="text-xs text-ink-muted mt-3 text-center">
                                Showing only actions from <span className="text-ink font-semibold">{selectedAgent}</span>
                                <button onClick={() => setSelectedAgent(null)} className="ml-2 text-sage hover:underline">
                                    Clear filter
                                </button>
                            </p>
                        )}
                    </div>

                    {/* Activity Timeline */}
                    <div>
                        <h2 className="text-sm font-display font-normal text-ink-secondary uppercase tracking-wider mb-4">
                            Activity Timeline ({filteredDecisions.length} actions)
                        </h2>
                        <div className="card p-0 overflow-hidden">
                            <div className="max-h-[600px] overflow-y-auto">
                                <div className="divide-y divide-edge">
                                    {filteredDecisions.map((decision, idx) => (
                                        <ActionCard key={decision.id} decision={decision} />
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </main>
            )}
        </div>
    );
}

/* ── Action Card ────────────────────────────────────────── */
function ActionCard({ decision }: { decision: AgentDecision }) {
    const [expanded, setExpanded] = useState(false);
    const cfg = roleConfig[decision.agent_role.toLowerCase()] || defaultRoleCfg;

    // Parse action into human-readable text
    const action = getActionDescription(decision);
    const hasDetails = decision.reasoning || (decision.inputs && Object.keys(decision.inputs).length > 0) || (decision.metrics && Object.keys(decision.metrics).length > 0);

    return (
        <div className="p-4 hover:bg-card/50 transition-colors">
            {/* Header */}
            <div className="flex items-start gap-3">
                <div className={`w-10 h-10 rounded-lg ${cfg.bgColor} flex items-center justify-center shrink-0`}>
                    <span className={`material-symbols-outlined text-lg ${cfg.color}`}>{cfg.icon}</span>
                </div>

                <div className="flex-1 min-w-0">
                    {/* Agent & Time */}
                    <div className="flex items-center gap-2 mb-1.5">
                        <span className={`text-sm font-semibold ${cfg.color}`}>{cfg.label}</span>
                        <span className="text-ink-muted">&middot;</span>
                        <span className="text-xs text-ink-muted font-mono">
                            {new Date(decision.created_at).toLocaleString("en-US", {
                                month: "short",
                                day: "numeric",
                                hour: "2-digit",
                                minute: "2-digit",
                                second: "2-digit"
                            })}
                        </span>
                        {decision.iteration !== null && (
                            <>
                                <span className="text-ink-muted">&middot;</span>
                                <span className="text-xs text-ink-muted">
                                    <span className="material-symbols-outlined text-xs align-middle mr-0.5">replay</span>
                                    Iteration {decision.iteration}
                                </span>
                            </>
                        )}
                    </div>

                    {/* Action Description */}
                    <p className="text-sm text-ink leading-relaxed mb-2">{action}</p>

                    {/* Quick metrics */}
                    {decision.metrics && Object.keys(decision.metrics).length > 0 && (
                        <div className="flex flex-wrap gap-2 mb-2">
                            {Object.entries(decision.metrics).slice(0, 3).map(([key, value]) => (
                                <span key={key} className="inline-flex items-center gap-1 px-2 py-0.5 bg-card-inset rounded text-xs">
                                    <span className="text-ink-muted">{key}:</span>
                                    <span className="text-ink-secondary font-mono">{String(value)}</span>
                                </span>
                            ))}
                        </div>
                    )}

                    {/* Expand button */}
                    {hasDetails && (
                        <button
                            onClick={() => setExpanded(!expanded)}
                            className="text-xs text-ink-muted hover:text-sage transition-colors flex items-center gap-1 mt-2"
                        >
                            <span className="material-symbols-outlined text-sm">
                                {expanded ? "expand_less" : "expand_more"}
                            </span>
                            {expanded ? "Hide details" : "Show details"}
                        </button>
                    )}

                    {/* Expanded Details */}
                    {expanded && hasDetails && (
                        <div className="mt-3 space-y-3 pl-4 border-l-2 border-edge">
                            {/* Reasoning */}
                            {decision.reasoning && (
                                <div>
                                    <p className="text-xs text-ink-muted uppercase tracking-wider mb-1.5 flex items-center gap-1">
                                        <span className="material-symbols-outlined text-sm">psychology</span>
                                        Why this action was taken
                                    </p>
                                    <p className="text-sm text-ink-secondary leading-relaxed italic bg-card/30 rounded p-3">
                                        &ldquo;{decision.reasoning}&rdquo;
                                    </p>
                                </div>
                            )}

                            {/* Inputs */}
                            {decision.inputs && Object.keys(decision.inputs).length > 0 && (
                                <div>
                                    <p className="text-xs text-ink-muted uppercase tracking-wider mb-1.5 flex items-center gap-1">
                                        <span className="material-symbols-outlined text-sm">input</span>
                                        Input data used
                                    </p>
                                    <div className="bg-card-inset border border-edge rounded p-3 overflow-x-auto">
                                        <pre className="font-mono text-xs text-ink-secondary">
                                            {JSON.stringify(decision.inputs, null, 2)}
                                        </pre>
                                    </div>
                                </div>
                            )}

                            {/* All Metrics */}
                            {decision.metrics && Object.keys(decision.metrics).length > 3 && (
                                <div>
                                    <p className="text-xs text-ink-muted uppercase tracking-wider mb-1.5 flex items-center gap-1">
                                        <span className="material-symbols-outlined text-sm">monitoring</span>
                                        Performance metrics
                                    </p>
                                    <div className="grid grid-cols-2 gap-2">
                                        {Object.entries(decision.metrics).map(([key, value]) => (
                                            <div key={key} className="bg-card/50 rounded px-3 py-2">
                                                <p className="text-xs text-ink-muted">{key}</p>
                                                <p className="text-sm text-ink font-mono">{String(value)}</p>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

/* ── Helpers ────────────────────────────────────────────── */
function getActionDescription(decision: AgentDecision): string {
    const { decision_type, decision_outcome } = decision;

    // Map to human-readable descriptions
    const actionMap: Record<string, (outcome: string) => string> = {
        decompose_goal: (o) => `Decomposed research goal → ${o}`,
        assign_topic: (o) => `Assigned new research topic: "${o}"`,
        critique_findings: (o) => `Critiqued findings from research → ${o}`,
        web_search: (o) => `Searched the web for: "${o}"`,
        extract_finding: (o) => `Found: ${o}`,
        verify_finding: (o) => `Verified finding → ${o}`,
        synthesis: (o) => `Synthesized results → ${o}`,
        directive_complete: (o) => `Completed directive → ${o}`,
        topic_complete: (o) => `Finished researching: "${o}"`,
        query_expansion: (o) => `Expanded search query → "${o}"`,
        fallback_search: (o) => `Tried alternative search → "${o}"`,
        deep_report_section: (o) => `Wrote report section → ${o}`,
        knowledge_graph_update: (o) => `Updated knowledge graph → ${o}`,
        contradiction_detected: (o) => `Detected contradiction → ${o}`,
        retrieve_context: (o) => `Retrieved context for research → ${o}`,
        route_tool: (o) => `Selected tool → ${o}`,
        plan: (o) => `Planned next steps → ${o}`,
    };

    const descFn = actionMap[decision_type.toLowerCase()];
    if (descFn) return descFn(decision_outcome);

    // Fallback: humanize the decision_type
    const humanType = decision_type
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');

    return `${humanType} → ${decision_outcome}`;
}
