"use client";

import { useState, useEffect, useRef } from "react";
import { useParams } from "next/navigation";
import ActivityFeed from "@/components/ActivityFeed";
import FindingsBrowser from "@/components/FindingsBrowser";
import ReportPreview from "@/components/ReportPreview";
import SourcesBrowser from "@/components/SourcesBrowser";
import QuestionModal from "@/components/QuestionModal";
import { ResearchWebSocket } from "@/lib/websocket";
import Link from "next/link";

interface Session {
  session_id: string;
  goal: string;
  time_limit: number;
  status: string;
  created_at: string;
  completed_at?: string | null;
}

const tabs = [
  { id: "activity", label: "Live Activity", icon: "stream" },
  { id: "findings", label: "Findings", icon: "description" },
  { id: "report", label: "Report", icon: "article" },
  { id: "graph", label: "Knowledge Graph", icon: "hub" },
  { id: "sources", label: "Sources", icon: "travel_explore" },
  { id: "agents", label: "Agents", icon: "psychology" },
  { id: "verify", label: "Verification", icon: "verified" },
] as const;

type TabId = (typeof tabs)[number]["id"];

export default function SessionDetail() {
  const params = useParams();
  const sessionId = params.id as string;

  const [session, setSession] = useState<Session | null>(null);
  const [stats, setStats] = useState<{ findings: number; sources: number; topics: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState<TabId>("activity");
  const [nowTick, setNowTick] = useState(() => Date.now());
  const [pendingQuestion, setPendingQuestion] = useState<{
    questionId: string;
    question: string;
    context: string;
    options: string[];
    timeout: number;
  } | null>(null);
  const wsRef = useRef<ResearchWebSocket | null>(null);

  useEffect(() => {
    fetchSession();
    fetchStats();
  }, [sessionId]);

  useEffect(() => {
    const timer = setInterval(() => setNowTick(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!session || session.status === "completed" || session.status === "error") return;
    const interval = setInterval(() => {
      fetchSession();
      fetchStats();
    }, 10000);
    return () => clearInterval(interval);
  }, [session?.status]);

  useEffect(() => {
    // Set up WebSocket listener for mid-research questions
    const ws = new ResearchWebSocket(sessionId);

    ws.onEvent((event) => {
      if (event.event_type === "question_asked" && event.data) {
        setPendingQuestion({
          questionId: event.data.question_id,
          question: event.data.question,
          context: event.data.context || "",
          options: event.data.options || [],
          timeout: event.data.timeout || 60,
        });
      } else if (event.event_type === "question_answered" || event.event_type === "question_timeout") {
        setPendingQuestion(null);
      }
    });

    ws.connect();
    wsRef.current = ws;

    return () => {
      ws.disconnect();
    };
  }, [sessionId]);

  const fetchSession = async () => {
    try {
      const response = await fetch(`/api/sessions/${sessionId}`);
      if (!response.ok) {
        if (response.status === 404) setError("Session not found");
        else throw new Error("Failed to fetch session");
        return;
      }
      const data = await response.json();
      setSession(data);
    } catch (err) {
      console.error("Error fetching session:", err);
      setError("Failed to load session");
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`/api/sessions/${sessionId}/stats`);
      if (response.ok) setStats(await response.json());
    } catch {
      // Non-critical
    }
  };

  const handleQuestionSubmit = (response: string) => {
    console.log("Question answered:", response);
    setPendingQuestion(null);
  };

  const handleQuestionTimeout = () => {
    console.log("Question timed out");
    setPendingQuestion(null);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <span className="material-symbols-outlined text-4xl text-primary animate-spin mb-4 block">progress_activity</span>
          <p className="text-gray-400">Loading session...</p>
        </div>
      </div>
    );
  }

  if (error || !session) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <span className="material-symbols-outlined text-5xl text-error mb-4 block">error</span>
          <h2 className="text-2xl font-bold mb-2">{error || "Session not found"}</h2>
          <Link href="/" className="text-primary hover:underline text-sm">← Back to Dashboard</Link>
        </div>
      </div>
    );
  }

  const isRunning = session.status === "running" || session.status === "pending";
  const elapsed = getElapsedTime(session.created_at, nowTick);
  const remaining = getRemainingTime(session.created_at, session.time_limit, nowTick);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-dark-border bg-dark-surface/50 backdrop-blur-sm sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          {/* Breadcrumb */}
          <div className="flex items-center gap-2 text-xs text-gray-500 mb-3">
            <Link href="/" className="hover:text-primary transition-colors">Sessions</Link>
            <span className="material-symbols-outlined text-[12px]">chevron_right</span>
            <span className={isRunning ? "text-accent-success" : "text-gray-400"}>
              {isRunning ? "Running" : session.status}
            </span>
          </div>

          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-4 min-w-0">
              <Link href="/" className="text-gray-400 hover:text-primary transition-colors mt-1 shrink-0">
                <span className="material-symbols-outlined">arrow_back</span>
              </Link>
              <div className="min-w-0">
                <h1 className="text-xl font-bold line-clamp-2 leading-tight">{session.goal}</h1>
                <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                  <span className="font-mono">#{session.session_id.slice(0, 8)}</span>
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-[14px]">schedule</span>
                    {formatDate(session.created_at)}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <span className={`badge ${getStatusBadgeClass(session.status)}`}>
                {isRunning && (
                  <span className="relative flex h-1.5 w-1.5 mr-1">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75" />
                    <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-current" />
                  </span>
                )}
                {session.status}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Stats Cards */}
      <div className="border-b border-dark-border bg-dark-surface/20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard
              icon="travel_explore"
              label="Sources Found"
              value={stats ? String(stats.sources) : "--"}
              color="text-accent-blue"
            />
            <StatCard
              icon="description"
              label="Findings"
              value={stats ? String(stats.findings) : "--"}
              color="text-accent-green"
            />
            <StatCard
              icon="topic"
              label="Topics"
              value={stats ? String(stats.topics) : "--"}
              color="text-accent-yellow"
            />
            <StatCard
              icon="timer"
              label={isRunning ? "Time Remaining" : "Duration"}
              value={isRunning ? remaining : elapsed}
              color={isRunning ? "text-primary" : "text-gray-400"}
              animate={isRunning}
            />
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-dark-border bg-dark-surface/10 sticky top-[89px] z-10 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6">
          <div className="tab-bar overflow-x-auto scrollbar-hide">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`tab-item whitespace-nowrap ${activeTab === tab.id ? "tab-item-active" : ""}`}
              >
                <span className="material-symbols-outlined text-lg">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Tab Content */}
      <main className="max-w-7xl mx-auto w-full px-6 py-8 flex-1">
        {activeTab === "activity" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <ActivityFeed sessionId={sessionId} />
            </div>
            <div className="space-y-6">
              {/* Agent Status */}
              <div className="card">
                <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Agent Status</h3>
                <div className="space-y-3">
                  <AgentStatusCard name="Director" role="Level 0" status="active" icon="military_tech" />
                  <AgentStatusCard name="Manager" role="Level 1" status="thinking" icon="psychology" />
                  <AgentStatusCard name="Intern Pool" role="Level 2" status="active" icon="group" count={3} />
                </div>
              </div>

              {/* System Logs */}
              <div className="card bg-terminal-black border-terminal-border">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">System Logs</h3>
                  <span className="status-dot status-dot-active" />
                </div>
                <div className="space-y-1.5 font-mono text-xs text-gray-400 max-h-48 overflow-y-auto scrollbar-hide">
                  <p><span className="text-primary">[{new Date().toLocaleTimeString()}]</span> Monitoring research session...</p>
                  <p><span className="text-primary">[{new Date().toLocaleTimeString()}]</span> WebSocket connection established</p>
                  <p className="text-white bg-white/5 p-1 -mx-1 rounded">
                    <span className="text-primary animate-pulse">›</span> Awaiting agent events...
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === "findings" && <FindingsBrowser sessionId={sessionId} />}
        {activeTab === "report" && <ReportPreview sessionId={sessionId} />}

        {activeTab === "graph" && (
          <SubPagePlaceholder
            icon="hub"
            title="Knowledge Graph"
            description="Interactive visualization of entities, relationships, and the knowledge network built during research."
            linkHref={`/session/${sessionId}/graph`}
            linkText="Open Full Graph View"
          />
        )}

        {activeTab === "sources" && <SourcesBrowser sessionId={sessionId} />}

        {activeTab === "agents" && (
          <SubPagePlaceholder
            icon="psychology"
            title="Agent Transparency"
            description="See the hierarchical agent structure, their current goals, reasoning traces, and real-time decision-making process."
            linkHref={`/session/${sessionId}/agents`}
            linkText="Open Agent View"
          />
        )}

        {activeTab === "verify" && (
          <SubPagePlaceholder
            icon="verified"
            title="CoVe Verification Pipeline"
            description="Track how findings are verified through the Chain-of-Verification pipeline with confidence scoring."
            linkHref={`/session/${sessionId}/verify`}
            linkText="Open Verification Pipeline"
          />
        )}
      </main>

      {/* Mid-Research Question Modal */}
      {pendingQuestion && (
        <QuestionModal
          sessionId={sessionId}
          questionId={pendingQuestion.questionId}
          question={pendingQuestion.question}
          context={pendingQuestion.context}
          options={pendingQuestion.options}
          timeout={pendingQuestion.timeout}
          onSubmit={handleQuestionSubmit}
          onTimeout={handleQuestionTimeout}
        />
      )}
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  color,
  animate,
}: {
  icon: string;
  label: string;
  value: string;
  color: string;
  animate?: boolean;
}) {
  return (
    <div className="bg-dark-surface border border-dark-border rounded-xl p-4 hover:border-primary/30 transition-colors group">
      <div className="flex justify-between items-start mb-2">
        <p className="text-xs font-medium text-gray-500">{label}</p>
        <span className={`material-symbols-outlined text-lg transition-colors group-hover:text-primary ${color}`}>
          {icon}
        </span>
      </div>
      <p className={`font-mono text-2xl font-bold tracking-tighter ${animate ? "animate-pulse" : ""}`}>{value}</p>
    </div>
  );
}

function AgentStatusCard({
  name,
  role,
  status,
  icon,
  count,
}: {
  name: string;
  role: string;
  status: string;
  icon: string;
  count?: number;
}) {
  const statusConfig: Record<string, { dot: string; label: string }> = {
    active: { dot: "status-dot-active", label: "Active" },
    thinking: { dot: "bg-primary animate-pulse", label: "Thinking" },
    idle: { dot: "status-dot-idle", label: "Idle" },
  };
  const cfg = statusConfig[status] || statusConfig.idle;

  return (
    <div className="flex items-center justify-between p-3 bg-dark-bg rounded-lg border border-dark-border hover:border-primary/20 transition-colors">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
          <span className="material-symbols-outlined text-primary text-base">{icon}</span>
        </div>
        <div>
          <span className="text-sm font-medium block">{name}</span>
          <span className="text-xs text-gray-500">{role}{count ? ` (${count})` : ""}</span>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span className={`status-dot ${cfg.dot}`} />
        <span className="text-xs text-gray-400">{cfg.label}</span>
      </div>
    </div>
  );
}

function SubPagePlaceholder({
  icon,
  title,
  description,
  linkHref,
  linkText,
}: {
  icon: string;
  title: string;
  description: string;
  linkHref: string;
  linkText: string;
}) {
  return (
    <div className="card flex flex-col items-center justify-center py-16 text-center">
      <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
        <span className="material-symbols-outlined text-primary text-3xl">{icon}</span>
      </div>
      <h2 className="text-xl font-bold mb-2">{title}</h2>
      <p className="text-sm text-gray-400 max-w-md mb-6">{description}</p>
      <Link href={linkHref} className="btn btn-primary">
        <span className="material-symbols-outlined text-lg">open_in_new</span>
        {linkText}
      </Link>
    </div>
  );
}

function getStatusBadgeClass(status: string): string {
  switch (status) {
    case "active":
    case "running":
      return "badge-action";
    case "completed":
      return "badge-success";
    case "error":
      return "badge-error";
    default:
      return "badge-system";
  }
}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getElapsedTime(startDate: string, nowMs: number): string {
  const start = new Date(startDate);
  const diffMs = nowMs - start.getTime();
  const diffSecs = Math.max(0, Math.floor(diffMs / 1000));
  const mins = Math.floor(diffSecs / 60);
  const hrs = Math.floor(mins / 60);
  const secs = diffSecs % 60;

  if (hrs > 0) return `${hrs}h ${mins % 60}m`;
  if (mins > 0) return `${mins}m ${secs}s`;
  return `${secs}s`;
}

function getRemainingTime(startDate: string, limitMins: number, nowMs: number): string {
  const start = new Date(startDate);
  const endMs = start.getTime() + limitMins * 60 * 1000;
  const remainMs = endMs - nowMs;
  if (remainMs <= 0) return "0:00";
  const remainSecs = Math.floor(remainMs / 1000);
  const mins = Math.floor(remainSecs / 60);
  const secs = remainSecs % 60;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}
