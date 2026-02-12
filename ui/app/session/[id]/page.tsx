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
  elapsed_seconds?: number;
  paused_at?: string | null;
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
    if (!session) return;
    // Keep polling for running, paused (might be transitioning), and crashed sessions
    const terminalStatuses = ["completed", "error", "interrupted"];
    if (terminalStatuses.includes(session.status)) return;
    const interval = setInterval(() => {
      fetchSession();
      fetchStats();
    }, 5000);
    return () => clearInterval(interval);
  }, [session?.status]);

  useEffect(() => {
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

  const [actionPending, setActionPending] = useState(false);

  const handlePause = async () => {
    setActionPending(true);
    try {
      const res = await fetch(`/api/research/${sessionId}/pause`, { method: "POST" });
      if (res.ok) {
        // Poll until status changes to paused
        const poll = setInterval(async () => {
          await fetchSession();
        }, 2000);
        setTimeout(() => clearInterval(poll), 30000);
      }
    } catch (err) {
      console.error("Failed to pause:", err);
    } finally {
      setActionPending(false);
    }
  };

  const handleResume = async () => {
    setActionPending(true);
    try {
      const res = await fetch(`/api/research/${sessionId}/resume`, { method: "POST" });
      if (res.ok) {
        // Start polling for updates
        await fetchSession();
      }
    } catch (err) {
      console.error("Failed to resume:", err);
    } finally {
      setActionPending(false);
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
          <span className="material-symbols-outlined text-4xl text-sage animate-spin mb-4 block">progress_activity</span>
          <p className="text-ink-secondary">Loading session...</p>
        </div>
      </div>
    );
  }

  if (error || !session) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <span className="material-symbols-outlined text-5xl text-coral mb-4 block">error</span>
          <h2 className="text-2xl font-display mb-2">{error || "Session not found"}</h2>
          <Link href="/" className="text-sage hover:underline text-sm">&larr; Back to Dashboard</Link>
        </div>
      </div>
    );
  }

  const isRunning = session.status === "running" || session.status === "pending";
  const isPaused = session.status === "paused";
  const isCrashed = session.status === "crashed";
  const isResumable = isPaused || isCrashed;

  const elapsed = isRunning
    ? getElapsedTime(session.created_at, nowTick)
    : session.elapsed_seconds && session.elapsed_seconds > 0
      ? formatSeconds(session.elapsed_seconds)
      : getDuration(session.created_at, session.completed_at || null);
  const remaining = isRunning
    ? getRemainingTimeWithElapsed(session.time_limit, session.elapsed_seconds || 0, session.created_at, nowTick)
    : isResumable && session.elapsed_seconds
      ? formatSeconds(session.time_limit * 60 - session.elapsed_seconds)
      : getRemainingTime(session.created_at, session.time_limit, nowTick);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-edge bg-card/50 backdrop-blur-sm sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center gap-2 text-xs text-ink-muted mb-3">
            <Link href="/" className="hover:text-sage transition-colors">Sessions</Link>
            <span className="material-symbols-outlined text-[12px]">chevron_right</span>
            <span className={isRunning ? "text-olive" : "text-ink-secondary"}>
              {isRunning ? "Running" : session.status}
            </span>
          </div>

          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-4 min-w-0">
              <Link href="/" className="text-ink-secondary hover:text-sage transition-colors mt-1 shrink-0">
                <span className="material-symbols-outlined">arrow_back</span>
              </Link>
              <div className="min-w-0">
                <h1 className="text-xl font-display line-clamp-2 leading-tight">{session.goal}</h1>
                <div className="flex items-center gap-4 mt-2 text-xs text-ink-muted">
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
                    <span className="animate-soft-pulse absolute inline-flex h-full w-full rounded-full bg-current opacity-75" />
                    <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-current" />
                  </span>
                )}
                {session.status}
              </span>
              {isRunning && (
                <button onClick={handlePause} disabled={actionPending} className="btn btn-secondary text-xs py-1 px-3">
                  <span className="material-symbols-outlined text-sm">pause</span>
                  Pause
                </button>
              )}
              {isResumable && (
                <button onClick={handleResume} disabled={actionPending} className="btn btn-primary text-xs py-1 px-3">
                  <span className="material-symbols-outlined text-sm">play_arrow</span>
                  Resume
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Crash Banner */}
      {isCrashed && (
        <div className="border-b border-coral/30 bg-coral-soft">
          <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="material-symbols-outlined text-coral">warning</span>
              <div>
                <p className="text-sm font-medium text-coral">This research session crashed unexpectedly</p>
                <p className="text-xs text-ink-secondary">Progress was saved. You can resume from the last checkpoint.</p>
              </div>
            </div>
            <button onClick={handleResume} disabled={actionPending} className="btn btn-primary text-xs py-1 px-3">
              <span className="material-symbols-outlined text-sm">play_arrow</span>
              Resume Research
            </button>
          </div>
        </div>
      )}

      {/* Paused Banner */}
      {isPaused && (
        <div className="border-b border-gold/30 bg-gold-soft">
          <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="material-symbols-outlined text-gold">pause_circle</span>
              <div>
                <p className="text-sm font-medium text-gold">Research is paused</p>
                <p className="text-xs text-ink-secondary">All progress has been saved. Resume when ready.</p>
              </div>
            </div>
            <button onClick={handleResume} disabled={actionPending} className="btn btn-primary text-xs py-1 px-3">
              <span className="material-symbols-outlined text-sm">play_arrow</span>
              Resume
            </button>
          </div>
        </div>
      )}

      {/* Stats Cards */}
      <div className="border-b border-edge bg-card/20">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard icon="travel_explore" label="Sources Found" value={stats ? String(stats.sources) : "--"} color="text-sage" />
            <StatCard icon="description" label="Findings" value={stats ? String(stats.findings) : "--"} color="text-olive" />
            <StatCard icon="topic" label="Topics" value={stats ? String(stats.topics) : "--"} color="text-gold" />
            <StatCard icon="timer" label={isRunning ? "Time Remaining" : "Duration"} value={isRunning ? remaining : elapsed} color={isRunning ? "text-sage" : "text-ink-secondary"} animate={isRunning} />
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-edge bg-card/10 sticky top-[89px] z-10 backdrop-blur-sm">
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
              <div className="card">
                <h3 className="text-sm font-semibold text-ink-secondary uppercase tracking-wider mb-4">Agent Status</h3>
                <div className="space-y-3">
                  <AgentStatusCard name="Director" role="Level 0" status="active" icon="military_tech" variant="iris" />
                  <AgentStatusCard name="Manager" role="Level 1" status="thinking" icon="psychology" variant="sage" />
                  <AgentStatusCard name="Intern Pool" role="Level 2" status="active" icon="group" count={3} variant="olive" />
                </div>
              </div>

              <div className="card bg-card-inset border-edge">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold text-ink-secondary uppercase tracking-wider">System Logs</h3>
                  <span className="status-dot status-dot-active" />
                </div>
                <div className="space-y-1.5 font-mono text-xs text-ink-secondary max-h-48 overflow-y-auto scrollbar-hide">
                  <p><span className="text-sage">[{new Date().toLocaleTimeString()}]</span> Monitoring research session...</p>
                  <p><span className="text-sage">[{new Date().toLocaleTimeString()}]</span> WebSocket connection established</p>
                  <p className="text-ink bg-ink/5 p-1 -mx-1 rounded">
                    <span className="text-sage animate-soft-pulse">&rsaquo;</span> Awaiting agent events...
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
          <div className="w-full h-[calc(100vh-320px)] min-h-[600px]">
            <iframe
              src={`/session/${sessionId}/agents`}
              className="w-full h-full border-0 rounded-xl bg-card"
              title="Agent Transparency"
            />
          </div>
        )}

        {activeTab === "verify" && (
          <div className="w-full h-[calc(100vh-320px)] min-h-[600px]">
            <iframe
              src={`/session/${sessionId}/verify`}
              className="w-full h-full border-0 rounded-xl bg-card"
              title="CoVe Verification Pipeline"
            />
          </div>
        )}
      </main>

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

function StatCard({ icon, label, value, color, animate }: { icon: string; label: string; value: string; color: string; animate?: boolean }) {
  return (
    <div className="bg-card border border-edge rounded-xl p-4 hover:border-sage/30 transition-colors group">
      <div className="flex justify-between items-start mb-2">
        <p className="text-xs font-medium text-ink-muted">{label}</p>
        <span className={`material-symbols-outlined text-lg transition-colors group-hover:text-sage ${color}`}>{icon}</span>
      </div>
      <p className={`font-mono text-2xl font-bold tracking-tighter ${animate ? "animate-soft-pulse" : ""}`}>{value}</p>
    </div>
  );
}

function AgentStatusCard({ name, role, status, icon, count, variant }: { name: string; role: string; status: string; icon: string; count?: number; variant?: "iris" | "sage" | "olive" }) {
  const statusConfig: Record<string, { dot: string; label: string }> = {
    active: { dot: "status-dot-active", label: "Active" },
    thinking: { dot: "bg-iris animate-soft-pulse", label: "Thinking" },
    idle: { dot: "status-dot-idle", label: "Idle" },
  };
  const cfg = statusConfig[status] || statusConfig.idle;

  const variantConfig = {
    iris: { bg: "bg-iris-soft", text: "text-iris" },
    sage: { bg: "bg-sage-soft", text: "text-sage" },
    olive: { bg: "bg-olive-soft", text: "text-olive" },
  };
  const vc = variantConfig[variant || "sage"];

  return (
    <div className="flex items-center justify-between p-3 bg-card-inset rounded-lg border border-edge hover:border-sage/20 transition-colors">
      <div className="flex items-center gap-3">
        <div className={`w-8 h-8 rounded-lg ${vc.bg} flex items-center justify-center`}>
          <span className={`material-symbols-outlined ${vc.text} text-base`}>{icon}</span>
        </div>
        <div>
          <span className="text-sm font-medium block">{name}</span>
          <span className="text-xs text-ink-muted">{role}{count ? ` (${count})` : ""}</span>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span className={`status-dot ${cfg.dot}`} />
        <span className="text-xs text-ink-secondary">{cfg.label}</span>
      </div>
    </div>
  );
}

function SubPagePlaceholder({ icon, title, description, linkHref, linkText }: { icon: string; title: string; description: string; linkHref: string; linkText: string }) {
  return (
    <div className="card flex flex-col items-center justify-center py-16 text-center">
      <div className="w-16 h-16 rounded-2xl bg-sage-soft flex items-center justify-center mb-4">
        <span className="material-symbols-outlined text-sage text-3xl">{icon}</span>
      </div>
      <h2 className="text-xl font-display mb-2">{title}</h2>
      <p className="text-sm text-ink-secondary max-w-md mb-6">{description}</p>
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
    case "crashed":
      return "badge-error";
    case "paused":
      return "badge-warning";
    default:
      return "badge-system";
  }
}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
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

function getDuration(startDateString: string, endDateString: string | null): string {
  if (!endDateString) {
    return "N/A";
  }

  const start = new Date(startDateString);
  const end = new Date(endDateString);
  const diffMs = end.getTime() - start.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);

  if (diffSecs < 60) return `${diffSecs}s`;
  if (diffMins < 60) return `${diffMins}m ${diffSecs % 60}s`;
  return `${diffHours}h ${diffMins % 60}m`;
}

function formatSeconds(totalSecs: number): string {
  const secs = Math.max(0, Math.floor(totalSecs));
  const mins = Math.floor(secs / 60);
  const hrs = Math.floor(mins / 60);

  if (hrs > 0) return `${hrs}h ${mins % 60}m`;
  if (mins > 0) return `${mins}m ${secs % 60}s`;
  return `${secs}s`;
}

function getRemainingTimeWithElapsed(limitMins: number, elapsedSecs: number, startDate: string, nowMs: number): string {
  // For a running session that was previously resumed, account for both
  // the accumulated elapsed_seconds and the time since the current run started
  const start = new Date(startDate);
  const currentRunSecs = (nowMs - start.getTime()) / 1000;
  // The elapsed_seconds from DB already includes previous runs. During a running session,
  // the manager adjusts start_time so _get_elapsed_minutes works correctly,
  // but the DB elapsed_seconds may lag. Use whichever is greater.
  const totalElapsed = Math.max(elapsedSecs, currentRunSecs);
  const remainSecs = Math.max(0, limitMins * 60 - totalElapsed);
  const mins = Math.floor(remainSecs / 60);
  const secs = Math.floor(remainSecs % 60);
  return `${mins}:${String(secs).padStart(2, "0")}`;
}
