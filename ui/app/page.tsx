"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import NewSessionForm from "@/components/NewSessionForm";

interface Session {
  session_id: string;
  goal: string;
  time_limit: number;
  status: string;
  created_at: string;
  completed_at?: string | null;
}

export default function Home() {
  const router = useRouter();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNewSession, setShowNewSession] = useState(false);

  const fetchSessions = async () => {
    try {
      const response = await fetch("/api/sessions/?limit=50");
      if (response.ok) {
        const data = await response.json();
        setSessions(data);
      }
    } catch (err) {
      console.error("Failed to fetch sessions:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  const activeSessions = sessions.filter((s) => s.status === "running" || s.status === "pending");
  const completedSessions = sessions.filter((s) => s.status !== "running" && s.status !== "pending");

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-edge bg-card/50 backdrop-blur-sm sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-sage-soft text-sage">
              <span className="material-symbols-outlined text-xl">science</span>
            </div>
            <div>
              <h1 className="text-lg font-display tracking-tight">Claude Researcher</h1>
              <p className="text-xs text-ink-muted">Deep research with glass-box transparency</p>
            </div>
          </div>
          <button
            onClick={() => setShowNewSession(true)}
            className="btn btn-primary"
          >
            <span className="material-symbols-outlined text-lg">add</span>
            New Research
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8 stagger-children">
          <div className="card card-hover animate-fade-up-in bg-gradient-to-br from-sage/5 to-transparent">
            <div className="flex justify-between items-start mb-2">
              <p className="text-sm font-medium text-ink-secondary">Total Sessions</p>
              <span className="material-symbols-outlined text-ink-secondary">folder_open</span>
            </div>
            <p className="font-mono text-3xl font-bold tracking-tighter">{sessions.length}</p>
          </div>
          <div className="card card-hover animate-fade-up-in bg-gradient-to-br from-olive/5 to-transparent">
            <div className="flex justify-between items-start mb-2">
              <p className="text-sm font-medium text-ink-secondary">Active</p>
              <span className="material-symbols-outlined text-olive">play_circle</span>
            </div>
            <p className="font-mono text-3xl font-bold tracking-tighter text-olive">{activeSessions.length}</p>
          </div>
          <div className="card card-hover animate-fade-up-in bg-gradient-to-br from-iris/5 to-transparent">
            <div className="flex justify-between items-start mb-2">
              <p className="text-sm font-medium text-ink-secondary">Completed</p>
              <span className="material-symbols-outlined text-ink-secondary">check_circle</span>
            </div>
            <p className="font-mono text-3xl font-bold tracking-tighter">{completedSessions.length}</p>
          </div>
        </div>

        {/* Active Sessions */}
        {activeSessions.length > 0 && (
          <div className="mb-8">
            <h2 className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-4 px-1">
              Active Sessions
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {activeSessions.map((session) => (
                <SessionCard key={session.session_id} session={session} onClick={() => router.push(`/session/${session.session_id}`)} />
              ))}
            </div>
          </div>
        )}

        {/* Session History */}
        <div>
          <h2 className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-4 px-1">
            {activeSessions.length > 0 ? "History" : "All Sessions"}
          </h2>

          {loading ? (
            <div className="card">
              <div className="flex items-center gap-3 text-ink-secondary">
                <span className="material-symbols-outlined animate-spin">progress_activity</span>
                <span className="text-sm">Loading sessions...</span>
              </div>
            </div>
          ) : completedSessions.length === 0 && activeSessions.length === 0 ? (
            <div className="card flex flex-col items-center justify-center py-16 text-center">
              <div className="w-16 h-16 rounded-2xl bg-sage-soft flex items-center justify-center mb-4">
                <span className="material-symbols-outlined text-sage text-3xl">science</span>
              </div>
              <h3 className="text-lg font-display mb-2">No research sessions yet</h3>
              <p className="text-sm text-ink-secondary max-w-md mb-6">
                Start your first research session to explore topics with hierarchical deep research and transparent AI reasoning.
              </p>
              <button
                onClick={() => setShowNewSession(true)}
                className="btn btn-primary"
              >
                <span className="material-symbols-outlined text-lg">rocket_launch</span>
                Start Research
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {completedSessions.map((session) => (
                <SessionCard key={session.session_id} session={session} onClick={() => router.push(`/session/${session.session_id}`)} />
              ))}
            </div>
          )}
        </div>
      </main>

      {/* New Session Modal */}
      {showNewSession && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4">
          <div className="w-full max-w-[640px]">
            <NewSessionForm
              onClose={() => setShowNewSession(false)}
              onSuccess={() => {
                setShowNewSession(false);
                fetchSessions();
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function SessionCard({ session, onClick }: { session: Session; onClick: () => void }) {
  const isActive = session.status === "running" || session.status === "pending";
  const timeDisplay = isActive
    ? getElapsedTime(session.created_at)
    : getDuration(session.created_at, session.completed_at);

  return (
    <button
      onClick={onClick}
      className="card card-hover text-left w-full group relative overflow-hidden"
    >
      {isActive && (
        <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-sage/40 via-sage to-sage/40" />
      )}

      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          {isActive ? (
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-soft-pulse absolute inline-flex h-full w-full rounded-full bg-olive opacity-75" />
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-olive" />
            </span>
          ) : (
            <span className="status-dot status-dot-idle" />
          )}
          <span className={`text-xs font-mono font-medium ${isActive ? "text-olive" : "text-ink-muted"}`}>
            {session.status}
          </span>
        </div>
        <span className="material-symbols-outlined text-ink-muted group-hover:text-sage transition-colors text-lg">
          arrow_forward
        </span>
      </div>

      <h3 className="text-sm font-medium text-ink group-hover:text-sage transition-colors line-clamp-2 mb-3 leading-snug">
        {session.goal}
      </h3>

      <div className="flex items-center justify-between text-xs text-ink-muted font-mono">
        <span className="flex items-center gap-1">
          <span className="material-symbols-outlined text-[14px]">{isActive ? "schedule" : "timer"}</span>
          {timeDisplay}
        </span>
        <span className="flex items-center gap-1">
          <span className="material-symbols-outlined text-[14px]">hourglass_empty</span>
          {session.time_limit}m limit
        </span>
      </div>
    </button>
  );
}

function getElapsedTime(startDateString: string): string {
  const now = new Date();
  const start = new Date(startDateString);
  const diffMs = now.getTime() - start.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);

  if (diffMins < 1) return "just started";
  if (diffMins < 60) return `${diffMins}m elapsed`;
  if (diffHours < 24) return `${diffHours}h ${diffMins % 60}m elapsed`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ${diffHours % 24}h elapsed`;
}

function getDuration(startDateString: string, endDateString?: string | null): string {
  if (!endDateString) {
    const now = new Date();
    const start = new Date(startDateString);
    const diffMs = now.getTime() - start.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return start.toLocaleDateString("en-US", { month: "short", day: "numeric" });
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
