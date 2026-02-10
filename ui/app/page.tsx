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
      <header className="border-b border-dark-border bg-dark-surface/50 backdrop-blur-sm sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary/10 text-primary">
              <span className="material-symbols-outlined text-xl">science</span>
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">Claude Researcher</h1>
              <p className="text-xs text-gray-500">Deep research with glass-box transparency</p>
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
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="card card-hover">
            <div className="flex justify-between items-start mb-2">
              <p className="text-sm font-medium text-gray-400">Total Sessions</p>
              <span className="material-symbols-outlined text-gray-400">folder_open</span>
            </div>
            <p className="font-mono text-3xl font-bold tracking-tighter">{sessions.length}</p>
          </div>
          <div className="card card-hover">
            <div className="flex justify-between items-start mb-2">
              <p className="text-sm font-medium text-gray-400">Active</p>
              <span className="material-symbols-outlined text-accent-success">play_circle</span>
            </div>
            <p className="font-mono text-3xl font-bold tracking-tighter text-accent-success">{activeSessions.length}</p>
          </div>
          <div className="card card-hover">
            <div className="flex justify-between items-start mb-2">
              <p className="text-sm font-medium text-gray-400">Completed</p>
              <span className="material-symbols-outlined text-gray-400">check_circle</span>
            </div>
            <p className="font-mono text-3xl font-bold tracking-tighter">{completedSessions.length}</p>
          </div>
        </div>

        {/* Active Sessions */}
        {activeSessions.length > 0 && (
          <div className="mb-8">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4 px-1">
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
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4 px-1">
            {activeSessions.length > 0 ? "History" : "All Sessions"}
          </h2>

          {loading ? (
            <div className="card">
              <div className="flex items-center gap-3 text-gray-400">
                <span className="material-symbols-outlined animate-spin">progress_activity</span>
                <span className="text-sm">Loading sessions...</span>
              </div>
            </div>
          ) : completedSessions.length === 0 && activeSessions.length === 0 ? (
            <div className="card flex flex-col items-center justify-center py-16 text-center">
              <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
                <span className="material-symbols-outlined text-primary text-3xl">science</span>
              </div>
              <h3 className="text-lg font-semibold mb-2">No research sessions yet</h3>
              <p className="text-sm text-gray-400 max-w-md mb-6">
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
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
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
  const timeAgo = getTimeAgo(session.created_at);

  return (
    <button
      onClick={onClick}
      className="card card-hover text-left w-full group relative overflow-hidden"
    >
      {/* Active indicator bar */}
      {isActive && (
        <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-primary/60 via-primary to-primary/60" />
      )}

      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          {isActive ? (
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-success opacity-75" />
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-accent-success" />
            </span>
          ) : (
            <span className="status-dot status-dot-idle" />
          )}
          <span className={`text-xs font-mono font-medium ${isActive ? "text-accent-success" : "text-gray-500"}`}>
            {session.status}
          </span>
        </div>
        <span className="material-symbols-outlined text-gray-600 group-hover:text-primary transition-colors text-lg">
          arrow_forward
        </span>
      </div>

      <h3 className="text-sm font-medium text-white group-hover:text-primary transition-colors line-clamp-2 mb-3 leading-snug">
        {session.goal}
      </h3>

      <div className="flex items-center justify-between text-xs text-gray-500 font-mono">
        <span className="flex items-center gap-1">
          <span className="material-symbols-outlined text-[14px]">schedule</span>
          {timeAgo}
        </span>
        <span className="flex items-center gap-1">
          <span className="material-symbols-outlined text-[14px]">timer</span>
          {session.time_limit}m
        </span>
      </div>
    </button>
  );
}

function getTimeAgo(dateString: string): string {
  const now = new Date();
  const date = new Date(dateString);
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}
