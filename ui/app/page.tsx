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

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/sessions/");
      const data = await response.json();
      setSessions(data);
    } catch (error) {
      console.error("Failed to fetch sessions:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleNewSessionSuccess = () => {
    setShowNewSession(false);
    fetchSessions();
  };

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Header */}
      <header className="border-b border-dark-border bg-dark-surface/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-bold text-gradient">Claude Researcher</h1>
            <span className="badge bg-primary/20 text-primary">v0.1.0</span>
            <button
              onClick={() => router.push('/test-websocket')}
              className="text-xs text-gray-400 hover:text-primary transition-colors"
              title="Test WebSocket Events"
            >
              [Test WebSocket]
            </button>
          </div>
          <button
            onClick={() => setShowNewSession(true)}
            className="btn btn-primary flex items-center gap-2"
          >
            <span>+</span>
            <span>New Research</span>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-2">Research Sessions</h2>
          <p className="text-gray-400">
            Start a new research session or continue from where you left off
          </p>
        </div>

        {/* Sessions Grid */}
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-gray-400">Loading sessions...</div>
          </div>
        ) : sessions.length === 0 ? (
          <div className="card text-center py-16">
            <div className="text-gray-400 mb-4">
              <svg
                className="w-16 h-16 mx-auto mb-4 opacity-50"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <p className="text-lg font-medium">No research sessions yet</p>
              <p className="text-sm mt-2">Click "New Research" to get started</p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {sessions.map((session) => (
              <div
                key={session.session_id}
                onClick={() => router.push(`/session/${session.session_id}`)}
                className="card hover:border-primary/50 cursor-pointer transition-all hover:scale-[1.02] group"
              >
                {/* Header */}
                <div className="flex items-start justify-between mb-3">
                  <span className={`badge ${getStatusBadgeClass(session.status)}`}>
                    {session.status}
                  </span>
                  <span className="text-xs text-gray-500 font-mono">
                    #{session.session_id}
                  </span>
                </div>

                {/* Goal */}
                <h3 className="font-semibold text-lg mb-3 line-clamp-3 group-hover:text-primary transition-colors">
                  {session.goal}
                </h3>

                {/* Metadata */}
                <div className="space-y-2 text-sm text-gray-400">
                  <div className="flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>{session.time_limit} minutes</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span>{formatDate(session.created_at)}</span>
                  </div>
                  {session.completed_at && (
                    <div className="text-xs text-success">
                      ✓ Completed {formatDate(session.completed_at)}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* New Session Modal */}
      {showNewSession && (
        <NewSessionForm
          onClose={() => setShowNewSession(false)}
          onSuccess={handleNewSessionSuccess}
        />
      )}
    </div>
  );
}

function getStatusBadgeClass(status: string): string {
  switch (status) {
    case "active":
    case "running":
      return "badge-action";
    case "completed":
      return "badge-finding";
    case "error":
      return "badge-error";
    default:
      return "badge-thinking";
  }
}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: date.getFullYear() !== now.getFullYear() ? "numeric" : undefined,
  });
}
