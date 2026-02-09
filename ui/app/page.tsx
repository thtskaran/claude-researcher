"use client";

import { useState, useEffect } from "react";

interface Session {
  session_id: string;
  goal: string;
  time_limit: number;
  status: string;
  created_at: string;
}

export default function Home() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNewSession, setShowNewSession] = useState(false);

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    try {
      const response = await fetch("http://localhost:8080/api/sessions/");
      const data = await response.json();
      setSessions(data);
    } catch (error) {
      console.error("Failed to fetch sessions:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Header */}
      <header className="border-b border-dark-border bg-dark-surface/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-bold text-gradient">Claude Researcher</h1>
            <span className="badge bg-primary/20 text-primary">v0.1.0</span>
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
                className="card hover:border-primary/50 cursor-pointer transition-all hover:scale-105"
              >
                <div className="flex items-start justify-between mb-3">
                  <span className={`badge ${getStatusBadgeClass(session.status)}`}>
                    {session.status}
                  </span>
                  <span className="text-xs text-gray-500 font-mono">
                    {session.session_id}
                  </span>
                </div>
                <h3 className="font-semibold text-lg mb-2 line-clamp-2">
                  {session.goal}
                </h3>
                <div className="flex items-center gap-4 text-sm text-gray-400">
                  <span>{session.time_limit} min</span>
                  <span>•</span>
                  <span>{new Date(session.created_at).toLocaleDateString()}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* New Session Modal (placeholder) */}
      {showNewSession && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="card max-w-2xl w-full mx-4">
            <h3 className="text-2xl font-bold mb-4">New Research Session</h3>
            <p className="text-gray-400 mb-4">
              Coming soon: Start a new research session from the UI
            </p>
            <button
              onClick={() => setShowNewSession(false)}
              className="btn btn-secondary"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function getStatusBadgeClass(status: string): string {
  switch (status) {
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
