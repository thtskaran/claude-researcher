"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import ActivityFeed from "@/components/ActivityFeed";
import FindingsBrowser from "@/components/FindingsBrowser";
import ReportPreview from "@/components/ReportPreview";
import Link from "next/link";

interface Session {
  session_id: string;
  goal: string;
  time_limit: number;
  status: string;
  created_at: string;
  completed_at?: string | null;
}

interface Finding {
  id: number;
  content: string;
  source_url?: string;
  confidence: number;
  created_at: string;
}

export default function SessionDetail() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.id as string;

  const [session, setSession] = useState<Session | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState<"activity" | "findings" | "report" | "graph">("activity");

  useEffect(() => {
    fetchSession();
  }, [sessionId]);

  const fetchSession = async () => {
    try {
      const response = await fetch(`http://localhost:8080/api/sessions/${sessionId}`);

      if (!response.ok) {
        if (response.status === 404) {
          setError("Session not found");
        } else {
          throw new Error("Failed to fetch session");
        }
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

  if (loading) {
    return (
      <div className="min-h-screen bg-dark-bg flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-gray-400">Loading session...</p>
        </div>
      </div>
    );
  }

  if (error || !session) {
    return (
      <div className="min-h-screen bg-dark-bg flex items-center justify-center">
        <div className="text-center">
          <svg
            className="w-16 h-16 text-error mx-auto mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
          <h2 className="text-2xl font-bold mb-2">{error || "Session not found"}</h2>
          <Link href="/" className="text-primary hover:underline">
            ← Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Header */}
      <header className="border-b border-dark-border bg-dark-surface/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center gap-4 mb-3">
            <Link href="/" className="text-gray-400 hover:text-primary transition-colors">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </Link>
            <h1 className="text-2xl font-bold line-clamp-1">{session.goal}</h1>
          </div>

          <div className="flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <span className={`badge ${getStatusBadgeClass(session.status)}`}>
                {session.status}
              </span>
            </div>
            <div className="flex items-center gap-2 text-gray-400">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
              </svg>
              <span className="font-mono">#{session.session_id}</span>
            </div>
            <div className="flex items-center gap-2 text-gray-400">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{session.time_limit} minutes</span>
            </div>
            <div className="flex items-center gap-2 text-gray-400">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <span>{formatDate(session.created_at)}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Tabs */}
      <div className="border-b border-dark-border bg-dark-surface/30">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-6">
            <button
              onClick={() => setActiveTab("activity")}
              className={`py-3 px-2 border-b-2 transition-colors ${
                activeTab === "activity"
                  ? "border-primary text-primary"
                  : "border-transparent text-gray-400 hover:text-gray-200"
              }`}
            >
              Live Activity
            </button>
            <button
              onClick={() => setActiveTab("findings")}
              className={`py-3 px-2 border-b-2 transition-colors ${
                activeTab === "findings"
                  ? "border-primary text-primary"
                  : "border-transparent text-gray-400 hover:text-gray-200"
              }`}
            >
              Findings
            </button>
            <button
              onClick={() => setActiveTab("report")}
              className={`py-3 px-2 border-b-2 transition-colors ${
                activeTab === "report"
                  ? "border-primary text-primary"
                  : "border-transparent text-gray-400 hover:text-gray-200"
              }`}
            >
              Report
            </button>
            <button
              onClick={() => setActiveTab("graph")}
              className={`py-3 px-2 border-b-2 transition-colors ${
                activeTab === "graph"
                  ? "border-primary text-primary"
                  : "border-transparent text-gray-400 hover:text-gray-200"
              }`}
            >
              Knowledge Graph
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === "activity" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main Activity Feed */}
            <div className="lg:col-span-2">
              <ActivityFeed sessionId={sessionId} />
            </div>

            {/* Side Panel - Agent Status */}
            <div className="space-y-6">
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Agent Status</h3>
                <div className="space-y-3">
                  <AgentStatusCard name="Director" status="active" />
                  <AgentStatusCard name="Manager" status="thinking" />
                  <AgentStatusCard name="Intern Pool" status="active" count={3} />
                </div>
              </div>

              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Quick Stats</h3>
                <div className="space-y-2 text-sm">
                  <StatRow label="Findings" value="--" />
                  <StatRow label="Sources" value="--" />
                  <StatRow label="Topics" value="--" />
                  <StatRow label="Elapsed" value={getElapsedTime(session.created_at)} />
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === "findings" && (
          <FindingsBrowser sessionId={sessionId} />
        )}

        {activeTab === "report" && (
          <ReportPreview sessionId={sessionId} />
        )}

        {activeTab === "graph" && (
          <div className="card">
            <h2 className="text-2xl font-bold mb-6">Knowledge Graph</h2>
            <div className="text-center py-12 text-gray-400">
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
                  d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"
                />
              </svg>
              <p className="text-lg">Interactive graph visualization coming soon</p>
              <p className="text-sm mt-2">Will show entities, relationships, and knowledge network</p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function AgentStatusCard({ name, status, count }: { name: string; status: string; count?: number }) {
  const statusColors = {
    active: "bg-success",
    thinking: "bg-primary",
    idle: "bg-gray-500",
  };

  return (
    <div className="flex items-center justify-between p-3 bg-dark-bg rounded-lg border border-dark-border">
      <div className="flex items-center gap-3">
        <div className={`w-2 h-2 rounded-full ${statusColors[status as keyof typeof statusColors]} animate-pulse`} />
        <span className="font-medium">{name}</span>
        {count && <span className="text-xs text-gray-500">({count})</span>}
      </div>
      <span className="text-xs text-gray-400 capitalize">{status}</span>
    </div>
  );
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between py-2 border-b border-dark-border last:border-0">
      <span className="text-gray-400">{label}</span>
      <span className="font-semibold">{value}</span>
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
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getElapsedTime(startDate: string): string {
  const start = new Date(startDate);
  const now = new Date();
  const diffMs = now.getTime() - start.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);

  if (diffMins < 60) return `${diffMins}m`;
  return `${diffHours}h ${diffMins % 60}m`;
}
