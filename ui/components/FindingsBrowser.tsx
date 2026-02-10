"use client";

import { useEffect, useMemo, useState } from "react";

interface Finding {
  id: number;
  session_id: string;
  content: string;
  finding_type: string;
  source_url?: string | null;
  confidence?: number | null;
  search_query?: string | null;
  created_at: string;
  verification_status?: string | null;
  verification_method?: string | null;
  kg_support_score?: number | null;
}

interface FindingsBrowserProps {
  sessionId: string;
}

export default function FindingsBrowser({ sessionId }: FindingsBrowserProps) {
  const [findings, setFindings] = useState<Finding[]>([]);
  const [loadingFindings, setLoadingFindings] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [findingType, setFindingType] = useState("all");
  const [minConfidence, setMinConfidence] = useState(0);
  const [order, setOrder] = useState<"desc" | "asc">("desc");
  const [selectedId, setSelectedId] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoadingFindings(true);
      setError("");
      try {
        const params = new URLSearchParams();
        params.set("limit", "500");
        params.set("order", order);
        if (search.trim()) params.set("search", search.trim());
        if (findingType !== "all") params.set("finding_type", findingType);
        if (minConfidence > 0) params.set("min_confidence", (minConfidence / 100).toFixed(2));
        const response = await fetch(`/api/sessions/${sessionId}/findings?${params.toString()}`);
        if (!response.ok) throw new Error("Failed to load findings");
        const data: Finding[] = await response.json();
        if (!cancelled) setFindings(data || []);
      } catch {
        if (!cancelled) setError("Unable to load findings");
      } finally {
        if (!cancelled) setLoadingFindings(false);
      }
    })();
    return () => { cancelled = true; };
  }, [sessionId, search, findingType, minConfidence, order]);

  const typeOptions = useMemo(() => {
    const types = new Set(findings.map((f) => f.finding_type?.toLowerCase()));
    return ["all", ...Array.from(types).filter(Boolean)];
  }, [findings]);

  const selectedFinding = useMemo(() => {
    if (selectedId === null) return findings[0] || null;
    return findings.find((f) => f.id === selectedId) || null;
  }, [findings, selectedId]);

  return (
    <div className="flex flex-col gap-4">
      {/* Filter Bar */}
      <div className="card py-4">
        <div className="flex flex-wrap gap-3 items-center">
          <div className="relative flex-1 min-w-[200px] max-w-sm">
            <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 text-lg">search</span>
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search findings or sources..."
              className="input h-9 w-full pl-10 text-sm"
            />
          </div>

          <div className="flex flex-wrap gap-2 items-center">
            {typeOptions.map((type) => (
              <button
                key={type}
                onClick={() => setFindingType(type)}
                className={`chip ${findingType === type ? "chip-active" : ""}`}
              >
                {type === "all" ? "All types" : type}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 text-xs text-gray-400 ml-auto">
            <span>Confidence</span>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={minConfidence}
              onChange={(e) => setMinConfidence(parseInt(e.target.value))}
              className="w-20 accent-primary"
            />
            <span className="font-mono w-8">{minConfidence}%</span>
          </div>

          <select
            value={order}
            onChange={(e) => setOrder(e.target.value as "asc" | "desc")}
            className="input h-9 text-sm"
          >
            <option value="desc">Newest first</option>
            <option value="asc">Oldest first</option>
          </select>
        </div>
      </div>

      {error && <div className="text-error text-sm">{error}</div>}

      {/* Split Pane */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 h-[calc(100vh-14rem)]">
        {/* Left: Findings List */}
        <div className="lg:col-span-2 card p-0 flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-dark-border flex items-center justify-between bg-dark-surface">
            <span className="text-xs font-mono text-gray-500 uppercase tracking-wider">
              Latest Findings
            </span>
            <span className="text-xs text-gray-500">
              {loadingFindings ? "Loading…" : `${findings.length} items`}
            </span>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {loadingFindings ? (
              <div className="text-sm text-gray-400 p-4">Loading findings…</div>
            ) : findings.length === 0 ? (
              <div className="text-sm text-gray-400 p-4 text-center">
                <span className="material-symbols-outlined text-3xl text-gray-600 block mb-2">science</span>
                No findings yet. Run a research session to populate this view.
              </div>
            ) : (
              findings.map((finding) => {
                const isActive = finding.id === (selectedFinding?.id ?? findings[0]?.id);
                return (
                  <button
                    key={finding.id}
                    onClick={() => setSelectedId(finding.id)}
                    className={`w-full text-left flex flex-col gap-2 p-3 rounded-xl transition-all border cursor-pointer ${isActive
                      ? "bg-card-hover border-primary/40 shadow-lg shadow-black/20 ring-1 ring-primary/20"
                      : "bg-dark-bg/60 border-dark-border hover:border-gray-600"
                      }`}
                  >
                    <div className="flex items-center gap-2 text-xs">
                      <span className={`badge ${getFindingBadge(finding.finding_type)}`}>
                        {finding.finding_type}
                      </span>
                      {typeof finding.confidence === "number" && (
                        <ConfidenceRing value={finding.confidence} />
                      )}
                    </div>
                    <p className="text-sm text-gray-200 line-clamp-2 leading-snug">{finding.content}</p>
                    <span className="text-xs text-gray-500 font-mono">{formatDate(finding.created_at)}</span>
                  </button>
                );
              })
            )}
          </div>
        </div>

        {/* Right: Detail View */}
        <div className="lg:col-span-3 card p-0 flex flex-col overflow-hidden">
          {selectedFinding ? (
            <>
              <div className="px-5 py-3 border-b border-dark-border bg-dark-surface">
                <div className="flex items-center gap-2 mb-1.5">
                  <span className={`badge ${getFindingBadge(selectedFinding.finding_type)}`}>
                    {selectedFinding.finding_type}
                  </span>
                  {typeof selectedFinding.confidence === "number" && (
                    <span className="badge badge-system">
                      {Math.round(selectedFinding.confidence * 100)}% confidence
                    </span>
                  )}
                  <span className="text-xs text-gray-500 font-mono ml-auto">{formatDate(selectedFinding.created_at)}</span>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-5 space-y-4">
                <div>
                  <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1.5">Content</h4>
                  <p className="text-sm text-gray-200 leading-relaxed">{selectedFinding.content}</p>
                </div>
                {selectedFinding.source_url && (
                  <div>
                    <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1.5">Source</h4>
                    <a
                      href={selectedFinding.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-sm text-info hover:underline break-all flex items-start gap-2 bg-dark-bg/50 p-2 rounded-lg border border-dark-border/50"
                    >
                      <span className="material-symbols-outlined text-sm mt-0.5 text-gray-500">open_in_new</span>
                      <span className="line-clamp-2">{selectedFinding.source_url}</span>
                    </a>
                  </div>
                )}
                {selectedFinding.search_query && (
                  <div>
                    <h4 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1.5">Search Query</h4>
                    <p className="text-xs text-gray-400 font-mono bg-dark-bg p-2 rounded-lg border border-dark-border/50">
                      {selectedFinding.search_query}
                    </p>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-gray-500">
              <div className="text-center">
                <span className="material-symbols-outlined text-4xl text-gray-600 mb-2 block">description</span>
                <p className="text-sm">Select a finding to view details</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ConfidenceRing({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 70 ? "text-accent-green" : pct >= 40 ? "text-accent-yellow" : "text-accent-red";
  return (
    <span className={`text-xs font-mono font-medium ${color}`}>
      {pct}%
    </span>
  );
}

function formatDate(value: string): string {
  const date = new Date(value);
  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getFindingBadge(type?: string | null): string {
  const normalized = (type || "fact").toLowerCase();
  switch (normalized) {
    case "fact": return "badge-finding";
    case "insight": return "badge-thinking";
    case "question": return "badge-action";
    case "connection": return "badge-action";
    case "source": return "badge-system";
    default: return "badge-finding";
  }
}
