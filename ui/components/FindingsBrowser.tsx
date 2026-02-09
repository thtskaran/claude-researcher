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

interface SourceIndexItem {
  source_url: string;
  findings_count: number;
  avg_confidence: number | null;
  last_seen: string;
  domain?: string | null;
  final_score?: number | null;
  credibility_label?: string | null;
}

interface FindingsBrowserProps {
  sessionId: string;
}

export default function FindingsBrowser({ sessionId }: FindingsBrowserProps) {
  const [findings, setFindings] = useState<Finding[]>([]);
  const [sources, setSources] = useState<SourceIndexItem[]>([]);
  const [loadingFindings, setLoadingFindings] = useState(true);
  const [loadingSources, setLoadingSources] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [findingType, setFindingType] = useState("all");
  const [minConfidence, setMinConfidence] = useState(0);
  const [order, setOrder] = useState<"desc" | "asc">("desc");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoadingFindings(true);
      setError("");
      try {
        const params = new URLSearchParams();
        params.set("limit", "500");
        params.set("order", order);
        if (search.trim()) {
          params.set("search", search.trim());
        }
        if (findingType !== "all") {
          params.set("finding_type", findingType);
        }
        if (minConfidence > 0) {
          params.set("min_confidence", (minConfidence / 100).toFixed(2));
        }
        const response = await fetch(
          `/api/sessions/${sessionId}/findings?${params.toString()}`
        );
        if (!response.ok) {
          throw new Error("Failed to load findings");
        }
        const data: Finding[] = await response.json();
        if (!cancelled) {
          setFindings(data || []);
        }
      } catch (err) {
        if (!cancelled) {
          setError("Unable to load findings");
        }
      } finally {
        if (!cancelled) {
          setLoadingFindings(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [sessionId, search, findingType, minConfidence, order]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoadingSources(true);
      try {
        const response = await fetch(
          `/api/sessions/${sessionId}/sources?limit=200`
        );
        if (!response.ok) {
          throw new Error("Failed to load sources");
        }
        const data: SourceIndexItem[] = await response.json();
        if (!cancelled) {
          setSources(data || []);
        }
      } catch (err) {
        if (!cancelled) {
          setSources([]);
        }
      } finally {
        if (!cancelled) {
          setLoadingSources(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const typeOptions = useMemo(() => {
    const types = new Set(findings.map((f) => f.finding_type?.toLowerCase()));
    return ["all", ...Array.from(types).filter(Boolean)];
  }, [findings]);

  const filteredSources = useMemo(() => {
    if (!search.trim()) {
      return sources;
    }
    const q = search.toLowerCase();
    return sources.filter((s) =>
      s.source_url.toLowerCase().includes(q) ||
      (s.domain || "").toLowerCase().includes(q)
    );
  }, [sources, search]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-4">
        <div className="card">
          <div className="flex flex-wrap gap-3 items-center justify-between">
            <div className="flex flex-wrap gap-2 items-center">
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search findings or sources..."
                className="input h-9 w-full md:w-64 text-sm"
              />
              <select
                value={findingType}
                onChange={(e) => setFindingType(e.target.value)}
                className="input h-9 text-sm"
              >
                {typeOptions.map((type) => (
                  <option key={type} value={type}>
                    {type === "all" ? "All types" : type}
                  </option>
                ))}
              </select>
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <span>Min confidence</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="5"
                  value={minConfidence}
                  onChange={(e) => setMinConfidence(parseInt(e.target.value))}
                  className="accent-primary"
                />
                <span>{minConfidence}%</span>
              </div>
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

        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Findings</h3>
            <span className="text-xs text-gray-400">
              {loadingFindings ? "Loading…" : `${findings.length} items`}
            </span>
          </div>

          {error ? (
            <div className="text-error text-sm">{error}</div>
          ) : null}

          {loadingFindings ? (
            <div className="text-sm text-gray-400">Loading findings…</div>
          ) : findings.length === 0 ? (
            <div className="text-sm text-gray-400">
              No findings yet. Run a research session to populate this view.
            </div>
          ) : (
            <div className="max-h-[36rem] overflow-y-auto space-y-3 pr-1">
              {findings.map((finding) => (
                <div
                  key={finding.id}
                  className="border border-dark-border rounded-lg p-3 bg-dark-bg/60"
                >
                  <div className="flex flex-wrap items-center gap-2 text-xs text-gray-400 mb-2">
                    <span className={`badge ${getFindingBadge(finding.finding_type)}`}>
                      {finding.finding_type}
                    </span>
                    {typeof finding.confidence === "number" ? (
                      <span className="badge badge-system">
                        {Math.round(finding.confidence * 100)}% confidence
                      </span>
                    ) : null}
                    <span>{formatDate(finding.created_at)}</span>
                    {finding.verification_status ? (
                      <span className="badge badge-system">
                        {finding.verification_status}
                      </span>
                    ) : null}
                  </div>
                  <p className="text-sm text-gray-200 leading-relaxed">
                    {finding.content}
                  </p>
                  {finding.source_url ? (
                    <div className="mt-2 text-xs">
                      <a
                        href={finding.source_url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-info hover:underline break-all"
                      >
                        {finding.source_url}
                      </a>
                    </div>
                  ) : null}
                  {finding.search_query ? (
                    <div className="mt-2 text-xs text-gray-500">
                      Query: {finding.search_query}
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="space-y-4">
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Sources Index</h3>
            <span className="text-xs text-gray-400">
              {loadingSources ? "Loading…" : `${filteredSources.length} sources`}
            </span>
          </div>

          {loadingSources ? (
            <div className="text-sm text-gray-400">Loading sources…</div>
          ) : filteredSources.length === 0 ? (
            <div className="text-sm text-gray-400">
              Sources will appear once findings are collected.
            </div>
          ) : (
            <div className="max-h-[36rem] overflow-y-auto space-y-3 pr-1">
              {filteredSources.map((source) => (
                <div
                  key={source.source_url}
                  className="border border-dark-border rounded-lg p-3 bg-dark-bg/60"
                >
                  <div className="text-xs text-gray-400 mb-2 flex items-center justify-between">
                    <span>{formatDate(source.last_seen)}</span>
                    <span className="badge badge-system">
                      {source.findings_count} findings
                    </span>
                  </div>
                  <a
                    href={source.source_url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-sm text-info hover:underline break-all"
                  >
                    {source.source_url}
                  </a>
                  <div className="mt-2 flex flex-wrap gap-2 text-xs text-gray-400">
                    {source.domain ? (
                      <span className="badge badge-system">{source.domain}</span>
                    ) : null}
                    {source.credibility_label ? (
                      <span className="badge badge-system">
                        {source.credibility_label}
                      </span>
                    ) : null}
                    {source.final_score !== null && source.final_score !== undefined ? (
                      <span className="badge badge-system">
                        Credibility {Math.round(source.final_score * 100)}%
                      </span>
                    ) : null}
                    {source.avg_confidence !== null && source.avg_confidence !== undefined ? (
                      <span className="badge badge-system">
                        Avg {Math.round(source.avg_confidence * 100)}%
                      </span>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
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
    case "fact":
      return "badge-finding";
    case "insight":
      return "badge-thinking";
    case "question":
      return "badge-action";
    case "connection":
      return "badge-action";
    case "source":
      return "badge-system";
    default:
      return "badge-finding";
  }
}
