"use client";

import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";

interface ReportPreviewProps {
  sessionId: string;
}

export default function ReportPreview({ sessionId }: ReportPreviewProps) {
  const [report, setReport] = useState<string>("");
  const [path, setPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [format, setFormat] = useState<"md" | "pdf" | "json">("md");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError("");
      setCopied(false);
      try {
        const response = await fetch(`/api/sessions/${sessionId}/report`);
        if (!response.ok) {
          if (response.status === 404) throw new Error("Report not generated yet");
          throw new Error("Failed to load report");
        }
        const data = await response.json();
        if (!cancelled) {
          setReport(data.report || "");
          setPath(data.path || null);
        }
      } catch (err: unknown) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Unable to load report");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [sessionId]);

  const fileName = useMemo(() => `report_${sessionId}.md`, [sessionId]);

  const tocItems = useMemo(() => {
    if (!report) return [];
    const lines = report.split("\n");
    const items: { level: number; text: string; id: string }[] = [];
    for (const line of lines) {
      const match = line.match(/^(#{1,3})\s+(.+)/);
      if (match) {
        const level = match[1].length;
        const text = match[2].replace(/\*\*/g, "").trim();
        const id = text.toLowerCase().replace(/[^a-z0-9]+/g, "-");
        items.push({ level, text, id });
      }
    }
    return items;
  }, [report]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(report);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  const handleDownload = () => {
    const blob = new Blob([report], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col lg:flex-row gap-4" style={{ minHeight: "36rem" }}>
      {/* Table of Contents Sidebar */}
      {tocItems.length > 0 && (
        <aside className="lg:w-64 shrink-0">
          <div className="card sticky top-40">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
              Table of Contents
            </h3>
            <nav className="space-y-1 max-h-96 overflow-y-auto scrollbar-hide">
              {tocItems.map((item, i) => (
                <button
                  key={i}
                  className={`block w-full text-left text-sm truncate py-1 transition-colors hover:text-primary ${item.level === 1
                      ? "text-gray-200 font-medium"
                      : item.level === 2
                        ? "text-gray-400 pl-4"
                        : "text-gray-500 pl-8 text-xs"
                    }`}
                >
                  {item.text}
                </button>
              ))}
            </nav>
          </div>
        </aside>
      )}

      {/* Main Report Content */}
      <div className="flex-1 card">
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-3 mb-6 pb-4 border-b border-dark-border">
          <div>
            <h3 className="text-lg font-semibold">Research Report</h3>
            <p className="text-xs text-gray-500 mt-1">
              {path ? (
                <>Saved at: <span className="font-mono">{path}</span></>
              ) : (
                "Rendered from latest report.md"
              )}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {/* Format Toggles */}
            <div className="flex bg-dark-bg rounded-lg p-0.5 border border-dark-border">
              {(["md", "pdf", "json"] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => setFormat(f)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${format === f
                      ? "bg-surface-highlight text-primary"
                      : "text-gray-500 hover:text-white"
                    }`}
                >
                  {f.toUpperCase()}
                </button>
              ))}
            </div>
            <button className="btn btn-ghost text-sm" onClick={handleCopy} disabled={!report}>
              <span className="material-symbols-outlined text-base">
                {copied ? "check" : "content_copy"}
              </span>
              {copied ? "Copied" : "Copy"}
            </button>
            <button className="btn btn-primary text-sm" onClick={handleDownload} disabled={!report}>
              <span className="material-symbols-outlined text-base">download</span>
              Download
            </button>
          </div>
        </div>

        {/* Content */}
        {loading ? (
          <div className="flex items-center gap-3 text-gray-400">
            <span className="material-symbols-outlined animate-spin">progress_activity</span>
            <span className="text-sm">Loading report…</span>
          </div>
        ) : error ? (
          <div className="text-center py-12">
            <span className="material-symbols-outlined text-4xl text-gray-600 mb-3 block">article</span>
            <p className="text-sm text-error mb-1">{error}</p>
            <p className="text-xs text-gray-500">Run a session to completion to generate a report.</p>
          </div>
        ) : report.trim().length === 0 ? (
          <div className="text-center py-12">
            <span className="material-symbols-outlined text-4xl text-gray-600 mb-3 block">draft</span>
            <p className="text-sm text-gray-400">Report is empty.</p>
            <p className="text-xs text-gray-500 mt-1">Try rerunning research or exporting findings.</p>
          </div>
        ) : (
          <div className="report-markdown max-h-[42rem] overflow-y-auto pr-2">
            <ReactMarkdown
              components={{
                h1: ({ children }) => <h1 className="text-2xl font-bold mt-8 mb-4 text-white">{children}</h1>,
                h2: ({ children }) => <h2 className="text-xl font-semibold mt-6 mb-3 text-white">{children}</h2>,
                h3: ({ children }) => <h3 className="text-lg font-semibold mt-5 mb-2 text-white">{children}</h3>,
                p: ({ children }) => <p className="text-sm text-gray-200 leading-relaxed mb-4">{children}</p>,
                ul: ({ children }) => <ul className="list-disc list-outside ml-6 text-sm text-gray-200 mb-4 space-y-2">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal list-outside ml-6 text-sm text-gray-200 mb-4 space-y-2">{children}</ol>,
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-primary/50 pl-6 italic text-gray-400 my-6 bg-primary/5 py-3 pr-4 rounded-r-lg">
                    {children}
                  </blockquote>
                ),
                code: ({ children }) => (
                  <code className="text-xs bg-dark-bg/80 border border-dark-border rounded px-1.5 py-0.5 text-primary-light">
                    {children}
                  </code>
                ),
                pre: ({ children }) => (
                  <pre className="text-xs bg-terminal-black border border-terminal-border rounded-lg p-4 overflow-x-auto mb-4">
                    {children}
                  </pre>
                ),
                a: ({ children, href }) => (
                  <a href={href} target="_blank" rel="noreferrer" className="text-primary hover:underline">
                    {children}
                  </a>
                ),
              }}
            >
              {report}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}
