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

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError("");
      setCopied(false);
      try {
        const response = await fetch(`/api/sessions/${sessionId}/report`);
        if (!response.ok) {
          if (response.status === 404) {
            throw new Error("Report not generated yet");
          }
          throw new Error("Failed to load report");
        }
        const data = await response.json();
        if (!cancelled) {
          setReport(data.report || "");
          setPath(data.path || null);
        }
      } catch (err: any) {
        if (!cancelled) {
          setError(err?.message || "Unable to load report");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const fileName = useMemo(() => {
    return `report_${sessionId}.md`;
  }, [sessionId]);

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
    <div className="card">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <h3 className="text-lg font-semibold">Report Preview</h3>
          <p className="text-xs text-gray-400">
            Rendered from the latest `report.md`
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button className="btn btn-secondary text-sm" onClick={handleCopy} disabled={!report}>
            {copied ? "Copied" : "Copy Markdown"}
          </button>
          <button className="btn btn-primary text-sm" onClick={handleDownload} disabled={!report}>
            Download
          </button>
        </div>
      </div>

      {path ? (
        <div className="text-xs text-gray-500 mb-4">
          Saved at: <span className="font-mono">{path}</span>
        </div>
      ) : null}

      {loading ? (
        <div className="text-sm text-gray-400">Loading report…</div>
      ) : error ? (
        <div className="text-sm text-error">
          {error}. Run a session to completion to generate a report.
        </div>
      ) : report.trim().length === 0 ? (
        <div className="text-sm text-gray-400">
          Report is empty. Try rerunning research or exporting findings.
        </div>
      ) : (
        <div className="report-markdown max-h-[36rem] overflow-y-auto pr-2">
          <ReactMarkdown
            components={{
              h1: ({ children }) => (
                <h1 className="text-2xl font-bold mt-6 mb-3">{children}</h1>
              ),
              h2: ({ children }) => (
                <h2 className="text-xl font-semibold mt-5 mb-2">{children}</h2>
              ),
              h3: ({ children }) => (
                <h3 className="text-lg font-semibold mt-4 mb-2">{children}</h3>
              ),
              p: ({ children }) => (
                <p className="text-sm text-gray-200 leading-relaxed mb-3">{children}</p>
              ),
              ul: ({ children }) => (
                <ul className="list-disc list-inside text-sm text-gray-200 mb-3 space-y-1">
                  {children}
                </ul>
              ),
              ol: ({ children }) => (
                <ol className="list-decimal list-inside text-sm text-gray-200 mb-3 space-y-1">
                  {children}
                </ol>
              ),
              blockquote: ({ children }) => (
                <blockquote className="border-l-2 border-primary/40 pl-3 italic text-sm text-gray-300 mb-3">
                  {children}
                </blockquote>
              ),
              code: ({ children }) => (
                <code className="text-xs bg-dark-bg/80 border border-dark-border rounded px-1.5 py-0.5">
                  {children}
                </code>
              ),
              pre: ({ children }) => (
                <pre className="text-xs bg-dark-bg/80 border border-dark-border rounded-lg p-3 overflow-x-auto mb-3">
                  {children}
                </pre>
              ),
            }}
          >
            {report}
          </ReactMarkdown>
        </div>
      )}
    </div>
  );
}
