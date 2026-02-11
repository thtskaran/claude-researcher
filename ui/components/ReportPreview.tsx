"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ReportPreviewProps {
  sessionId: string;
}

/** Convert heading text to a stable slug id. */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/\*\*/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

/** Extract plain text from React children (handles nested elements). */
function childrenToText(children: React.ReactNode): string {
  if (typeof children === "string") return children;
  if (typeof children === "number") return String(children);
  if (Array.isArray(children)) return children.map(childrenToText).join("");
  if (children && typeof children === "object" && "props" in children) {
    const el = children as React.ReactElement<{ children?: React.ReactNode }>;
    return childrenToText(el.props.children);
  }
  return "";
}

export default function ReportPreview({ sessionId }: ReportPreviewProps) {
  const [report, setReport] = useState<string>("");
  const [path, setPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [format, setFormat] = useState<"md" | "pdf" | "json">("md");
  const [activeTocId, setActiveTocId] = useState<string | null>(null);

  const reportContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError("");
      setCopied(false);
      try {
        // Fetch report
        const response = await fetch(`/api/sessions/${sessionId}/report`);
        if (!response.ok) {
          if (response.status === 404) throw new Error("Report not generated yet");
          throw new Error("Failed to load report");
        }
        const data = await response.json();

        // Fetch findings for JSON export
        const findingsResponse = await fetch(`/api/sessions/${sessionId}/findings`);
        const findingsData = findingsResponse.ok ? await findingsResponse.json() : { findings: [] };

        if (!cancelled) {
          setReport(data.report || "");
          setPath(data.path || null);
          setFindings(findingsData.findings || []);
        }
      } catch (err: unknown) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Unable to load report");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [sessionId]);

  const [findings, setFindings] = useState<any[]>([]);

  const fileName = useMemo(() => {
    const base = `report_${sessionId}`;
    return format === "md" ? `${base}.md` : format === "json" ? `${base}.json` : `${base}.pdf`;
  }, [sessionId, format]);

  const tocItems = useMemo(() => {
    if (!report) return [];
    const lines = report.split("\n");
    const items: { level: number; text: string; id: string }[] = [];
    for (const line of lines) {
      const match = line.match(/^(#{1,3})\s+(.+)/);
      if (match) {
        const level = match[1].length;
        const text = match[2].replace(/\*\*/g, "").trim();
        const id = slugify(text);
        items.push({ level, text, id });
      }
    }
    return items;
  }, [report]);

  /** Scroll to a heading inside the report container. */
  const scrollToSection = useCallback((id: string) => {
    const container = reportContainerRef.current;
    if (!container) return;
    const target = container.querySelector(`[id="${CSS.escape(id)}"]`);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      setActiveTocId(id);
    }
  }, []);

  /** Handle anchor link clicks — scroll within the report div instead of page navigation. */
  const handleLinkClick = useCallback((e: React.MouseEvent<HTMLAnchorElement>, href: string | undefined) => {
    if (!href) return;
    if (href.startsWith("#")) {
      e.preventDefault();
      const id = href.slice(1);
      scrollToSection(id);
    }
    // External links proceed normally (target="_blank")
  }, [scrollToSection]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(report);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  const handleDownload = async () => {
    if (format === "md") {
      // Export as Markdown
      const blob = new Blob([report], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } else if (format === "json") {
      // Export findings as JSON
      const jsonData = {
        session_id: sessionId,
        report_text: report,
        findings: findings,
        exported_at: new Date().toISOString(),
      };
      const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } else if (format === "pdf") {
      // Export as PDF using browser print
      // Create a temporary div with the markdown rendered
      const printWindow = window.open("", "_blank");
      if (!printWindow) {
        alert("Please allow popups to export PDF");
        return;
      }

      // Convert markdown to HTML for printing
      const htmlContent = `
        <!DOCTYPE html>
        <html>
        <head>
          <title>Research Report - ${sessionId}</title>
          <style>
            @media print {
              @page { margin: 2cm; size: A4; }
            }
            body {
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
              line-height: 1.6;
              color: #1a1915;
              max-width: 800px;
              margin: 0 auto;
              padding: 20px;
            }
            h1 { font-size: 28px; margin-top: 40px; margin-bottom: 20px; page-break-after: avoid; }
            h2 { font-size: 22px; margin-top: 32px; margin-bottom: 16px; page-break-after: avoid; }
            h3 { font-size: 18px; margin-top: 24px; margin-bottom: 12px; page-break-after: avoid; }
            p { margin-bottom: 16px; text-align: justify; }
            ul, ol { margin-bottom: 16px; padding-left: 24px; }
            li { margin-bottom: 8px; }
            blockquote {
              border-left: 4px solid #82af78;
              padding-left: 20px;
              margin: 24px 0;
              font-style: italic;
              background: #f5f5f0;
              padding: 16px 16px 16px 20px;
            }
            code {
              background: #f5f5f0;
              padding: 2px 6px;
              border-radius: 3px;
              font-family: monospace;
              font-size: 0.9em;
            }
            pre {
              background: #f5f5f0;
              padding: 16px;
              border-radius: 8px;
              overflow-x: auto;
              margin-bottom: 16px;
            }
            pre code { background: none; padding: 0; }
            table {
              width: 100%;
              border-collapse: collapse;
              margin-bottom: 20px;
            }
            th, td {
              border: 1px solid #ddd;
              padding: 12px;
              text-align: left;
            }
            th {
              background: #f5f5f0;
              font-weight: 600;
            }
            a { color: #82af78; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .page-break { page-break-before: always; }
          </style>
        </head>
        <body>
          ${report
            .replace(/\n\n/g, "</p><p>")
            .replace(/^# (.+)$/gm, "<h1>$1</h1>")
            .replace(/^## (.+)$/gm, "<h2>$1</h2>")
            .replace(/^### (.+)$/gm, "<h3>$1</h3>")
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
            .replace(/\*(.+?)\*/g, "<em>$1</em>")
            .replace(/\[(\d+)\]/g, '<sup>[$1]</sup>')
            .replace(/^> (.+)$/gm, "<blockquote>$1</blockquote>")
            .replace(/^- (.+)$/gm, "<li>$1</li>")
            .replace(/(<li>.*<\/li>\n?)+/g, "<ul>$&</ul>")
            .replace(/^\d+\. (.+)$/gm, "<li>$1</li>")
            .split("\n").join("<br/>")}
        </body>
        </html>
      `;

      printWindow.document.write(htmlContent);
      printWindow.document.close();

      // Wait for content to load, then trigger print dialog
      printWindow.onload = () => {
        setTimeout(() => {
          printWindow.print();
          // Don't close the window automatically - let user close it after printing
        }, 250);
      };
    }
  };

  /** Build a heading component that includes an id for anchor links. */
  const makeHeading = (Tag: "h1" | "h2" | "h3", className: string) => {
    const HeadingComponent = ({ children }: { children?: React.ReactNode }) => {
      const text = childrenToText(children);
      const id = slugify(text);
      return <Tag id={id} className={className}>{children}</Tag>;
    };
    HeadingComponent.displayName = `Heading_${Tag}`;
    return HeadingComponent;
  };

  return (
    <div className="flex flex-col lg:flex-row gap-4" style={{ minHeight: "36rem" }}>
      {/* Table of Contents Sidebar */}
      {tocItems.length > 0 && (
        <aside className="lg:w-64 shrink-0">
          <div className="card sticky top-40">
            <h3 className="text-xs font-display font-normal text-ink-muted uppercase tracking-wider mb-4">
              Table of Contents
            </h3>
            <nav className="space-y-1 max-h-96 overflow-y-auto scrollbar-hide">
              {tocItems.map((item, i) => (
                <button
                  key={i}
                  onClick={() => scrollToSection(item.id)}
                  className={`block w-full text-left text-sm truncate py-1 transition-colors hover:text-sage ${
                    activeTocId === item.id
                      ? "text-sage font-medium"
                      : item.level === 1
                        ? "text-ink font-medium"
                        : item.level === 2
                          ? "text-ink-secondary pl-4"
                          : "text-ink-muted pl-8 text-xs"
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
        <div className="flex flex-wrap items-center justify-between gap-3 mb-6 pb-4 border-b border-edge">
          <div>
            <h3 className="text-lg font-display">Research Report</h3>
            <p className="text-xs text-ink-muted mt-1">
              {path ? (
                <>Saved at: <span className="font-mono">{path}</span></>
              ) : (
                "Rendered from latest report.md"
              )}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {/* Format Toggles */}
            <div className="flex bg-card-inset rounded-lg p-0.5 border border-edge">
              {(["md", "pdf", "json"] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => setFormat(f)}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${format === f
                      ? "bg-card-hover text-sage"
                      : "text-ink-muted hover:text-ink"
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
          <div className="flex items-center gap-3 text-ink-secondary">
            <span className="material-symbols-outlined animate-spin">progress_activity</span>
            <span className="text-sm">Loading report…</span>
          </div>
        ) : error ? (
          <div className="text-center py-12">
            <span className="material-symbols-outlined text-4xl text-ink-muted mb-3 block">article</span>
            <p className="text-sm text-coral mb-1">{error}</p>
            <p className="text-xs text-ink-muted">Run a session to completion to generate a report.</p>
          </div>
        ) : report.trim().length === 0 ? (
          <div className="text-center py-12">
            <span className="material-symbols-outlined text-4xl text-ink-muted mb-3 block">draft</span>
            <p className="text-sm text-ink-secondary">Report is empty.</p>
            <p className="text-xs text-ink-muted mt-1">Try rerunning research or exporting findings.</p>
          </div>
        ) : (
          <div ref={reportContainerRef} className="report-markdown max-h-[42rem] overflow-y-auto pr-2 scroll-smooth">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h1: makeHeading("h1", "text-2xl font-display mt-8 mb-4 text-ink scroll-mt-4"),
                h2: makeHeading("h2", "text-xl font-display font-normal mt-6 mb-3 text-ink scroll-mt-4"),
                h3: makeHeading("h3", "text-lg font-semibold mt-5 mb-2 text-ink scroll-mt-4"),
                p: ({ children }) => <p className="text-sm text-ink leading-relaxed mb-4">{children}</p>,
                ul: ({ children }) => <ul className="list-disc list-outside ml-6 text-sm text-ink mb-4 space-y-2">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal list-outside ml-6 text-sm text-ink mb-4 space-y-2">{children}</ol>,
                li: ({ children }) => <li className="text-sm text-ink leading-relaxed">{children}</li>,
                strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
                em: ({ children }) => <em className="italic text-ink-secondary">{children}</em>,
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-sage/50 pl-6 italic text-ink-secondary my-6 bg-sage/5 py-3 pr-4 rounded-r-lg">
                    {children}
                  </blockquote>
                ),
                hr: () => <hr className="border-edge my-6" />,
                // Tables (requires remark-gfm)
                table: ({ children }) => (
                  <div className="overflow-x-auto mb-6 rounded-lg border border-edge">
                    <table className="w-full text-sm">{children}</table>
                  </div>
                ),
                thead: ({ children }) => <thead className="bg-card-inset border-b border-edge">{children}</thead>,
                tbody: ({ children }) => <tbody className="divide-y divide-edge/50">{children}</tbody>,
                tr: ({ children }) => <tr className="hover:bg-card-hover/50 transition-colors">{children}</tr>,
                th: ({ children }) => (
                  <th className="text-left text-xs font-semibold text-ink-secondary uppercase tracking-wider px-4 py-2.5">
                    {children}
                  </th>
                ),
                td: ({ children }) => <td className="px-4 py-2.5 text-sm text-ink">{children}</td>,
                code: ({ children, className }) => {
                  // Detect code blocks (inside <pre>) vs inline code
                  const isBlock = className?.startsWith("language-");
                  if (isBlock) {
                    return <code className={`text-xs text-ink ${className}`}>{children}</code>;
                  }
                  return (
                    <code className="text-xs bg-card-inset border border-edge rounded px-1.5 py-0.5 text-sage">
                      {children}
                    </code>
                  );
                },
                pre: ({ children }) => (
                  <pre className="text-xs bg-card-inset border border-edge rounded-lg p-4 overflow-x-auto mb-4">
                    {children}
                  </pre>
                ),
                a: ({ children, href }) => {
                  const isAnchor = href?.startsWith("#");
                  return (
                    <a
                      href={href}
                      target={isAnchor ? undefined : "_blank"}
                      rel={isAnchor ? undefined : "noreferrer"}
                      className="text-sage hover:underline cursor-pointer"
                      onClick={(e) => handleLinkClick(e, href)}
                    >
                      {children}
                    </a>
                  );
                },
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
