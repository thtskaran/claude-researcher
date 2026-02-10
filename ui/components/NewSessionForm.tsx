"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

interface NewSessionFormProps {
  onClose: () => void;
  onSuccess: () => void;
}

export default function NewSessionForm({ onClose, onSuccess }: NewSessionFormProps) {
  const router = useRouter();
  const [goal, setGoal] = useState("");
  const [timeLimit, setTimeLimit] = useState(30);
  const [depth, setDepth] = useState<"quick" | "standard" | "deep">("standard");
  const [autonomous, setAutonomous] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [sourceType, setSourceType] = useState("all");
  const [outputFormat, setOutputFormat] = useState("summary");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!goal.trim()) {
      setError("Please enter a research goal");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/sessions/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          goal: goal.trim(),
          time_limit: timeLimit,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to create session");
      }

      const data = await response.json();
      onSuccess();
      router.push(`/session/${data.session_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const estimatedCost = depth === "quick" ? "$0.50 - $1.50" : depth === "standard" ? "$2.00 - $4.00" : "$5.00 - $10.00";

  return (
    <div className="bg-card-dark rounded-2xl shadow-xl border border-dark-border overflow-hidden relative">
      {/* Top accent */}
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-primary/60 via-primary to-primary/60" />

      <form onSubmit={handleSubmit} className="p-8 md:p-10 space-y-8">
        {/* Header */}
        <div className="text-center space-y-2">
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">New Research Session</h2>
          <p className="text-sm text-gray-400">Configure your parameters to start a new analysis task.</p>
        </div>

        {/* Close button */}
        <button
          type="button"
          onClick={onClose}
          className="absolute top-6 right-6 text-gray-500 hover:text-white transition-colors"
        >
          <span className="material-symbols-outlined">close</span>
        </button>

        {/* Goal Input */}
        <div className="space-y-3">
          <label className="block text-xs font-medium text-gray-400 uppercase tracking-wider">
            Research Goal
          </label>
          <div className="relative">
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              className="input w-full min-h-[140px] resize-none text-base p-4 rounded-xl"
              placeholder="e.g., Analyze the market trends for renewable energy in Southeast Asia, focusing on solar panel adoption rates over the last 5 years..."
            />
            <div className="absolute bottom-3 right-3 pointer-events-none">
              <span className="material-symbols-outlined text-gray-600 text-lg">edit_note</span>
            </div>
          </div>
        </div>

        {/* Settings Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Time Limit Slider */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium uppercase tracking-wider text-gray-400">
                Time Limit
              </label>
              <span className="text-primary font-mono text-sm bg-primary/10 px-2 py-0.5 rounded">
                {timeLimit} mins
              </span>
            </div>
            <input
              type="range"
              min="5"
              max="120"
              step="5"
              value={timeLimit}
              onChange={(e) => setTimeLimit(parseInt(e.target.value))}
              className="w-full h-1.5 bg-dark-border rounded-full appearance-none cursor-pointer accent-primary"
            />
            <div className="flex justify-between text-xs text-gray-500 font-mono">
              <span>5m</span>
              <span>120m</span>
            </div>
          </div>

          {/* Research Depth */}
          <div className="space-y-4">
            <label className="block text-xs font-medium uppercase tracking-wider text-gray-400">
              Research Depth
            </label>
            <div className="segmented-control">
              {(["quick", "standard", "deep"] as const).map((d) => (
                <label key={d}>
                  <input
                    type="radio"
                    name="depth"
                    value={d}
                    checked={depth === d}
                    onChange={() => setDepth(d)}
                  />
                  <div className="segment capitalize">{d}</div>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Autonomous Mode Toggle */}
        <div className="flex items-center justify-between py-2">
          <div className="flex flex-col">
            <span className="text-sm font-medium text-white">Autonomous Mode</span>
            <span className="text-xs text-gray-400">Allow AI to follow new leads automatically</span>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              className="sr-only peer"
              checked={autonomous}
              onChange={(e) => setAutonomous(e.target.checked)}
            />
            <div className="w-11 h-6 bg-dark-border peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary" />
          </label>
        </div>

        {/* Advanced Options */}
        <div className="border-t border-dark-border pt-4">
          <details className="group" open={showAdvanced} onToggle={(e) => setShowAdvanced((e.target as HTMLDetailsElement).open)}>
            <summary className="flex items-center justify-between cursor-pointer list-none text-sm font-medium text-gray-500 hover:text-white transition-colors">
              <span>Advanced Options</span>
              <span className="transition group-open:rotate-180">
                <span className="material-symbols-outlined text-lg">expand_more</span>
              </span>
            </summary>
            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-2">
                <label className="text-xs uppercase tracking-wider text-gray-500">Source Types</label>
                <select
                  value={sourceType}
                  onChange={(e) => setSourceType(e.target.value)}
                  className="input text-sm"
                >
                  <option value="all">All Sources</option>
                  <option value="academic">Academic Only</option>
                  <option value="news">News Only</option>
                </select>
              </div>
              <div className="flex flex-col gap-2">
                <label className="text-xs uppercase tracking-wider text-gray-500">Output Format</label>
                <select
                  value={outputFormat}
                  onChange={(e) => setOutputFormat(e.target.value)}
                  className="input text-sm"
                >
                  <option value="summary">Summary Report</option>
                  <option value="raw">Raw Data</option>
                  <option value="bullets">Bullet Points</option>
                </select>
              </div>
            </div>
          </details>
        </div>

        {/* Error */}
        {error && (
          <div className="text-sm text-error bg-error/10 border border-error/20 rounded-lg px-4 py-3">
            {error}
          </div>
        )}

        {/* Action Footer */}
        <div className="pt-2 flex flex-col items-center gap-6">
          <div className="font-mono text-sm text-gray-400 flex items-center gap-2">
            <span className="material-symbols-outlined text-base">payments</span>
            Estimated Cost: <span className="text-white font-bold">{estimatedCost}</span>
          </div>
          <button
            type="submit"
            disabled={loading || !goal.trim()}
            className="w-full py-4 px-6 bg-primary hover:bg-primary-dark text-white font-semibold rounded-xl shadow-[0_0_25px_rgba(43,124,238,0.35)] hover:shadow-[0_0_35px_rgba(43,124,238,0.5)] transition-all duration-300 transform active:scale-[0.98] flex items-center justify-center gap-2 group/btn disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <span className="material-symbols-outlined animate-spin">progress_activity</span>
                Creating Session...
              </>
            ) : (
              <>
                <span className="material-symbols-outlined group-hover/btn:animate-pulse">rocket_launch</span>
                Start Research
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
