"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

interface NewSessionFormProps {
  onClose: () => void;
  onSuccess: () => void;
}

interface ClarificationQuestion {
  question: string;
  options: string[];
  allow_multiple: boolean;
}

export default function NewSessionForm({ onClose, onSuccess }: NewSessionFormProps) {
  const router = useRouter();
  const [goal, setGoal] = useState("");
  const [maxIterations, setMaxIterations] = useState(5);
  const [enableClarification, setEnableClarification] = useState(true);
  const [enableMidQuestions, setEnableMidQuestions] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [showClarification, setShowClarification] = useState(false);
  const [questions, setQuestions] = useState<ClarificationQuestion[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!goal.trim()) {
      setError("Please enter a research goal");
      return;
    }

    if (enableClarification && !showClarification) {
      await getClarificationQuestions();
      return;
    }

    if (showClarification && Object.keys(answers).length > 0) {
      await startWithEnrichedGoal();
    } else {
      await startResearch(goal);
    }
  };

  const getClarificationQuestions = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch("/api/research/clarify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal: goal.trim(), max_questions: 4 }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate questions");
      }

      const data = await response.json();
      const qs = data.questions || [];
      if (qs.length === 0) {
        // No questions generated — start research directly
        await startResearch(goal.trim());
        return;
      }
      setQuestions(qs);
      setShowClarification(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to get clarification questions");
    } finally {
      setLoading(false);
    }
  };

  const startWithEnrichedGoal = async () => {
    setLoading(true);
    setError("");

    try {
      const enrichResponse = await fetch("/api/research/enrich", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          goal: goal.trim(),
          questions: questions,
          answers: answers,
        }),
      });

      if (!enrichResponse.ok) {
        throw new Error("Failed to enrich goal");
      }

      const enrichData = await enrichResponse.json();
      const enrichedGoal = enrichData.enriched_goal;

      await startResearch(enrichedGoal);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start research");
      setLoading(false);
    }
  };

  const startResearch = async (researchGoal: string) => {
    try {
      const response = await fetch("/api/research/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          goal: researchGoal,
          max_iterations: maxIterations,
          autonomous: !enableClarification && !enableMidQuestions,
          enable_mid_questions: enableMidQuestions,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to start research");
      }

      const data = await response.json();
      onSuccess();
      router.push(`/session/${data.session_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start research");
    } finally {
      setLoading(false);
    }
  };

  const depthLabel = maxIterations <= 3 ? "Quick scan" : maxIterations <= 7 ? "Standard depth" : maxIterations <= 14 ? "Deep research" : "Exhaustive";

  return (
    <div className="bg-card rounded-2xl border border-edge overflow-hidden relative" style={{ boxShadow: "var(--shadow-lg)" }}>
      {/* Top accent */}
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-sage/40 via-sage to-sage/40" />

      <form onSubmit={handleSubmit} className="p-8 md:p-10 space-y-8">
        {/* Header */}
        <div className="text-center space-y-2">
          <h2 className="text-2xl md:text-3xl font-display tracking-tight">
            {showClarification ? "Clarify Your Research" : "New Research Session"}
          </h2>
          <p className="text-sm text-ink-secondary">
            {showClarification
              ? "Answer these questions to refine your research goal"
              : "Configure your parameters to start a new analysis task"}
          </p>
        </div>

        {/* Close button */}
        <button
          type="button"
          onClick={onClose}
          className="absolute top-6 right-6 text-ink-muted hover:text-ink transition-colors"
        >
          <span className="material-symbols-outlined">close</span>
        </button>

        {!showClarification ? (
          <>
            {/* Goal Input */}
            <div className="space-y-3">
              <label className="block text-xs font-medium text-ink-secondary uppercase tracking-wider">
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
                  <span className="material-symbols-outlined text-ink-muted text-lg">edit_note</span>
                </div>
              </div>
            </div>

            {/* Iterations Slider */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium uppercase tracking-wider text-ink-secondary">
                  Research Depth (Iterations)
                </label>
                <span className="text-sage font-mono text-sm bg-sage-soft px-2 py-0.5 rounded">
                  {maxIterations} {maxIterations === 1 ? "iteration" : "iterations"}
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="20"
                step="1"
                value={maxIterations}
                onChange={(e) => setMaxIterations(parseInt(e.target.value))}
                className="w-full h-1.5 bg-edge rounded-full appearance-none cursor-pointer accent-sage"
              />
              <div className="flex justify-between text-xs text-ink-muted font-mono">
                <span>1 (quick)</span>
                <span>20 (deep)</span>
              </div>
            </div>

            {/* Clarification Toggle */}
            <div className="flex items-center justify-between py-2 border border-edge rounded-lg px-4 bg-card-hover/30">
              <div className="flex flex-col">
                <span className="text-sm font-medium text-ink">Enable Clarification Questions</span>
                <span className="text-xs text-ink-secondary">AI will ask questions to refine your research goal</span>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  className="sr-only peer"
                  checked={enableClarification}
                  onChange={(e) => setEnableClarification(e.target.checked)}
                />
                <div className="w-11 h-6 bg-edge peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-sage/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-sage" />
              </label>
            </div>

            {/* Mid-Research Questions Toggle */}
            <div className="flex items-center justify-between py-2 border border-edge rounded-lg px-4 bg-card-hover/30">
              <div className="flex flex-col">
                <span className="text-sm font-medium text-ink">Enable Mid-Research Questions</span>
                <span className="text-xs text-ink-secondary">AI can ask questions during research to guide the process</span>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  className="sr-only peer"
                  checked={enableMidQuestions}
                  onChange={(e) => setEnableMidQuestions(e.target.checked)}
                />
                <div className="w-11 h-6 bg-edge peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-sage/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-sage" />
              </label>
            </div>
          </>
        ) : (
          <>
            {/* Clarification Questions */}
            <div className="space-y-6">
              {questions.map((q, idx) => (
                <div key={idx} className="space-y-3">
                  <label className="block text-sm font-medium text-ink">
                    {idx + 1}. {q.question}
                  </label>
                  <input
                    type="text"
                    value={answers[idx.toString()] || ""}
                    onChange={(e) => setAnswers({ ...answers, [idx.toString()]: e.target.value })}
                    className="input w-full"
                    placeholder="Your answer..."
                  />
                </div>
              ))}
            </div>

            <button
              type="button"
              onClick={() => {
                setShowClarification(false);
                setQuestions([]);
                setAnswers({});
              }}
              className="text-sm text-ink-secondary hover:text-ink transition-colors"
            >
              &larr; Back to edit goal
            </button>
          </>
        )}

        {/* Error */}
        {error && (
          <div className="text-sm text-coral bg-coral-soft border border-coral/20 rounded-lg px-4 py-3">
            {error}
          </div>
        )}

        {/* Action Footer */}
        <div className="pt-2 flex flex-col items-center gap-6">
          <div className="font-mono text-sm text-ink-secondary flex items-center gap-2">
            <span className="material-symbols-outlined text-base">speed</span>
            Depth: <span className="text-ink font-bold">{depthLabel}</span>
          </div>
          <button
            type="submit"
            disabled={loading || !goal.trim() || (showClarification && Object.keys(answers).length === 0)}
            className="w-full py-4 px-6 bg-sage hover:bg-sage-hover text-white font-semibold rounded-xl transition-all duration-300 transform active:scale-[0.98] flex items-center justify-center gap-2 group/btn disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ boxShadow: "0 2px 12px rgb(var(--sage) / 0.3)" }}
          >
            {loading ? (
              <>
                <span className="material-symbols-outlined animate-spin">progress_activity</span>
                {showClarification ? "Starting Research..." : (enableClarification ? "Generating Questions..." : "Starting Research...")}
              </>
            ) : (
              <>
                <span className="material-symbols-outlined group-hover/btn:animate-soft-pulse">
                  {showClarification ? "rocket_launch" : (enableClarification ? "psychology" : "rocket_launch")}
                </span>
                {showClarification ? "Start Research" : (enableClarification ? "Get Clarification Questions" : "Start Research")}
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
