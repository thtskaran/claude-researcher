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
  const [timeLimit, setTimeLimit] = useState(30);
  const [autonomous, setAutonomous] = useState(true); // Default to autonomous
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Clarification flow state
  const [showClarification, setShowClarification] = useState(false);
  const [questions, setQuestions] = useState<ClarificationQuestion[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!goal.trim()) {
      setError("Please enter a research goal");
      return;
    }

    // If not autonomous, show clarification questions first
    if (!autonomous && !showClarification) {
      await getClarificationQuestions();
      return;
    }

    // If we have clarification questions, enrich the goal first
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
      setQuestions(data.questions || []);
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
      // Enrich goal with user answers
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

      // Start research with enriched goal
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
          time_limit: timeLimit,
          autonomous: true, // Always autonomous after enrichment
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

  const estimatedCost = timeLimit <= 15 ? "$0.50 - $1.50" : timeLimit <= 45 ? "$2.00 - $4.00" : "$5.00 - $10.00";

  return (
    <div className="bg-card-dark rounded-2xl shadow-xl border border-dark-border overflow-hidden relative">
      {/* Top accent */}
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-primary/60 via-primary to-primary/60" />

      <form onSubmit={handleSubmit} className="p-8 md:p-10 space-y-8">
        {/* Header */}
        <div className="text-center space-y-2">
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">
            {showClarification ? "Clarify Your Research" : "New Research Session"}
          </h2>
          <p className="text-sm text-gray-400">
            {showClarification
              ? "Answer these questions to refine your research goal"
              : "Configure your parameters to start a new analysis task"}
          </p>
        </div>

        {/* Close button */}
        <button
          type="button"
          onClick={onClose}
          className="absolute top-6 right-6 text-gray-500 hover:text-white transition-colors"
        >
          <span className="material-symbols-outlined">close</span>
        </button>

        {!showClarification ? (
          <>
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

            {/* Clarification Toggle */}
            <div className="flex items-center justify-between py-2 border border-dark-border rounded-lg px-4 bg-dark-surface/30">
              <div className="flex flex-col">
                <span className="text-sm font-medium text-white">Enable Clarification Questions</span>
                <span className="text-xs text-gray-400">AI will ask questions to refine your research goal</span>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  className="sr-only peer"
                  checked={!autonomous}
                  onChange={(e) => setAutonomous(!e.target.checked)}
                />
                <div className="w-11 h-6 bg-dark-border peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary/20 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary" />
              </label>
            </div>
          </>
        ) : (
          <>
            {/* Clarification Questions */}
            <div className="space-y-6">
              {questions.map((q, idx) => (
                <div key={idx} className="space-y-3">
                  <label className="block text-sm font-medium text-white">
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
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              ← Back to edit goal
            </button>
          </>
        )}

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
            disabled={loading || !goal.trim() || (showClarification && Object.keys(answers).length === 0)}
            className="w-full py-4 px-6 bg-primary hover:bg-primary-dark text-white font-semibold rounded-xl shadow-[0_0_25px_rgba(43,124,238,0.35)] hover:shadow-[0_0_35px_rgba(43,124,238,0.5)] transition-all duration-300 transform active:scale-[0.98] flex items-center justify-center gap-2 group/btn disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <span className="material-symbols-outlined animate-spin">progress_activity</span>
                {showClarification ? "Starting Research..." : "Generating Questions..."}
              </>
            ) : (
              <>
                <span className="material-symbols-outlined group-hover/btn:animate-pulse">
                  {showClarification ? "rocket_launch" : "psychology"}
                </span>
                {showClarification ? "Start Research" : (autonomous ? "Start Research" : "Get Clarification Questions")}
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
