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
  const [timeLimit, setTimeLimit] = useState(60);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showClarify, setShowClarify] = useState(false);
  const [clarifyLoading, setClarifyLoading] = useState(false);
  const [clarifyError, setClarifyError] = useState("");
  const [questions, setQuestions] = useState<Array<{ id: number; question: string; options?: string[] }>>([]);
  const [answers, setAnswers] = useState<Record<number, string>>({});

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!goal.trim()) {
      setError("Please enter a research goal");
      return;
    }

    if (!showClarify) {
      await loadClarifications();
      return;
    }

    await startResearch(false);
  };

  const loadClarifications = async () => {
    setClarifyLoading(true);
    setClarifyError("");
    try {
      const response = await fetch("/api/research/clarify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal: goal.trim(), max_questions: 4 }),
      });
      if (!response.ok) {
        throw new Error("Failed to load clarification questions");
      }
      const data = await response.json();
      const list = Array.isArray(data?.questions) ? data.questions : [];
      if (list.length === 0) {
        await startResearch(true);
        return;
      }
      setQuestions(list);
      setShowClarify(true);
    } catch (err) {
      setClarifyError("Unable to load questions. You can still start now.");
      setShowClarify(true);
    } finally {
      setClarifyLoading(false);
    }
  };

  const startResearch = async (skipClarify: boolean) => {
    setLoading(true);
    setError("");

    let enrichedGoal = goal.trim();
    if (!skipClarify && questions.length > 0) {
      try {
        const response = await fetch("/api/research/enrich", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            goal: goal.trim(),
            questions,
            answers,
          }),
        });
        if (response.ok) {
          const data = await response.json();
          if (typeof data?.enriched_goal === "string") {
            enrichedGoal = data.enriched_goal;
          }
        }
      } catch {
        // Fallback to original goal if enrichment fails
      }
    }

    try {
      const response = await fetch("/api/research/start", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          goal: enrichedGoal,
          time_limit: timeLimit,
          autonomous: true,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to start research");
      }

      const data = await response.json();
      console.log("Research started:", data);

      // Navigate to session detail page to watch live
      router.push(`/session/${data.session_id}`);
    } catch (err) {
      console.error("Error creating session:", err);
      setError("Failed to create session. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="card max-w-2xl w-full animate-in fade-in duration-200">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold">New Research Session</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200 transition-colors"
            aria-label="Close"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Goal Input */}
          <div>
            <label htmlFor="goal" className="block text-sm font-medium mb-2">
              Research Goal or Question
            </label>
            <textarea
              id="goal"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="What would you like to research? Be as specific as possible..."
              className="input w-full min-h-32 resize-none"
              disabled={loading}
            />
            <p className="text-xs text-gray-500 mt-2">
              💡 Tip: Specific questions get better results. Try "Latest developments in fusion energy" instead of "fusion"
            </p>
            <div className="mt-3 p-3 bg-primary/10 border border-primary/30 rounded-lg">
              <p className="text-xs text-primary">
                <strong>ℹ️ Note:</strong> You can add optional clarifications before starting.
                After you click "Continue", choose "Start Research" to begin.
              </p>
            </div>
          </div>

          {/* Time Limit Slider */}
          <div>
            <label htmlFor="timeLimit" className="block text-sm font-medium mb-2">
              Time Limit: <span className="text-primary">{timeLimit} minutes</span>
            </label>
            <input
              id="timeLimit"
              type="range"
              min="5"
              max="240"
              step="5"
              value={timeLimit}
              onChange={(e) => setTimeLimit(parseInt(e.target.value))}
              className="w-full h-2 bg-dark-border rounded-lg appearance-none cursor-pointer accent-primary"
              disabled={loading}
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>5 min</span>
              <span>60 min</span>
              <span>120 min</span>
              <span>240 min</span>
            </div>
          </div>

          {showClarify ? (
            <div className="space-y-4 border-t border-dark-border pt-4 max-h-80 overflow-y-auto pr-1">
              <div className="text-sm text-gray-300">
                Quick clarifications (optional) to focus the research. Leave any answer blank to skip.
              </div>
              {clarifyLoading ? (
                <div className="text-xs text-gray-400">Loading questions…</div>
              ) : null}
              {clarifyError ? (
                <div className="text-xs text-error">{clarifyError}</div>
              ) : null}
              {questions.map((q) => (
                <div key={q.id} className="space-y-2">
                  <label className="block text-xs font-medium mb-1 text-gray-400">
                    {q.question}
                  </label>
                  {q.options && q.options.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {q.options.map((option) => (
                        <button
                          key={option}
                          type="button"
                          className={`chip ${answers[q.id] === option ? "chip-active" : ""}`}
                          onClick={() =>
                            setAnswers((prev) => ({ ...prev, [q.id]: option }))
                          }
                        >
                          {option}
                        </button>
                      ))}
                      <button
                        type="button"
                        className={`chip ${answers[q.id] ? "" : "chip-active"}`}
                        onClick={() =>
                          setAnswers((prev) => ({ ...prev, [q.id]: "" }))
                        }
                      >
                        Custom
                      </button>
                    </div>
                  ) : null}
                  <input
                    value={answers[q.id] || ""}
                    onChange={(e) =>
                      setAnswers((prev) => ({ ...prev, [q.id]: e.target.value }))
                    }
                    placeholder="Type your answer (optional)"
                    className="input w-full"
                    disabled={loading}
                  />
                </div>
              ))}
            </div>
          ) : null}

          {/* Error Message */}
          {error && (
            <div className="bg-error/10 border border-error/30 rounded-lg p-3 text-error text-sm">
              {error}
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3 justify-end">
            <button
              type="button"
              onClick={onClose}
              className="btn btn-secondary"
              disabled={loading || clarifyLoading}
            >
              Cancel
            </button>
            {showClarify ? (
              <button
                type="button"
                onClick={() => startResearch(true)}
                className="btn btn-secondary"
                disabled={loading || clarifyLoading}
              >
                Skip & Start
              </button>
            ) : null}
            <button
              type="submit"
              className="btn btn-primary flex items-center gap-2"
              disabled={loading || clarifyLoading || !goal.trim()}
            >
              {loading ? (
                <>
                  <svg
                    className="animate-spin h-4 w-4"
                    viewBox="0 0 24 24"
                    fill="none"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  <span>Creating...</span>
                </>
              ) : clarifyLoading ? (
                <>
                  <svg
                    className="animate-spin h-4 w-4"
                    viewBox="0 0 24 24"
                    fill="none"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  <span>Generating questions...</span>
                </>
              ) : (
                <>
                  <span>{showClarify ? "Start Research" : "Continue"}</span>
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13 7l5 5m0 0l-5 5m5-5H6"
                    />
                  </svg>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
