"use client";

import { useState, useEffect } from "react";

interface QuestionModalProps {
  sessionId: string;
  questionId: string;
  question: string;
  context: string;
  options: string[];
  timeout: number;
  onSubmit: (response: string) => void;
  onTimeout: () => void;
}

export default function QuestionModal({
  sessionId,
  questionId,
  question,
  context,
  options,
  timeout,
  onSubmit,
  onTimeout,
}: QuestionModalProps) {
  const [response, setResponse] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(timeout);

  useEffect(() => {
    const interval = setInterval(() => {
      setTimeRemaining((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          onTimeout();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [timeout, onTimeout]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!response.trim() || submitting) return;

    setSubmitting(true);
    try {
      const res = await fetch(`/api/research/${sessionId}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question_id: questionId,
          response: response.trim(),
        }),
      });

      if (res.ok) {
        onSubmit(response.trim());
      } else {
        console.error("Failed to submit answer");
      }
    } catch (err) {
      console.error("Error submitting answer:", err);
    } finally {
      setSubmitting(false);
    }
  };

  const handleOptionClick = (option: string) => {
    setResponse(option);
  };

  const progress = (timeRemaining / timeout) * 100;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm animate-fadeIn">
      <div className="bg-dark-surface border border-primary/30 rounded-2xl shadow-2xl max-w-2xl w-full mx-4 overflow-hidden animate-slideUp">
        {/* Header */}
        <div className="bg-gradient-to-r from-primary/20 via-primary/10 to-primary/20 border-b border-primary/20 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center">
                <span className="material-symbols-outlined text-primary">help</span>
              </div>
              <div>
                <h2 className="text-lg font-bold">Mid-Research Question</h2>
                <p className="text-xs text-gray-400">The research agent needs your input</p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-400 mb-1">Time Remaining</div>
              <div className="text-lg font-mono font-bold text-primary">{timeRemaining}s</div>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mt-3 h-1 bg-dark-border rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-1000 ease-linear"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Content */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Context */}
          {context && (
            <div className="bg-dark-bg/50 border border-dark-border rounded-lg p-4">
              <div className="flex items-start gap-2">
                <span className="material-symbols-outlined text-sm text-gray-400 mt-0.5">info</span>
                <div className="text-sm text-gray-300">{context}</div>
              </div>
            </div>
          )}

          {/* Question */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-3">Question:</label>
            <p className="text-base text-white leading-relaxed">{question}</p>
          </div>

          {/* Options */}
          {options && options.length > 0 && (
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-400 mb-2">Suggested Options:</label>
              <div className="grid grid-cols-1 gap-2">
                {options.map((option, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleOptionClick(option)}
                    className={`text-left px-4 py-3 rounded-lg border transition-all ${
                      response === option
                        ? "bg-primary/20 border-primary text-white"
                        : "bg-dark-bg border-dark-border text-gray-300 hover:border-primary/50"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span className={`w-4 h-4 rounded-full border-2 flex-shrink-0 ${
                        response === option ? "border-primary bg-primary" : "border-gray-500"
                      }`} />
                      <span className="text-sm">{option}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Text Input */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              {options && options.length > 0 ? "Or provide your own answer:" : "Your Answer:"}
            </label>
            <textarea
              value={response}
              onChange={(e) => setResponse(e.target.value)}
              className="input w-full min-h-[100px] resize-none"
              placeholder="Type your response here..."
              disabled={submitting}
            />
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between pt-2">
            <div className="text-xs text-gray-500">
              <span className="material-symbols-outlined text-sm align-middle mr-1">schedule</span>
              Research will continue autonomously if no answer is provided
            </div>
            <button
              type="submit"
              disabled={!response.trim() || submitting}
              className="px-6 py-2.5 bg-primary hover:bg-primary-dark text-white font-semibold rounded-lg shadow-lg shadow-primary/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {submitting ? (
                <>
                  <span className="material-symbols-outlined animate-spin text-lg">progress_activity</span>
                  Submitting...
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined text-lg">send</span>
                  Submit Answer
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
