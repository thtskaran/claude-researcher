"use client";

import { useState } from "react";
import ActivityFeed from "@/components/ActivityFeed";
import Link from "next/link";

export default function TestWebSocket() {
  const [sessionId, setSessionId] = useState("test-session");
  const [eventType, setEventType] = useState("thinking");
  const [message, setMessage] = useState("Test event message");
  const [sending, setSending] = useState(false);

  const sendTestEvent = async () => {
    setSending(true);
    try {
      const response = await fetch(
        `http://localhost:8080/api/test/emit/${sessionId}?event_type=${eventType}&message=${encodeURIComponent(
          message
        )}`,
        { method: "POST" }
      );

      if (!response.ok) {
        throw new Error("Failed to send event");
      }

      const data = await response.json();
      console.log("Event sent:", data);
    } catch (error) {
      console.error("Error sending event:", error);
      alert("Failed to send event");
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="min-h-screen bg-dark-bg p-8">
      {/* Header */}
      <div className="max-w-6xl mx-auto mb-8">
        <Link href="/" className="text-primary hover:underline mb-4 inline-block">
          ← Back to Dashboard
        </Link>
        <h1 className="text-3xl font-bold mb-2">WebSocket Test Page</h1>
        <p className="text-gray-400">
          Test real-time event streaming from API to UI
        </p>
      </div>

      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Control Panel */}
        <div className="card">
          <h2 className="text-xl font-bold mb-4">Send Test Events</h2>

          <div className="space-y-4">
            {/* Session ID */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Session ID
              </label>
              <input
                type="text"
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                className="input w-full"
                placeholder="test-session"
              />
            </div>

            {/* Event Type */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Event Type
              </label>
              <select
                value={eventType}
                onChange={(e) => setEventType(e.target.value)}
                className="input w-full"
              >
                <option value="thinking">Thinking</option>
                <option value="action">Action</option>
                <option value="finding">Finding</option>
                <option value="synthesis">Synthesis</option>
                <option value="error">Error</option>
                <option value="system">System</option>
              </select>
            </div>

            {/* Message */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Message
              </label>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                className="input w-full min-h-24 resize-none"
                placeholder="Event message..."
              />
            </div>

            {/* Send Button */}
            <button
              onClick={sendTestEvent}
              disabled={sending}
              className="btn btn-primary w-full flex items-center justify-center gap-2"
            >
              {sending ? (
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
                  <span>Sending...</span>
                </>
              ) : (
                <>
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
                      d="M13 10V3L4 14h7v7l9-11h-7z"
                    />
                  </svg>
                  <span>Send Event</span>
                </>
              )}
            </button>

            {/* Quick Actions */}
            <div className="pt-4 border-t border-dark-border">
              <p className="text-sm text-gray-400 mb-3">Quick Actions:</p>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => {
                    setEventType("thinking");
                    setMessage("Analyzing research goal...");
                    setTimeout(sendTestEvent, 100);
                  }}
                  className="btn btn-secondary text-sm"
                >
                  Thinking Event
                </button>
                <button
                  onClick={() => {
                    setEventType("action");
                    setMessage("Searching for: quantum computing papers");
                    setTimeout(sendTestEvent, 100);
                  }}
                  className="btn btn-secondary text-sm"
                >
                  Action Event
                </button>
                <button
                  onClick={() => {
                    setEventType("finding");
                    setMessage("Found 3 relevant sources");
                    setTimeout(sendTestEvent, 100);
                  }}
                  className="btn btn-secondary text-sm"
                >
                  Finding Event
                </button>
                <button
                  onClick={() => {
                    setEventType("error");
                    setMessage("API rate limit exceeded");
                    setTimeout(sendTestEvent, 100);
                  }}
                  className="btn btn-secondary text-sm"
                >
                  Error Event
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Activity Feed */}
        <ActivityFeed sessionId={sessionId} />
      </div>

      {/* Instructions */}
      <div className="max-w-6xl mx-auto mt-8 card">
        <h3 className="text-lg font-semibold mb-3">How to Test</h3>
        <ol className="space-y-2 text-sm text-gray-300">
          <li className="flex gap-2">
            <span className="text-primary font-bold">1.</span>
            <span>
              Keep the <code className="text-primary">Session ID</code> as "test-session" or change it to match a real session
            </span>
          </li>
          <li className="flex gap-2">
            <span className="text-primary font-bold">2.</span>
            <span>
              Select an event type and enter a message
            </span>
          </li>
          <li className="flex gap-2">
            <span className="text-primary font-bold">3.</span>
            <span>
              Click "Send Event" and watch it appear in the Activity Feed in real-time
            </span>
          </li>
          <li className="flex gap-2">
            <span className="text-primary font-bold">4.</span>
            <span>
              Or use the Quick Actions buttons to send predefined events
            </span>
          </li>
          <li className="flex gap-2">
            <span className="text-primary font-bold">5.</span>
            <span>
              Open this page in multiple tabs to see events broadcast to all connected clients
            </span>
          </li>
        </ol>

        <div className="mt-4 p-3 bg-info/10 border border-info/30 rounded-lg">
          <p className="text-sm text-info">
            <strong>💡 Tip:</strong> Open the browser console (F12) to see WebSocket connection logs and event data
          </p>
        </div>
      </div>
    </div>
  );
}
