"use client";

import { useState, useEffect, useRef } from "react";
import { ResearchWebSocket, AgentEvent } from "@/lib/websocket";

interface ActivityFeedProps {
  sessionId: string;
}

export default function ActivityFeed({ sessionId }: ActivityFeedProps) {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<ResearchWebSocket | null>(null);

  useEffect(() => {
    // Create WebSocket connection
    const ws = new ResearchWebSocket(sessionId);

    // Subscribe to events
    ws.onEvent((event) => {
      console.log("Received event:", event);
      setEvents((prev) => [event, ...prev].slice(0, 100)); // Keep last 100 events
    });

    // Connect
    ws.connect();
    wsRef.current = ws;

    // Check connection status
    const checkConnection = setInterval(() => {
      setConnected(ws.isConnected());
    }, 1000);

    // Cleanup
    return () => {
      clearInterval(checkConnection);
      ws.disconnect();
    };
  }, [sessionId]);

  return (
    <div className="card">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Live Activity Feed</h3>
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              connected ? "bg-success" : "bg-gray-500"
            } ${connected ? "animate-pulse" : ""}`}
          />
          <span className="text-sm text-gray-400">
            {connected ? "Connected" : "Disconnected"}
          </span>
        </div>
      </div>

      {/* Events List */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {events.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <svg
              className="w-12 h-12 mx-auto mb-3 opacity-50"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
            <p className="text-sm">Waiting for events...</p>
            <p className="text-xs mt-1">
              Events will appear here when agents start working
            </p>
          </div>
        ) : (
          events.map((event) => (
            <div
              key={`${event.timestamp}-${event.event_type}-${event.agent}-${Math.random()}`}
              className="bg-dark-bg border border-dark-border rounded-lg p-3 hover:border-primary/30 transition-colors"
            >
              {/* Event Header */}
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className={`badge ${getBadgeClass(event.event_type)}`}>
                    {event.event_type}
                  </span>
                  <span className="text-xs text-gray-500">
                    {event.agent}
                  </span>
                </div>
                <span className="text-xs text-gray-500">
                  {formatTimestamp(event.timestamp)}
                </span>
              </div>

              {/* Event Data */}
              <div className="text-sm text-gray-300">
                {renderEventData(event.data)}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function getBadgeClass(eventType: string): string {
  switch (eventType) {
    case "thinking":
      return "badge-thinking";
    case "action":
      return "badge-action";
    case "finding":
      return "badge-finding";
    case "error":
      return "badge-error";
    case "system":
      return "bg-gray-500/20 text-gray-400";
    default:
      return "badge-thinking";
  }
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function renderEventData(data: Record<string, any>): React.ReactNode {
  // Handle different event data structures
  if (data.thought) {
    return <p className="italic">{data.thought}</p>;
  }

  if (data.action) {
    return <p className="font-medium">{data.action}</p>;
  }

  if (data.content) {
    return <p>{data.content}</p>;
  }

  if (data.message) {
    return <p>{data.message}</p>;
  }

  if (data.error) {
    return <p className="text-error">{data.error}</p>;
  }

  // Render as JSON for debugging (any other structure)
  return (
    <pre className="text-xs font-mono overflow-x-auto">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}
