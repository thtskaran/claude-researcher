/**
 * WebSocket client for real-time research events.
 *
 * Usage:
 *   const ws = new ResearchWebSocket(sessionId);
 *   ws.onEvent((event) => console.log(event));
 *   ws.connect();
 */

export interface AgentEvent {
  session_id: string;
  event_type: string;
  agent: string;
  timestamp: string;
  data: Record<string, any>;
}

export type EventCallback = (event: AgentEvent) => void;

export class ResearchWebSocket {
  private ws: WebSocket | null = null;
  private sessionId: string;
  private callbacks: Set<EventCallback> = new Set();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private pingInterval: NodeJS.Timeout | null = null;

  constructor(sessionId: string) {
    this.sessionId = sessionId;
  }

  connect(): void {
    const url = `ws://localhost:8080/ws/${this.sessionId}`;
    console.log(`[WebSocket] Connecting to ${url}...`);

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log(`[WebSocket] Connected to session ${this.sessionId}`);
        this.reconnectAttempts = 0;

        // Start ping/pong to keep connection alive
        this.startPing();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Handle system messages
          if (data.type === "connected") {
            console.log(`[WebSocket] Connection confirmed:`, data);
            return;
          }

          if (data.type === "pong") {
            // Keepalive response
            return;
          }

          // Handle agent events
          if (data.event_type) {
            this.notifyCallbacks(data as AgentEvent);
          }
        } catch (error) {
          console.error("[WebSocket] Failed to parse message:", error);
        }
      };

      this.ws.onerror = (event) => {
        // Avoid noisy empty error objects in Next.js overlay
        const detail = (event as any)?.message;
        if (detail) {
          console.warn("[WebSocket] Error:", detail);
        } else {
          console.warn("[WebSocket] Error event");
        }
      };

      this.ws.onclose = () => {
        console.log("[WebSocket] Connection closed");
        this.stopPing();

        // Attempt reconnection
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          const delay = this.reconnectDelay * this.reconnectAttempts;
          console.log(
            `[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`
          );
          setTimeout(() => this.connect(), delay);
        } else {
          console.error("[WebSocket] Max reconnection attempts reached");
        }
      };
    } catch (error) {
      console.error("[WebSocket] Failed to create connection:", error);
    }
  }

  disconnect(): void {
    console.log("[WebSocket] Disconnecting...");
    this.stopPing();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  onEvent(callback: EventCallback): () => void {
    this.callbacks.add(callback);

    // Return unsubscribe function
    return () => {
      this.callbacks.delete(callback);
    };
  }

  private notifyCallbacks(event: AgentEvent): void {
    this.callbacks.forEach((callback) => {
      try {
        callback(event);
      } catch (error) {
        console.error("[WebSocket] Callback error:", error);
      }
    });
  }

  private startPing(): void {
    // Send ping every 30 seconds
    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send("ping");
      }
    }, 30000);
  }

  private stopPing(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
