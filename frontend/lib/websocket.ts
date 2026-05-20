import { Client, IMessage } from "@stomp/stompjs";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8080/ws";

export type StreamChunk = {
  type: "token" | "step" | "done" | "error";
  step?: string;
  content: string;
};

export function createSessionClient(
  sessionId: string,
  onChunk: (chunk: StreamChunk) => void,
  onConnected?: () => void
): Client {
  const client = new Client({
    brokerURL: WS_URL,
    reconnectDelay: 5000,
    onConnect: () => {
      onConnected?.();
      client.subscribe(`/topic/session/${sessionId}/stream`, (msg: IMessage) => {
        try {
          const chunk: StreamChunk = JSON.parse(msg.body);
          onChunk(chunk);
        } catch {
          onChunk({ type: "token", content: msg.body });
        }
      });
    },
    onStompError: (frame) => {
      console.error("STOMP error", frame);
      onChunk({ type: "error", content: `WebSocket STOMP error: ${frame.headers?.message ?? "unknown"}` });
    },
    onWebSocketError: (event) => {
      console.error("WebSocket error", event);
      onChunk({ type: "error", content: `WebSocket failed to connect. Check that NEXT_PUBLIC_WS_URL is set correctly (currently: ${WS_URL})` });
    },
    onDisconnect: () => {
      console.warn("STOMP disconnected from", WS_URL);
    },
  });

  client.activate();
  return client;
}
