import { Client, IMessage } from "@stomp/stompjs";

const WS_URL = process.env.NEXT_PUBLIC_JAVA_WS_URL ?? "ws://localhost:8080/ws";

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
    reconnectDelay: 3000,
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
      onChunk({ type: "error", content: "WebSocket connection failed." });
    },
  });

  client.activate();
  return client;
}
