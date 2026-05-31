package com.data2model.model;

public record StreamChunk(
    Type type,
    String step,
    String content
) {
    public enum Type { token, step, done, error }

    public static StreamChunk stepStart(String step) {
        return new StreamChunk(Type.step, step, "Starting " + step.toLowerCase() + "…");
    }

    public static StreamChunk token(String content) {
        return new StreamChunk(Type.token, null, content);
    }

    public static StreamChunk done() {
        return new StreamChunk(Type.done, null, "");
    }

    public static StreamChunk error(String message) {
        return new StreamChunk(Type.error, null, message);
    }
}
