package com.data2model.model;

public record SessionState(
    String sessionId,
    String datasetName,
    String userId,
    AgentStep currentStep,
    AnalysisResult analysisResult,
    String recommendation,
    String generatedCode,
    boolean awaitingSmoteConfirmation
) {
    public enum AgentStep {
        ANALYSIS, RECOMMENDATION, CLARIFICATION, CODE_GENERATION, DONE, ERROR
    }
}
