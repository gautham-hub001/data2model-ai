package com.data2model.agent;

import com.data2model.model.AnalysisResult;
import com.data2model.model.SessionState;
import com.data2model.model.SessionState.AgentStep;
import com.data2model.model.StreamChunk;
import com.data2model.tool.MLPipelineTool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

import reactor.core.publisher.Flux;

import java.time.Duration;
import java.util.function.Supplier;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CompletableFuture;

@Service
public class MultiAgentService {

    private static final Logger log = LoggerFactory.getLogger(MultiAgentService.class);

    private final ChatClient chatClient;
    private final MLPipelineTool mlPipelineTool;
    private final SimpMessagingTemplate ws;

    // In-memory session store. Replace with Supabase persistence for production.
    private final Map<String, SessionState> sessions = new ConcurrentHashMap<>();
    // Signals whether user confirmed SMOTE (null = waiting, true/false = decided)
    private final Map<String, Boolean> smoteDecisions = new ConcurrentHashMap<>();

    public MultiAgentService(
        ChatClient.Builder chatClientBuilder,
        MLPipelineTool mlPipelineTool,
        SimpMessagingTemplate ws
    ) {
        this.chatClient = chatClientBuilder
            .defaultTools(mlPipelineTool)
            .build();
        this.mlPipelineTool = mlPipelineTool;
        this.ws = ws;
    }

    /**
     * Starts the 4-step agentic workflow asynchronously.
     * Each step streams tokens to /topic/session/{id}/stream via WebSocket.
     */
    public String startWorkflow(String datasetId, String datasetName, String userId) {
        String sessionId = UUID.randomUUID().toString();
        SessionState initial = new SessionState(
            sessionId, datasetName, userId, AgentStep.ANALYSIS,
            null, null, null, false
        );
        sessions.put(sessionId, initial);

        CompletableFuture.runAsync(() -> runWorkflow(sessionId, datasetId));
        return sessionId;
    }

    public void confirmSmote(String sessionId, boolean apply) {
        smoteDecisions.put(sessionId, apply);
    }

    public SessionState getSession(String sessionId) {
        return sessions.get(sessionId);
    }

    // ──────────────────────────────────────────────────────────────────────────

    private void runWorkflow(String sessionId, String datasetId) {
        try {
            // Give the browser 3 seconds to establish the STOMP subscription before
            // we start emitting — prevents early messages being missed due to the
            // WebSocket handshake happening after the workflow thread starts.
            Thread.sleep(3_000);

            // Step 1: Data Analysis (tool call to Python)
            emit(sessionId, StreamChunk.stepStart("ANALYSIS"));
            AnalysisResult analysis = runAnalysisStep(sessionId, datasetId);

            // Step 2: Model Recommendation (LLM reasoning, streamed)
            emit(sessionId, StreamChunk.stepStart("RECOMMENDATION"));
            String recommendation = runRecommendationStep(sessionId, analysis);

            // Step 3: Clarification (SMOTE loop if imbalance detected)
            if (analysis.imbalanceDetected()) {
                emit(sessionId, StreamChunk.stepStart("CLARIFICATION"));
                emit(sessionId, StreamChunk.token(
                    "⚠️ Class imbalance detected in your dataset. Would you like to apply SMOTE oversampling to balance it before training?\n"
                ));

                boolean applySmote = waitForSmoteDecision(sessionId);
                if (applySmote) {
                    emit(sessionId, StreamChunk.token("Applying SMOTE and re-analyzing…\n"));
                    analysis = mlPipelineTool.runAnalysisWithSmote(datasetId);
                    emit(sessionId, StreamChunk.token("✓ Re-analysis complete with SMOTE applied.\n"));
                } else {
                    emit(sessionId, StreamChunk.token("Continuing without SMOTE.\n"));
                }
            }

            // Pause between LLM calls to stay within OpenRouter free-tier rate limit (~20 RPM)
            Thread.sleep(5_000);

            // Step 4: Code Generation (streamed token by token)
            emit(sessionId, StreamChunk.stepStart("CODE_GENERATION"));
            String code = runCodeGenerationStep(sessionId, analysis, recommendation);

            // Mark done
            SessionState finalState = new SessionState(
                sessionId,
                sessions.get(sessionId).datasetName(),
                sessions.get(sessionId).userId(),
                AgentStep.DONE,
                analysis, recommendation, code, false
            );
            sessions.put(sessionId, finalState);
            emit(sessionId, StreamChunk.done());

        } catch (Exception e) {
            log.error("Workflow error for session {}", sessionId, e);
            emit(sessionId, StreamChunk.error("Analysis failed: " + e.getMessage()));
        }
    }

    private AnalysisResult runAnalysisStep(String sessionId, String datasetId) {
        // Use tool call so the LLM can invoke runAnalysis if needed; here we call directly
        // for a deterministic first step (no hallucination risk).
        AnalysisResult result = mlPipelineTool.runAnalysis(datasetId);
        emit(sessionId, StreamChunk.token(
            "✓ Analyzed " + result.analysis().getOrDefault("num_rows", "?") + " rows × " +
            result.analysis().getOrDefault("num_columns", "?") + " columns.\n"
        ));
        return result;
    }

    private String runRecommendationStep(String sessionId, AnalysisResult analysis) {
        // Supplier ensures each retry attempt builds a brand-new HTTP request
        Supplier<Flux<String>> streamSupplier = () -> chatClient.prompt()
            .user(u -> u.text("""
                You are an expert ML engineer. Given the following dataset analysis, recommend the best ML model.
                Explain your reasoning in 2-3 sentences. Be specific about WHY this model fits this data.

                Dataset analysis:
                {analysis}

                Current recommendation from rules engine: {recommendation}
                """)
                .param("analysis", analysis.analysis().toString())
                .param("recommendation", analysis.recommendation().toString())
            )
            .stream()
            .content();

        return streamWithRetry(sessionId, streamSupplier);
    }

    private String runCodeGenerationStep(String sessionId, AnalysisResult analysis, String recommendation) {
        Supplier<Flux<String>> streamSupplier = () -> chatClient.prompt()
            .user(u -> u.text("""
                Generate production-ready scikit-learn Python code for the following ML pipeline.
                Include preprocessing, train/test split, model training, and evaluation metrics.
                Output ONLY the Python code block — no explanation text before or after.

                Dataset profile:
                {analysis}

                Recommended model and reasoning:
                {recommendation}

                SMOTE applied: {smote}
                """)
                .param("analysis", analysis.analysis().toString())
                .param("recommendation", recommendation)
                .param("smote", String.valueOf(analysis.smoteApplied()))
            )
            .stream()
            .content();

        return streamWithRetry(sessionId, streamSupplier);
    }

    /** Blocks until the user responds to the SMOTE clarification (max 5 minutes). */
    private boolean waitForSmoteDecision(String sessionId) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 5 * 60 * 1000;
        while (!smoteDecisions.containsKey(sessionId)) {
            if (System.currentTimeMillis() > deadline) return false; // timeout → skip SMOTE
            Thread.sleep(500);
        }
        return smoteDecisions.remove(sessionId);
    }

    private void emit(String sessionId, StreamChunk chunk) {
        ws.convertAndSend("/topic/session/" + sessionId + "/stream", chunk);
    }

    /**
     * Streams an LLM response with manual retry on 429.
     *
     * Spring AI's MessageAggregator catches 429s internally before Reactor's
     * retryWhen can see them, so we use a plain try-catch loop instead.
     * Each retry calls the supplier to create a brand-new HTTP request.
     */
    private String streamWithRetry(String sessionId, Supplier<Flux<String>> streamSupplier) {
        // Backoff delays: 30s → 60s → 120s
        long[] backoffMs = {30_000, 60_000, 120_000};

        Exception lastEx = null;
        for (int attempt = 0; attempt <= backoffMs.length; attempt++) {
            StringBuilder sb = new StringBuilder();
            try {
                streamSupplier.get()
                    // Batch up to 20 tokens OR flush every 100 ms — whichever comes first.
                    // Cuts WebSocket messages from ~400 down to ~20-30 per LLM response.
                    .bufferTimeout(20, Duration.ofMillis(100))
                    .doOnNext(batch -> {
                        String chunk = String.join("", batch);
                        sb.append(chunk);
                        emit(sessionId, StreamChunk.token(chunk));
                    })
                    .blockLast();
                return sb.toString(); // success
            } catch (Exception e) {
                lastEx = e;
                boolean is429 = e.getMessage() != null && e.getMessage().contains("429");
                if (!is429 || attempt >= backoffMs.length) break;

                long waitMs = backoffMs[attempt];
                log.warn("OpenRouter 429 on session {}, attempt {}/{}, waiting {}s",
                    sessionId, attempt + 1, backoffMs.length, waitMs / 1000);
                emit(sessionId, StreamChunk.token(
                    "\n⏳ Rate limited by OpenRouter. Retrying in " + (waitMs / 1000) + "s…\n"));
                try {
                    Thread.sleep(waitMs);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
        throw new RuntimeException("LLM call failed after retries: " + lastEx.getMessage(), lastEx);
    }
}
