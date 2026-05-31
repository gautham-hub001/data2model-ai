package com.data2model.agent;

import com.data2model.model.AnalysisResult;
import com.data2model.model.SessionState;
import com.data2model.model.SessionState.AgentStep;
import com.data2model.model.StreamChunk;
import com.data2model.tool.MLPipelineTool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

import reactor.core.publisher.Flux;

import java.time.Duration;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.function.Supplier;

@Service
public class MultiAgentService {

    private static final Logger log = LoggerFactory.getLogger(MultiAgentService.class);

    private final ChatClient chatClient;
    private final MLPipelineTool mlPipelineTool;
    private final SimpMessagingTemplate ws;
    private final SupabaseSessionRepository sessionRepository;
    private final long streamTimeoutSeconds;
    private final long stompReadyDelayMs;

    private static final long WORKFLOW_LOCK_EXPIRY_MS = 15 * 60 * 1000; // 15 minutes

    private final Map<String, SessionState> sessions = new ConcurrentHashMap<>();
    private final Map<String, CompletableFuture<Boolean>> smoteDecisions = new ConcurrentHashMap<>();
    // userId → workflow start time; acts as the concurrency lock
    private final Map<String, Long> activeUsers = new ConcurrentHashMap<>();

    public MultiAgentService(
        ChatClient.Builder chatClientBuilder,
        MLPipelineTool mlPipelineTool,
        SimpMessagingTemplate ws,
        SupabaseSessionRepository sessionRepository,
        @Value("${app.workflow.stream-timeout-seconds:120}") long streamTimeoutSeconds,
        @Value("${app.workflow.stomp-ready-delay-ms:1000}") long stompReadyDelayMs
    ) {
        this.chatClient = chatClientBuilder.defaultTools(mlPipelineTool).build();
        this.mlPipelineTool = mlPipelineTool;
        this.ws = ws;
        this.sessionRepository = sessionRepository;
        this.streamTimeoutSeconds = streamTimeoutSeconds;
        this.stompReadyDelayMs = stompReadyDelayMs;
    }

    public boolean isUserActive(String userId) {
        Long startTime = activeUsers.get(userId);
        if (startTime == null) return false;
        // Treat the lock as expired if the workflow has been running longer than the max expected duration
        if (System.currentTimeMillis() - startTime > WORKFLOW_LOCK_EXPIRY_MS) {
            activeUsers.remove(userId);
            return false;
        }
        return true;
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
        activeUsers.put(userId, System.currentTimeMillis());

        CompletableFuture.runAsync(() -> {
            try {
                runWorkflow(sessionId, datasetId);
            } finally {
                activeUsers.remove(userId);  // always release the lock when done or failed
            }
        });

        return sessionId;
    }

    public void confirmSmote(String sessionId, boolean apply) {
        CompletableFuture<Boolean> future = smoteDecisions.get(sessionId);
        if (future != null) {
            future.complete(apply);
        }
    }

    public SessionState getSession(String sessionId) {
        return sessions.get(sessionId);
    }

    // ──────────────────────────────────────────────────────────────────────────

    private void runWorkflow(String sessionId, String datasetId) {
        try {
            // Give the browser time to establish the STOMP subscription before
            // we start emitting. Configurable via STOMP_READY_DELAY_MS env var.
            // A more robust alternative is a client-driven READY signal over /app.
            Thread.sleep(stompReadyDelayMs);

            // Step 1: Data Analysis (deterministic tool call to Python — no LLM)
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

            // Pause between LLM calls to stay within free-tier rate limit (~20 RPM)
            Thread.sleep(5_000);

            // Step 4: Code Generation (streamed token by token)
            emit(sessionId, StreamChunk.stepStart("CODE_GENERATION"));
            String code = runCodeGenerationStep(sessionId, analysis, recommendation);

            SessionState finalState = new SessionState(
                sessionId,
                sessions.get(sessionId).datasetName(),
                sessions.get(sessionId).userId(),
                AgentStep.DONE,
                analysis, recommendation, code, false
            );
            sessions.put(sessionId, finalState);
            sessionRepository.save(finalState, datasetId);
            emit(sessionId, StreamChunk.done());

        } catch (Exception e) {
            log.error("Workflow error for session {}", sessionId, e);
            emit(sessionId, StreamChunk.error("Analysis failed: " + e.getMessage()));
        }
    }

    private AnalysisResult runAnalysisStep(String sessionId, String datasetId) {
        AnalysisResult result = mlPipelineTool.runAnalysis(datasetId);
        String rows = "?";
        String cols = "?";
        Object metaRaw = result.analysis().get("meta");
        if (metaRaw instanceof Map<?, ?> meta) {
            if (meta.get("num_rows") != null) rows = String.valueOf(meta.get("num_rows"));
            if (meta.get("num_cols") != null) cols = String.valueOf(meta.get("num_cols"));
        }
        emit(sessionId, StreamChunk.token("✓ Analyzed " + rows + " rows × " + cols + " columns.\n"));
        return result;
    }

    private String runRecommendationStep(String sessionId, AnalysisResult analysis) {
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

    /** Suspends the workflow thread until the user responds (max 5 minutes). */
    private boolean waitForSmoteDecision(String sessionId) {
        CompletableFuture<Boolean> future = new CompletableFuture<>();
        smoteDecisions.put(sessionId, future);
        try {
            return future.get(5, TimeUnit.MINUTES);
        } catch (TimeoutException e) {
            log.info("SMOTE decision timed out for session {} — skipping SMOTE", sessionId);
            return false;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        } catch (ExecutionException e) {
            log.error("Error in SMOTE decision for session {}", sessionId, e);
            return false;
        } finally {
            smoteDecisions.remove(sessionId);
        }
    }

    private void emit(String sessionId, StreamChunk chunk) {
        ws.convertAndSend("/topic/session/" + sessionId + "/stream", chunk);
    }

    /**
     * Streams an LLM response with manual retry on 429 (rate limit).
     *
     * Spring AI's MessageAggregator catches 429s internally before Reactor's
     * retryWhen can see them, so we use a plain try-catch loop.
     * Each retry calls the supplier to create a brand-new HTTP request.
     */
    private String streamWithRetry(String sessionId, Supplier<Flux<String>> streamSupplier) {
        long[] backoffMs = {30_000, 60_000, 120_000};

        Exception lastEx = null;
        for (int attempt = 0; attempt <= backoffMs.length; attempt++) {
            StringBuilder sb = new StringBuilder();
            try {
                streamSupplier.get()
                    .timeout(Duration.ofSeconds(streamTimeoutSeconds))
                    .bufferTimeout(20, Duration.ofMillis(100))
                    .doOnNext(batch -> {
                        String chunk = String.join("", batch);
                        sb.append(chunk);
                        emit(sessionId, StreamChunk.token(chunk));
                    })
                    .blockLast();
                return sb.toString();
            } catch (Exception e) {
                lastEx = e;
                boolean is429 = e.getMessage() != null && e.getMessage().contains("429");
                if (!is429 || attempt >= backoffMs.length) break;

                long waitMs = backoffMs[attempt];
                log.warn("Rate limited on session {}, attempt {}/{}, waiting {}s",
                    sessionId, attempt + 1, backoffMs.length, waitMs / 1000);
                emit(sessionId, StreamChunk.token(
                    "\n⏳ Rate limited. Retrying in " + (waitMs / 1000) + "s…\n"));
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
