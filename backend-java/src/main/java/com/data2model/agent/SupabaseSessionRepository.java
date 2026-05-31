package com.data2model.agent;

import com.data2model.model.SessionState;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

/**
 * Persists session state to Supabase via PostgREST.
 * Disabled (no-op) when SUPABASE_URL or SUPABASE_SERVICE_KEY env vars are absent.
 */
@Component
public class SupabaseSessionRepository {

    private static final Logger log = LoggerFactory.getLogger(SupabaseSessionRepository.class);

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    private final String supabaseUrl;
    private final String supabaseKey;
    private final boolean enabled;

    public SupabaseSessionRepository(
        RestTemplate restTemplate,
        @Value("${app.supabase.url:}") String supabaseUrl,
        @Value("${app.supabase.service-key:}") String supabaseKey
    ) {
        this.restTemplate = restTemplate;
        this.objectMapper = new ObjectMapper();
        this.supabaseUrl = supabaseUrl;
        this.supabaseKey = supabaseKey;
        this.enabled = !supabaseUrl.isBlank() && !supabaseKey.isBlank();
        if (!enabled) {
            log.warn("Supabase env vars not set — sessions will not be persisted across restarts.");
        }
    }

    public void save(SessionState session, String datasetId) {
        if (!enabled) return;
        try {
            Map<String, Object> body = new HashMap<>();
            body.put("id", session.sessionId());
            body.put("user_id", session.userId());
            body.put("dataset_name", session.datasetName());
            body.put("dataset_id", datasetId);
            body.put("smote_applied",
                session.analysisResult() != null && session.analysisResult().smoteApplied());
            if (session.analysisResult() != null) {
                body.put("analysis_result", session.analysisResult().analysis());
                body.put("recommendation", Map.of("text", session.recommendation() != null
                    ? session.recommendation() : ""));
            }
            if (session.generatedCode() != null) {
                body.put("generated_code", session.generatedCode());
            }

            HttpHeaders headers = buildHeaders();
            headers.set("Prefer", "resolution=merge-duplicates,return=minimal");

            restTemplate.exchange(
                supabaseUrl + "/rest/v1/sessions",
                HttpMethod.POST,
                new HttpEntity<>(body, headers),
                Void.class
            );
        } catch (Exception e) {
            log.error("Failed to persist session {} to Supabase: {}", session.sessionId(), e.getMessage());
        }
    }

    private HttpHeaders buildHeaders() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.set("Authorization", "Bearer " + supabaseKey);
        headers.set("apikey", supabaseKey);
        return headers;
    }
}
