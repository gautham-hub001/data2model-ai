package com.data2model.controller;

import com.data2model.agent.MultiAgentService;
import com.data2model.config.RateLimiterService;
import com.data2model.model.SessionState;
import com.data2model.tool.MLPipelineTool;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;
import java.util.Set;

@RestController
@RequestMapping("/api")
public class AnalysisController {

    private static final long MAX_FILE_BYTES = 10L * 1024 * 1024; // 10 MB
    private static final Set<String> ALLOWED_CONTENT_TYPES = Set.of(
        "text/csv", "application/csv", "application/vnd.ms-excel", "application/octet-stream"
    );

    private final MultiAgentService multiAgentService;
    private final MLPipelineTool mlPipelineTool;
    private final RateLimiterService rateLimiterService;

    public AnalysisController(
        MultiAgentService multiAgentService,
        MLPipelineTool mlPipelineTool,
        RateLimiterService rateLimiterService
    ) {
        this.multiAgentService = multiAgentService;
        this.mlPipelineTool = mlPipelineTool;
        this.rateLimiterService = rateLimiterService;
    }

    @PostMapping("/analyze")
    public ResponseEntity<?> analyze(
        @RequestParam("file") MultipartFile file,
        @AuthenticationPrincipal Jwt jwt
    ) throws IOException {
        String userId = jwt.getSubject();

        if (!rateLimiterService.tryConsume(userId)) {
            return ResponseEntity.status(429).body(
                Map.of("error", "Rate limit exceeded. You can submit up to 3 analyses per hour.")
            );
        }

        if (multiAgentService.isUserActive(userId)) {
            return ResponseEntity.status(429).body(
                Map.of("error", "You already have an analysis in progress. Please wait for it to finish.")
            );
        }

        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "File is empty."));
        }

        if (file.getSize() > MAX_FILE_BYTES) {
            return ResponseEntity.badRequest().body(Map.of("error", "File too large. Maximum size is 10 MB."));
        }

        String filename = file.getOriginalFilename();
        if (filename == null || !filename.toLowerCase().endsWith(".csv")) {
            return ResponseEntity.badRequest().body(Map.of("error", "Only CSV files are accepted."));
        }

        String contentType = file.getContentType();
        if (contentType != null && !ALLOWED_CONTENT_TYPES.contains(contentType)) {
            return ResponseEntity.badRequest().body(Map.of("error", "Only CSV files are accepted."));
        }

        String datasetId = mlPipelineTool.storeDataset(file.getBytes(), filename);
        String sessionId = multiAgentService.startWorkflow(datasetId, filename, userId);

        return ResponseEntity.ok(Map.of("sessionId", sessionId, "datasetName", filename));
    }

    @GetMapping("/session/{sessionId}")
    public ResponseEntity<?> getSession(
        @PathVariable String sessionId,
        @AuthenticationPrincipal Jwt jwt
    ) {
        SessionState state = multiAgentService.getSession(sessionId);
        if (state == null) return ResponseEntity.notFound().build();
        if (!state.userId().equals(jwt.getSubject())) return ResponseEntity.status(403).build();
        return ResponseEntity.ok(state);
    }

    @PostMapping("/session/{sessionId}/smote")
    public ResponseEntity<Void> confirmSmote(
        @PathVariable String sessionId,
        @RequestBody Map<String, Boolean> body,
        @AuthenticationPrincipal Jwt jwt
    ) {
        SessionState state = multiAgentService.getSession(sessionId);
        if (state == null) return ResponseEntity.notFound().build();
        if (!state.userId().equals(jwt.getSubject())) return ResponseEntity.status(403).build();

        multiAgentService.confirmSmote(sessionId, Boolean.TRUE.equals(body.get("applySmote")));
        return ResponseEntity.ok().build();
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of("status", "UP"));
    }
}
