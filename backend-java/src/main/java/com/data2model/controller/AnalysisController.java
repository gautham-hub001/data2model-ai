package com.data2model.controller;

import com.data2model.agent.MultiAgentService;
import com.data2model.model.SessionState;
import com.data2model.tool.MLPipelineTool;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class AnalysisController {

    private final MultiAgentService multiAgentService;
    private final MLPipelineTool mlPipelineTool;

    public AnalysisController(MultiAgentService multiAgentService, MLPipelineTool mlPipelineTool) {
        this.multiAgentService = multiAgentService;
        this.mlPipelineTool = mlPipelineTool;
    }

    @PostMapping("/analyze")
    public ResponseEntity<Map<String, String>> analyze(
        @RequestParam("file") MultipartFile file,
        @AuthenticationPrincipal Jwt jwt
    ) throws IOException {
        String userId = jwt.getSubject();
        String filename = file.getOriginalFilename();

        // Store dataset in Python/Supabase, get back a dataset_id
        String datasetId = mlPipelineTool.storeDataset(file.getBytes(), filename);

        // Kick off the async multi-agent workflow
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
