package com.data2model.tool;

import com.data2model.model.AnalysisResult;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

@Component
public class MLPipelineTool {

    private final RestTemplate restTemplate;
    private final String pythonApiUrl;
    private final String internalToken;

    public MLPipelineTool(
        RestTemplate restTemplate,
        @Value("${app.python-api-url}") String pythonApiUrl,
        @Value("${app.internal-token}") String internalToken
    ) {
        this.restTemplate = restTemplate;
        this.pythonApiUrl = pythonApiUrl;
        this.internalToken = internalToken;
    }

    @Tool(description = "Run ML analysis on an uploaded CSV dataset stored in Supabase. Returns dataset statistics, column types, missing values, correlations, and an initial model recommendation.")
    public AnalysisResult runAnalysis(String datasetId) {
        return callPython("/analyze-by-id", datasetId);
    }

    @Tool(description = "Re-run ML analysis with SMOTE oversampling applied to fix class imbalance. Use only when class imbalance was detected in a previous analysis.")
    public AnalysisResult runAnalysisWithSmote(String datasetId) {
        return callPython("/analyze-smote-by-id", datasetId);
    }

    private AnalysisResult callPython(String endpoint, String datasetId) {
        HttpHeaders headers = new HttpHeaders();
        headers.set("X-Internal-Token", internalToken);
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<Map<String, String>> request = new HttpEntity<>(Map.of("dataset_id", datasetId), headers);
        ResponseEntity<AnalysisResult> response = restTemplate.exchange(
            pythonApiUrl + endpoint,
            HttpMethod.POST,
            request,
            AnalysisResult.class
        );
        return response.getBody();
    }

    /** Called during upload — sends the raw CSV bytes to Python for storage. */
    public String storeDataset(byte[] csvBytes, String filename) {
        HttpHeaders headers = new HttpHeaders();
        headers.set("X-Internal-Token", internalToken);
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        ByteArrayResource fileResource = new ByteArrayResource(csvBytes) {
            @Override
            public String getFilename() { return filename; }
        };

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);

        HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);
        ResponseEntity<String> response = restTemplate.postForEntity(
            pythonApiUrl + "/store-dataset",
            request,
            String.class
        );
        // Python returns {"dataset_id": "..."}
        String raw = response.getBody();
        return raw != null ? raw.replaceAll(".*\"dataset_id\"\\s*:\\s*\"([^\"]+)\".*", "$1") : "";
    }
}
