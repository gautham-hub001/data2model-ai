package com.data2model.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public record AnalysisResult(
    Map<String, Object> analysis,
    Map<String, Object> recommendation,
    String explanation,
    String code,
    boolean imbalanceDetected,
    boolean smoteApplied
) {}
