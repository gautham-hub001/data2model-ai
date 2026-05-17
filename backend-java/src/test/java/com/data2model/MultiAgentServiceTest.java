package com.data2model;

import com.data2model.agent.MultiAgentService;
import com.data2model.model.AnalysisResult;
import com.data2model.model.SessionState;
import com.data2model.tool.MLPipelineTool;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import reactor.core.publisher.Flux;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class MultiAgentServiceTest {

    @Test
    void startWorkflow_returnsSessionId() {
        MLPipelineTool tool = mock(MLPipelineTool.class);
        SimpMessagingTemplate ws = mock(SimpMessagingTemplate.class);
        ChatClient.Builder builder = mock(ChatClient.Builder.class, RETURNS_DEEP_STUBS);

        AnalysisResult fakeAnalysis = new AnalysisResult(
            Map.of("num_rows", 100, "num_columns", 5),
            Map.of("model", "RandomForestClassifier"),
            "Good dataset",
            "",
            false,
            false
        );

        when(tool.runAnalysis(anyString())).thenReturn(fakeAnalysis);

        // Mock streaming responses
        when(builder.defaultTools((Object[]) any()).build()
            .prompt().user(any(java.util.function.Consumer.class))
            .stream().content()
        ).thenReturn(Flux.just("mocked token"));

        MultiAgentService service = new MultiAgentService(builder, tool, ws);
        String sessionId = service.startWorkflow("dataset-123", "test.csv", "user-1");

        assertThat(sessionId).isNotNull().isNotBlank();
        assertThat(service.getSession(sessionId)).isNotNull();
        assertThat(service.getSession(sessionId).datasetName()).isEqualTo("test.csv");
    }
}
