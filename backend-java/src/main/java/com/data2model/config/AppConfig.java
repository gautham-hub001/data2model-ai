package com.data2model.config;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;

@Configuration
public class AppConfig {

    @Bean
    public RestTemplate restTemplate() {
        // Buffer request body so RestTemplate sends Content-Length instead of
        // chunked transfer encoding — gunicorn sync workers choke on chunked uploads.
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setBufferRequestBody(true);
        factory.setConnectTimeout(10_000);
        factory.setReadTimeout(120_000); // 2 min for large file uploads to Python
        return new RestTemplate(factory);
    }

    @Bean
    public ChatClient.Builder chatClientBuilder(OpenAiChatModel chatModel) {
        return ChatClient.builder(chatModel)
            .defaultAdvisors()
            .defaultSystem("You are an expert ML engineer helping users choose and implement machine learning models.");
    }
}
