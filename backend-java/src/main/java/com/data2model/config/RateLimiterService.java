package com.data2model.config;

import io.github.bucket4j.Bandwidth;
import io.github.bucket4j.Bucket;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

@Service
public class RateLimiterService {

    private final Map<String, Bucket> buckets = new ConcurrentHashMap<>();
    private final int requestsPerHour;

    public RateLimiterService(@Value("${app.rate-limit.requests-per-hour:3}") int requestsPerHour) {
        this.requestsPerHour = requestsPerHour;
    }

    public boolean tryConsume(String userId) {
        Bucket bucket = buckets.computeIfAbsent(userId, id ->
            Bucket.builder()
                .addLimit(Bandwidth.builder()
                    .capacity(requestsPerHour)
                    .refillGreedy(requestsPerHour, Duration.ofHours(1))
                    .build())
                .build()
        );
        return bucket.tryConsume(1);
    }
}
