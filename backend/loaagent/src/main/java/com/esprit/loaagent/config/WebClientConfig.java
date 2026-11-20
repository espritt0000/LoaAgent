package com.esprit.loaagent.config;

import io.netty.channel.ChannelOption;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;

import java.time.Duration;

@Configuration
public class WebClientConfig {
    @Bean
    public WebClient fastapiWebClient(
            // FastAPI 내부 호출용 API Key (application-secret.properties, 값이 없으면 서버 시작 단계에서 실패)
            @Value("${fastapi.api-key}") String apiKey,
            @Value("${fastapi.connect-timeout-ms:3000}") int connectTimeoutMs,
            @Value("${fastapi.response-timeout-seconds:120}") long responseTimeoutSeconds) {
        return createFastapiWebClient("http://localhost:8000", apiKey, connectTimeoutMs, responseTimeoutSeconds);
    }

    // 테스트에서도 같은 설정으로 WebClient를 만들 수 있도록 분리
    public static WebClient createFastapiWebClient(String baseUrl, String apiKey, int connectTimeoutMs, long responseTimeoutSeconds) {
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("🚨 [API KEY CONFIG] application-secret.properties에 fastapi.api-key를 설정해야 합니다");
        }
        HttpClient httpClient = HttpClient.create()
                // FastAPI 서버에 연결하는 데 걸리는 최대 시간 (서버가 꺼져 있거나 응답이 없을 때)
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, connectTimeoutMs)
                // 요청을 보낸 뒤 응답을 기다리는 최대 시간 (LLM 처리가 길어질 때)
                .responseTimeout(Duration.ofSeconds(responseTimeoutSeconds));

        return WebClient.builder()
                .baseUrl(baseUrl)
                .defaultHeader("Content-Type", "application/json")
                // FastAPI가 Spring에서 온 요청인지 확인하는 키 (모든 요청에 붙음)
                .defaultHeader("X-API-Key", apiKey)
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .build();
    }
}
