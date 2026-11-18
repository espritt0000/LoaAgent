package com.esprit.loaagent.chat;

import com.esprit.loaagent.dto.ChatRequestDto;
import com.esprit.loaagent.dto.ChatResponseDto;
import io.netty.handler.timeout.ReadTimeoutException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientRequestException;
import org.springframework.web.reactive.function.client.WebClientResponseException;

@RestController
public class ChatApiController {
    private final WebClient fastapiWebClient;

    public ChatApiController(WebClient fastapiWebClient){
        this.fastapiWebClient = fastapiWebClient;
    }

    @PostMapping("/chat/ask")
    public ResponseEntity<ChatResponseDto> chatResponseDto(@RequestBody ChatRequestDto chatRequestDto){
        System.out.println("post /chat/ask");
        try {
            ChatResponseDto chatResponseDto = fastapiWebClient.post()
                    .uri("/chat")
                    .bodyValue(chatRequestDto)
                    .retrieve()
                    .bodyToMono(ChatResponseDto.class)
                    .block();
            return ResponseEntity.ok(chatResponseDto);
        } catch (WebClientRequestException e) {
            // 응답을 받기 전에 실패: 응답 시간 초과 또는 연결 실패
            if (e.getCause() instanceof ReadTimeoutException) {
                System.out.println("⏱️ [FastAPI Timeout] 응답 시간 초과 - " + e.getUri());
                return errorResponse(HttpStatus.GATEWAY_TIMEOUT,
                        "AI 응답이 지연되고 있습니다. 잠시 후 다시 시도해주세요.");
            }
            System.out.println("🔌 [FastAPI Connect Error] 연결 실패 - " + e.getMessage());
            return errorResponse(HttpStatus.SERVICE_UNAVAILABLE,
                    "AI 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.");
        } catch (WebClientResponseException e) {
            // FastAPI가 4xx/5xx로 응답한 경우
            System.out.println("❌ [FastAPI Error Response] 상태 코드 " + e.getStatusCode().value());
            return errorResponse(HttpStatus.BAD_GATEWAY,
                    "AI 서버에서 오류가 발생했습니다. 잠시 후 다시 시도해주세요.");
        }
    }

    private ResponseEntity<ChatResponseDto> errorResponse(HttpStatus status, String message) {
        return ResponseEntity.status(status).body(new ChatResponseDto(message));
    }
}
