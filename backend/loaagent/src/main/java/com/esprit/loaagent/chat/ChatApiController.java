package com.esprit.loaagent.chat;

import com.esprit.loaagent.dto.ChatHistoryDto;
import com.esprit.loaagent.dto.ChatRequestDto;
import com.esprit.loaagent.dto.ChatResponseDto;
import io.netty.handler.timeout.ReadTimeoutException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientRequestException;
import org.springframework.web.reactive.function.client.WebClientException;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import java.security.Principal;
import java.util.List;

@RestController
public class ChatApiController {
    private final WebClient fastapiWebClient;

    public ChatApiController(WebClient fastapiWebClient){
        this.fastapiWebClient = fastapiWebClient;
    }

    // 대화 기록 키(thread_id)는 브라우저가 보낸 값이 아니라 로그인 계정으로 서버에서 정한다
    // (브라우저 값을 믿으면 다른 사람의 thread_id를 넣어 그 사람의 대화를 이어 쓸 수 있음)
    static String threadIdOf(Principal principal) {
        return "user:" + principal.getName();
    }

    @PostMapping("/chat/ask")
    public ResponseEntity<ChatResponseDto> chatResponseDto(@RequestBody ChatRequestDto chatRequestDto, Principal principal){
        System.out.println("post /chat/ask");
        ChatRequestDto fastapiRequest = new ChatRequestDto(threadIdOf(principal), chatRequestDto.message());
        try {
            ChatResponseDto chatResponseDto = fastapiWebClient.post()
                    .uri("/chat")
                    .bodyValue(fastapiRequest)
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

    // 새로고침해도 이전 대화를 화면에 다시 보여 주기 위해 로그인 계정의 대화 기록을 FastAPI에서 가져온다
    @GetMapping("/chat/history")
    public ResponseEntity<ChatHistoryDto> chatHistory(Principal principal) {
        try {
            ChatHistoryDto history = fastapiWebClient.get()
                    .uri(uriBuilder -> uriBuilder.path("/history").queryParam("thread_id", threadIdOf(principal)).build())
                    .retrieve()
                    .bodyToMono(ChatHistoryDto.class)
                    .block();
            return ResponseEntity.ok(history);
        } catch (WebClientException e) {
            // 기록을 못 가져와도 채팅은 할 수 있으므로 빈 목록과 502만 돌려준다
            System.out.println("📜 [History Error] 대화 기록 조회 실패 - " + e.getClass().getSimpleName());
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body(new ChatHistoryDto(List.of()));
        }
    }

    private ResponseEntity<ChatResponseDto> errorResponse(HttpStatus status, String message) {
        return ResponseEntity.status(status).body(new ChatResponseDto(message));
    }
}
