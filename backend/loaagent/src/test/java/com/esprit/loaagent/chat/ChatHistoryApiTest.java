package com.esprit.loaagent.chat;

import com.esprit.loaagent.config.WebClientConfig;
import com.esprit.loaagent.dto.ChatHistoryDto;
import com.esprit.loaagent.dto.ChatRequestDto;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 대화 기록 키(thread_id)와 /chat/history 중계 테스트
 * - JDK 내장 HttpServer로 가짜 FastAPI를 띄워, Spring이 FastAPI에 보내는 값을 확인한다.
 * - Spring 컨텍스트(DB 등)를 띄우지 않으므로 MySQL 없이 실행된다.
 */
class ChatHistoryApiTest {
    private HttpServer server;
    private final AtomicReference<String> received = new AtomicReference<>();

    @AfterEach
    void stopServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    /** path로 들어온 요청의 본문(POST) 또는 쿼리(GET)를 기록하고 body로 응답하는 가짜 FastAPI */
    private ChatApiController controllerWithFakeFastapi(String path, int status, String body) throws Exception {
        server = HttpServer.create(new InetSocketAddress("localhost", 0), 0);
        server.createContext(path, exchange -> {
            String requestBody = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            received.set(requestBody.isEmpty() ? exchange.getRequestURI().getQuery() : requestBody);
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(status, bytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(bytes);
            }
        });
        server.start();
        String baseUrl = "http://localhost:" + server.getAddress().getPort();
        return new ChatApiController(WebClientConfig.createFastapiWebClient(baseUrl, 1000, 5));
    }

    @Test
    void 질문할_때_thread_id는_브라우저_값이_아니라_로그인_계정으로_정한다() throws Exception {
        ChatApiController controller = controllerWithFakeFastapi("/chat", 200, "{\"reply\":\"안녕하세요\"}");

        controller.chatResponseDto(new ChatRequestDto("user:someone-else", "안녕"), () -> "tester");

        assertTrue(received.get().contains("\"thread_id\":\"user:tester\""), received.get());
    }

    @Test
    void 대화_기록은_로그인_계정의_thread_id로_조회한다() throws Exception {
        ChatApiController controller = controllerWithFakeFastapi("/history", 200,
                "{\"messages\":[{\"role\":\"user\",\"content\":\"안녕\"},{\"role\":\"ai\",\"content\":\"안녕하세요\"}]}");

        ResponseEntity<ChatHistoryDto> response = controller.chatHistory(() -> "tester");

        assertEquals("thread_id=user:tester", received.get());
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(2, response.getBody().messages().size());
        assertEquals("ai", response.getBody().messages().get(1).role());
    }

    @Test
    void 대화_기록_조회가_실패하면_502와_빈_목록() throws Exception {
        ChatApiController controller = controllerWithFakeFastapi("/history", 500, "{}");

        ResponseEntity<ChatHistoryDto> response = controller.chatHistory(() -> "tester");

        assertEquals(HttpStatus.BAD_GATEWAY, response.getStatusCode());
        assertTrue(response.getBody().messages().isEmpty());
    }

    @Test
    void 새_대화는_로그인_계정의_대화_기록만_지운다() throws Exception {
        ChatApiController controller = controllerWithFakeFastapi("/history", 200, "{\"deleted\":true}");

        ResponseEntity<Void> response = controller.deleteChatHistory(() -> "tester");

        assertEquals("thread_id=user:tester", received.get());
        assertEquals(HttpStatus.NO_CONTENT, response.getStatusCode());
    }

    @Test
    void 대화_기록_삭제가_실패하면_502() throws Exception {
        ChatApiController controller = controllerWithFakeFastapi("/history", 500, "{}");

        ResponseEntity<Void> response = controller.deleteChatHistory(() -> "tester");

        assertEquals(HttpStatus.BAD_GATEWAY, response.getStatusCode());
    }
}
