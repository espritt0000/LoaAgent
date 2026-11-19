package com.esprit.loaagent.chat;

import com.esprit.loaagent.config.WebClientConfig;
import com.esprit.loaagent.dto.ChatRequestDto;
import com.esprit.loaagent.dto.ChatResponseDto;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * FastAPI 호출 타임아웃·오류 처리 테스트
 * - 실제 FastAPI 대신 JDK 내장 HttpServer로 가짜 서버를 띄워 지연/오류 상황을 만든다.
 * - Spring 컨텍스트(DB 등)를 띄우지 않으므로 MySQL 없이 실행된다.
 */
class ChatApiControllerTimeoutTest {
    private static final int CONNECT_TIMEOUT_MS = 1000;
    private static final long RESPONSE_TIMEOUT_SECONDS = 1;

    private HttpServer server;

    @AfterEach
    void stopServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    /** delayMs만큼 기다린 뒤 status 코드와 body로 응답하는 가짜 FastAPI 서버 */
    private String startFakeFastapi(long delayMs, int status, String body) throws Exception {
        server = HttpServer.create(new InetSocketAddress("localhost", 0), 0);
        server.createContext("/chat", exchange -> {
            try {
                Thread.sleep(delayMs);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(status, bytes.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(bytes);
            }
        });
        server.start();
        return "http://localhost:" + server.getAddress().getPort();
    }

    private ResponseEntity<ChatResponseDto> ask(String baseUrl) {
        ChatApiController controller = new ChatApiController(
                WebClientConfig.createFastapiWebClient(baseUrl, CONNECT_TIMEOUT_MS, RESPONSE_TIMEOUT_SECONDS));
        return controller.chatResponseDto(new ChatRequestDto(null, "안녕"), () -> "tester");
    }

    @Test
    void 정상_응답이면_200과_답변() throws Exception {
        String baseUrl = startFakeFastapi(0, 200, "{\"reply\":\"안녕하세요\"}");

        ResponseEntity<ChatResponseDto> response = ask(baseUrl);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals("안녕하세요", response.getBody().reply());
    }

    @Test
    void 응답이_제한시간보다_늦으면_504() throws Exception {
        String baseUrl = startFakeFastapi(3000, 200, "{\"reply\":\"늦은 답변\"}");

        long start = System.currentTimeMillis();
        ResponseEntity<ChatResponseDto> response = ask(baseUrl);
        long elapsedMs = System.currentTimeMillis() - start;

        assertEquals(HttpStatus.GATEWAY_TIMEOUT, response.getStatusCode());
        assertEquals("AI 응답이 지연되고 있습니다. 잠시 후 다시 시도해주세요.", response.getBody().reply());
        System.out.println("응답 시간 초과까지 걸린 시간(ms): " + elapsedMs);
    }

    @Test
    void 서버가_꺼져_있으면_503() throws Exception {
        // 빈 포트를 하나 얻은 뒤 바로 닫아서, 아무도 듣고 있지 않은 주소를 만든다
        int closedPort;
        try (ServerSocket socket = new ServerSocket(0)) {
            closedPort = socket.getLocalPort();
        }

        ResponseEntity<ChatResponseDto> response = ask("http://localhost:" + closedPort);

        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, response.getStatusCode());
        assertEquals("AI 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.", response.getBody().reply());
    }

    @Test
    void 서버가_500으로_응답하면_502() throws Exception {
        String baseUrl = startFakeFastapi(0, 500, "{\"detail\":\"Internal Server Error\"}");

        ResponseEntity<ChatResponseDto> response = ask(baseUrl);

        assertEquals(HttpStatus.BAD_GATEWAY, response.getStatusCode());
        assertEquals("AI 서버에서 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", response.getBody().reply());
    }
}
