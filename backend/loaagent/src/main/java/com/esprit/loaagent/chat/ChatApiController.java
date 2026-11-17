package com.esprit.loaagent.chat;

import com.esprit.loaagent.dto.ChatRequestDto;
import com.esprit.loaagent.dto.ChatResponseDto;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;

@RestController
public class ChatApiController {
    private final WebClient fastapiWebClient;

    public ChatApiController(WebClient fastapiWebClient){
        this.fastapiWebClient = fastapiWebClient;
    }

    @PostMapping("/chat/ask")
    public ChatResponseDto chatResponseDto(@RequestBody ChatRequestDto chatRequestDto){
        System.out.println("post /chat/ask");
        ChatResponseDto chatResponseDto = fastapiWebClient.post()
                .uri("/chat")
                .bodyValue(chatRequestDto)
                .retrieve()
                .bodyToMono(ChatResponseDto.class)
                .block();
        return chatResponseDto;
    }
}
