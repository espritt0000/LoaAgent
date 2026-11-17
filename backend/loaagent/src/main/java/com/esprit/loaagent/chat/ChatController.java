package com.esprit.loaagent.chat;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class ChatController {
    @GetMapping("/chat")
    String chat(){
        System.out.println("get /chat");
        return "chat.html";
    }
}
