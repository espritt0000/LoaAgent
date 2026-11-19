package com.esprit.loaagent.dto;

import java.util.List;

// FastAPI /history 응답: 저장된 대화의 사용자 질문과 최종 AI 답변
public record ChatHistoryDto(List<Message> messages) {
    public record Message(String role, String content) {}
}
