package com.esprit.loaagent.chat;

import com.esprit.loaagent.config.SecurityConfig;
import com.esprit.loaagent.member.MemberController;
import com.esprit.loaagent.member.MemberService;
import org.junit.jupiter.api.Test;
import org.mockito.Answers;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.context.annotation.Import;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockHttpSession;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.hamcrest.Matchers.containsString;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.csrf;
import static org.springframework.security.test.web.servlet.request.SecurityMockMvcRequestPostProcessors.user;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * CSRF 보호 테스트
 * - 실제 SecurityConfig를 적용한 MockMvc로, 토큰 없는 POST·DELETE가 막히고 토큰이 있으면 통과하는지 확인한다.
 * - FastAPI 호출(WebClient)과 회원 서비스는 Mock이라 FastAPI·MySQL 없이 실행된다.
 */
@WebMvcTest(controllers = {ChatApiController.class, ChatController.class, MemberController.class})
@Import(SecurityConfig.class)
class CsrfProtectionTest {
    @Autowired
    private MockMvc mockMvc;

    @MockitoBean(answers = Answers.RETURNS_DEEP_STUBS)
    private WebClient fastapiWebClient;

    @MockitoBean
    private MemberService memberService;

    @Test
    void 토큰_없는_질문_요청은_403() throws Exception {
        mockMvc.perform(post("/chat/ask").with(user("tester"))
                        .contentType(MediaType.APPLICATION_JSON).content("{\"message\":\"안녕\"}"))
                .andExpect(status().isForbidden());
    }

    @Test
    void 토큰이_있는_질문_요청은_통과() throws Exception {
        mockMvc.perform(post("/chat/ask").with(user("tester")).with(csrf())
                        .contentType(MediaType.APPLICATION_JSON).content("{\"message\":\"안녕\"}"))
                .andExpect(status().isOk());
    }

    @Test
    void 토큰_없는_새_대화_삭제는_403() throws Exception {
        mockMvc.perform(delete("/chat/history").with(user("tester")))
                .andExpect(status().isForbidden());
    }

    @Test
    void 토큰이_있는_새_대화_삭제는_통과() throws Exception {
        mockMvc.perform(delete("/chat/history").with(user("tester")).with(csrf()))
                .andExpect(status().isNoContent());
    }

    @Test
    void 채팅_화면의_meta_토큰을_헤더로_보내면_통과() throws Exception {
        // chat.html의 fetch와 같은 방식: 화면에 심어진 토큰을 읽어 같은 세션으로 X-CSRF-TOKEN 헤더에 넣는다
        MockHttpSession session = new MockHttpSession();
        String html = mockMvc.perform(get("/chat").with(user("tester")).session(session))
                .andExpect(status().isOk())
                .andReturn().getResponse().getContentAsString();
        Matcher token = Pattern.compile("<meta name=\"_csrf\" content=\"([^\"]+)\"").matcher(html);
        Matcher header = Pattern.compile("<meta name=\"_csrf_header\" content=\"([^\"]+)\"").matcher(html);
        assertTrue(token.find() && header.find(), "chat.html에 CSRF meta 태그가 없음");

        mockMvc.perform(delete("/chat/history").with(user("tester")).session(session)
                        .header(header.group(1), token.group(1)))
                .andExpect(status().isNoContent());
    }

    @Test
    void 로그인_화면_폼에_CSRF_토큰이_들어간다() throws Exception {
        mockMvc.perform(get("/login"))
                .andExpect(status().isOk())
                .andExpect(content().string(containsString("name=\"_csrf\"")));
    }
}
