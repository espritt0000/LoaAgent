package com.esprit.loaagent.member;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import org.springframework.dao.DataIntegrityViolationException;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.model;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.redirectedUrl;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.view;

/**
 * 회원가입 입력값 검증 테스트
 * - MemberController만 MockMvc(standalone)로 띄우고 MemberService는 Mock으로 바꾼다.
 * - Spring 컨텍스트(DB, Security)를 띄우지 않으므로 MySQL 없이 실행된다.
 * - 화면(Thymeleaf) 렌더링은 확인하지 않고, 돌려주는 화면 이름과 오류 필드만 확인한다.
 */
class MemberSignupValidationTest {
    private MemberService memberService;
    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        memberService = mock(MemberService.class);
        mockMvc = MockMvcBuilders.standaloneSetup(new MemberController(memberService)).build();
    }

    @Test
    void 올바른_입력이면_가입하고_메인으로_이동() throws Exception {
        mockMvc.perform(post("/signup")
                        .param("username", "loauser01")
                        .param("password", "password123")
                        .param("email", "user@example.com"))
                .andExpect(redirectedUrl("/"));

        verify(memberService).join(any(MemberFormDto.class));
    }

    @Test
    void 동시_가입으로_DB_unique_제약에_걸리면_500이_아니라_중복_안내() throws Exception {
        // 두 요청이 중복 확인을 함께 통과한 뒤 두 번째 저장이 unique 제약에 걸리는 상황
        when(memberService.join(any(MemberFormDto.class))).thenThrow(new DataIntegrityViolationException("Duplicate entry"));

        mockMvc.perform(post("/signup")
                        .param("username", "loauser01")
                        .param("password", "password123")
                        .param("email", "user@example.com"))
                .andExpect(redirectedUrl("/signup?error=duplicate"));
    }

    @Test
    void 빈_아이디와_비밀번호면_가입하지_않고_회원가입_화면으로() throws Exception {
        mockMvc.perform(post("/signup")
                        .param("username", "  ")
                        .param("password", "")
                        .param("email", "user@example.com"))
                .andExpect(view().name("signup.html"))
                .andExpect(model().attributeHasFieldErrors("memberFormDto", "username", "password"));

        verify(memberService, never()).join(any());
    }

    @Test
    void 이메일_형식이_틀리면_오류() throws Exception {
        mockMvc.perform(post("/signup")
                        .param("username", "loauser01")
                        .param("password", "password123")
                        .param("email", "not-an-email"))
                .andExpect(view().name("signup.html"))
                .andExpect(model().attributeHasFieldErrors("memberFormDto", "email"));

        verify(memberService, never()).join(any());
    }

    @Test
    void 아이디_길이와_문자_규칙_위반이면_오류() throws Exception {
        // 3자(너무 짧음)
        mockMvc.perform(post("/signup")
                        .param("username", "abc")
                        .param("password", "password123")
                        .param("email", "user@example.com"))
                .andExpect(model().attributeHasFieldErrors("memberFormDto", "username"));

        // 한글·특수문자 포함
        mockMvc.perform(post("/signup")
                        .param("username", "로아user!")
                        .param("password", "password123")
                        .param("email", "user@example.com"))
                .andExpect(model().attributeHasFieldErrors("memberFormDto", "username"));

        verify(memberService, never()).join(any());
    }

    @Test
    void 비밀번호가_8자_미만이거나_64자_초과면_오류() throws Exception {
        mockMvc.perform(post("/signup")
                        .param("username", "loauser01")
                        .param("password", "short1")
                        .param("email", "user@example.com"))
                .andExpect(model().attributeHasFieldErrors("memberFormDto", "password"));

        mockMvc.perform(post("/signup")
                        .param("username", "loauser01")
                        .param("password", "a".repeat(65))
                        .param("email", "user@example.com"))
                .andExpect(model().attributeHasFieldErrors("memberFormDto", "password"));

        verify(memberService, never()).join(any());
    }

    @Test
    void 한글_비밀번호는_BCrypt_72바이트_제한_때문에_막음() throws Exception {
        // 한글 30자 = 90바이트. 글자 수(8~64자)는 통과하지만 BCrypt 제한(72바이트)을 넘어 가입 시 예외가 나므로 입력 단계에서 막는다
        mockMvc.perform(post("/signup")
                        .param("username", "loauser01")
                        .param("password", "가".repeat(30))
                        .param("email", "user@example.com"))
                .andExpect(view().name("signup.html"))
                .andExpect(model().attributeHasFieldErrors("memberFormDto", "password"));

        verify(memberService, never()).join(any());
    }
}
