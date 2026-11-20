package com.esprit.loaagent.member;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller @RequiredArgsConstructor
public class MemberController {
    private final MemberService memberService;

    @GetMapping("/login")
    String login(@RequestParam(value = "error", required = false) String error,
                 Model model){
        System.out.println("get /login");
        return "login.html";
    }

    @GetMapping("/signup")
    String singup(Model model){
        System.out.println("get /signup");
        model.addAttribute("memberFormDto", MemberFormDto.empty());
        return "signup.html";
    }

    @PostMapping("/signup")
    String post_signup(@Valid @ModelAttribute MemberFormDto memberFormDto, BindingResult bindingResult){
        System.out.println("Post /signup");
        // 입력값 검증 실패 시 저장하지 않고, 입력한 값과 오류 메시지를 담아 회원가입 화면으로 돌아간다
        if (bindingResult.hasErrors()) {
            System.out.println("⚠️ [Signup Validation] 입력값 오류 " + bindingResult.getErrorCount() + "건");
            return "signup.html";
        }
        try {
            memberService.join(memberFormDto);
        } catch (IllegalStateException | DataIntegrityViolationException e){
            // DataIntegrityViolationException: 가입 버튼을 두 번 눌러 두 요청이 중복 확인을 함께 통과하면, 두 번째 저장이 DB unique 제약에 걸린다 (500 대신 중복 안내)
            return "redirect:/signup?error=duplicate";
        }
        return "redirect:/";
    }

    @GetMapping("/mypage")
    String mypage(Authentication authentication, Model model){
        System.out.println("get /mypage");
        LoginMemberDto loginMemberDto = memberService.getLoginMemberInfo(authentication.getName());
        model.addAttribute("loginMemberDto", loginMemberDto);
        return "mypage.html";
    }
}
