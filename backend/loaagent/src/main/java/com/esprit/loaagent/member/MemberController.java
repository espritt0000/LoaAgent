package com.esprit.loaagent.member;

import lombok.RequiredArgsConstructor;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
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
    String singup(){
        System.out.println("get /signup");
        return "signup.html";
    }

    @PostMapping("/signup")
    String post_signup(MemberFormDto memberFormDto){
        System.out.println("Post /signup");
        try {
            memberService.join(memberFormDto);
        } catch (IllegalStateException e){
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
