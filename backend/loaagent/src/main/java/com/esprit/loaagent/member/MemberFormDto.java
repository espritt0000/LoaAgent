package com.esprit.loaagent.member;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record MemberFormDto(
        @NotBlank(message = "아이디를 입력해주세요.")
        @Size(min = 4, max = 20, message = "아이디는 4~20자로 입력해주세요.")
        @Pattern(regexp = "^[a-zA-Z0-9]*$", message = "아이디는 영문과 숫자만 사용할 수 있습니다.")
        String username,

        // BCryptPasswordEncoder는 72바이트를 넘는 비밀번호에 예외를 던진다.
        // 한글은 1자에 3바이트라 글자 수만으로는 막을 수 없으므로, 영문·숫자·특수문자(1자 1바이트)만 허용하고 64자로 제한한다
        @NotBlank(message = "비밀번호를 입력해주세요.")
        @Size(min = 8, max = 64, message = "비밀번호는 8~64자로 입력해주세요.")
        @Pattern(regexp = "^[\\x21-\\x7E]*$", message = "비밀번호는 영문, 숫자, 특수문자만 사용할 수 있습니다. (공백·한글 불가)")
        String password,

        @NotBlank(message = "이메일을 입력해주세요.")
        @Email(message = "이메일 형식이 올바르지 않습니다.")
        String email) {

    // 회원가입 화면을 처음 열 때 쓰는 빈 폼
    public static MemberFormDto empty() {
        return new MemberFormDto("", "", "");
    }
}
