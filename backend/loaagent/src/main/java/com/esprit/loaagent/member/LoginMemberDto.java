package com.esprit.loaagent.member;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class LoginMemberDto {
    private Long id;
    private String username;
    private String role;
}
