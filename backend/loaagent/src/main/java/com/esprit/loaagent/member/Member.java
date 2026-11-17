package com.esprit.loaagent.member;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "member")
public class Member {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true, nullable = false)
    private  String username;

    @Column(nullable = false)
    private String password;

    private String role;
    private String email;

    @Builder
    public Member(String username, String password, String role, String email){
        this.username = username;
        this.password = password;
        this.role = role;
        this.email = email;
    }
}
