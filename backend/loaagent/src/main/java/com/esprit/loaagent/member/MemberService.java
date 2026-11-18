package com.esprit.loaagent.member;

import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class MemberService {
    private final MemberRepository memberRepository;
    private final PasswordEncoder passwordEncoder;

    public Long join(MemberFormDto memberFormDto){
        validateDuplicateMember(memberFormDto.username());
        Member member = Member.builder()
                .username(memberFormDto.username())
                .password(passwordEncoder.encode(memberFormDto.password()))
                .role("USER")
                .email(memberFormDto.email())
                .build();
        memberRepository.save(member);
        return member.getId();
    }

    private void validateDuplicateMember(String username){
        memberRepository.findByUsername(username).ifPresent(m -> {
            throw new IllegalStateException("이미 존재하는 회원입니다.");
        });
    }

    public LoginMemberDto getLoginMemberInfo(String username){
        Member member = memberRepository.findByUsername(username)
                .orElseThrow(()-> new IllegalArgumentException("존재하지 않는 회원입니다."));
        return new LoginMemberDto(
                member.getId(),
                member.getUsername(),
                member.getRole()
        );
    }
}
