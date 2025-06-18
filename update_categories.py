#!/usr/bin/env python3
"""
하루케글 폴더 안의 모든 .md 파일의 categories를 "하루케글"로 변경하는 스크립트
"""

import os
import re

def update_categories_in_file(file_path):
    """파일의 categories를 "하루케글"로 변경"""
    try:
        # 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='cp949') as f:
                content = f.read()
        except:
            print(f"파일 읽기 실패: {file_path}")
            return False
    
    # front matter에서 categories 부분 찾기 및 변경
    # 패턴: categories: 다음 줄에 - 1일1케글 또는 - 기존카테고리
    pattern = r'(categories:\s*\n\s*-\s*)[^\n]+(\n)'
    replacement = r'\g<1>하루케글\g<2>'
    
    new_content = re.sub(pattern, replacement, content)
    
    # 변경사항이 있는지 확인
    if new_content != content:
        try:
            # 파일 쓰기
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        except Exception as e:
            print(f"파일 쓰기 실패 {file_path}: {e}")
            return False
    else:
        return False

def main():
    folder_path = "_posts/하루케글"
    
    if not os.path.exists(folder_path):
        print(f"폴더가 존재하지 않습니다: {folder_path}")
        return
    
    updated_count = 0
    total_count = 0
    
    # 폴더 안의 모든 .md 파일 처리
    for filename in os.listdir(folder_path):
        if filename.endswith('.md'):
            total_count += 1
            file_path = os.path.join(folder_path, filename)
            
            if update_categories_in_file(file_path):
                print(f"업데이트 완료: {filename}")
                updated_count += 1
            else:
                print(f"업데이트 필요 없음 또는 실패: {filename}")
    
    print(f"\n총 {total_count}개 파일 중 {updated_count}개 파일 업데이트 완료")

if __name__ == "__main__":
    main() 