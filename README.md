# 🎵 YouTube 합창 악보 추출기

YouTube 합창 악보 영상에서 악보 프레임만 자동 추출하고, 중복 제거 후 인터랙티브 크롭과 A4 PDF/PNG 내보내기를 지원하는 Streamlit 웹앱입니다.

## 주요 기능

- **자동 프레임 추출**: YouTube URL 입력만으로 영상에서 악보 프레임을 자동 감지
- **중복 제거**: Perceptual Hash 기반으로 유사한 프레임 자동 제거
- **인터랙티브 크롭**: 전체/개별 크롭 설정으로 원하는 영역만 추출
- **다양한 내보내기**: PNG ZIP, A4 자동 배치 PDF, 개별 PDF

## 설치 및 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 실행
streamlit run app.py
```

## 기술 스택

| 기술 | 용도 |
|------|------|
| Python 3.10+ | 런타임 |
| Streamlit | 웹 UI |
| yt-dlp | YouTube 영상 다운로드 |
| OpenCV | 프레임 추출 및 악보 감지 |
| imagehash | 중복 프레임 제거 (Perceptual Hash) |
| Pillow | 이미지 크롭 및 프리뷰 |
| fpdf2 | PDF 생성 |

## 사용법

1. 사이드바에 YouTube URL을 입력합니다
2. 프레임 추출 간격, 밝기 임계값, 중복 제거 임계값을 조절합니다
3. **악보 추출 시작** 버튼을 클릭합니다
4. 추출된 프레임에서 원하는 프레임을 선택합니다
5. 크롭 설정으로 불필요한 영역을 제거합니다
6. PNG ZIP 또는 PDF로 내보냅니다

## 악보 감지 원리

1. **밝기 분석**: 프레임 내 밝은 영역(>200) 비율 확인
2. **수평선 감지**: OpenCV morphological operation으로 오선(staff line) 패턴 검출
   - `adaptiveThreshold` → 수평 커널로 `MORPH_OPEN` → 수평선 행 카운트
   - 3개 이상의 distinct 수평선 감지 시 악보로 판단
