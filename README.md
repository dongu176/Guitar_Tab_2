# YouTube 기타 TAB 영상 → score.json → SVG / PDF

```
video → 구간별 TAB box → 적응형 샘플링 → 월드 좌표 마디선 레지스트리 → raw 마디
      → Gemini TAB 인식 → raw_recognition.json → 중복 제거/순서 복원 → score.json
      → 검증 → SVG → PDF
```

`score.json` 이 단일 진실 원천입니다. SVG/PDF 는 `score.json` 만으로 만들어지며 영상이나 Gemini 결과를 직접 쓰지 않습니다.
PDF 는 SVG 를 변환하지 않고 SVG 와 같은 drawing primitive 로 reportlab 이 직접 그립니다(cairosvg 불필요).

## 파일

| 파일 | 역할 |
|---|---|
| `extract.py` | 영상 → raw_recognition.json → score.json → 검증 → SVG/PDF |
| `scorelib.py` | 스키마, 검증, 마디 레이아웃, SVG, PDF (영상/AI 의존성 없음) |
| `render_score.py` | 기존 score.json 만으로 SVG/PDF 재생성 |
| `ci_check.py` | 산출물/스키마 검증 (GitHub Actions 에서 사용) |
| `tests/selftest.py` | 합성 영상 셀프테스트 (네트워크/Gemini 불필요) |
| `.github/workflows/extract-score.yml` | **TAB Score Extraction** workflow |
| `.github/workflows/selftest.yml` | push 시 셀프테스트 |

## 로컬 실행

```bash
pip install -r requirements.txt      # ffmpeg 는 시스템에 설치되어 있어야 합니다(코덱 변환 시 사용)

python extract.py ./video.mp4                                   # 로컬 영상
python extract.py "https://www.youtube.com/watch?v=..."         # YouTube
```

### Gemini 사용 / 미사용

```bash
export GEMINI_API_KEY="..."
python extract.py ./video.mp4                # RECOGNIZE=1(기본): 마디마다 TAB 숫자 인식

RECOGNIZE=0 python extract.py ./video.mp4    # Gemini 를 호출하지 않음. API 키 불필요
```

`RECOGNIZE=1` 인데 키가 없으면 `ERROR: GEMINI_API_KEY is required when RECOGNIZE=1` 로 바로 종료합니다.
`RECOGNIZE=0` 이면 마디 구조(위치/순서/시간)만 복원하고, 마디 내용은 비어 있는 `unrecognized` 마디로 남습니다(숫자를 지어내지 않습니다).
인식에 실패한 마디는 `fret: null` + 낮은 confidence 로 보존되고 `raw_recognition.json` 의 `recognition: "failed"` 로 표시됩니다.

### EXPECTED_MEASURES (선택)

```bash
python extract.py ./video.mp4                          # 입력하지 않아도 정상 동작
EXPECTED_MEASURES=112 python extract.py ./video.mp4    # 검증 힌트로만 사용
```

- 입력하지 않으면: `Expected measures: not provided`, `Missing measures: not checked`. 중단/오류 없음.
- 입력하면: `Missing measures` 를 계산하고, 다르면 `incomplete extraction` 경고와 `possible_missing_regions` 를 출력합니다.
- 빈 문자열(`EXPECTED_MEASURES=""`)도 "입력 안 함"으로 처리합니다. 정답(ground truth)이 아니라 힌트입니다.

### 기존 score.json 재렌더링 (영상/AI 재실행 불필요)

```bash
python render_score.py output/score.json --per-line 4
```

### 그 밖의 환경변수 (기본값)

`SAMPLE_SEC=0.15` `SEGMENT_SEC=60` `MEASURES_PER_LINE=0(자동)` `LABEL_CORR_MIN=0.9`
`TIME_SIG=4/4`(영상에서 박자표를 읽지 않으므로 가정값) `TAB_TITLE` `TAB_COMPOSER` `OUT_DIR=output`
빈 값으로 두면 기본값을 씁니다.

## 산출물

```
output/
├── raw_recognition.json      # 영상에서 직접 얻은 결과 (마디별 시간/프레임/구간/인식 결과/중복 판정)
├── score.json                # 최종 악보 (Single Source of Truth)
├── score.schema.json
├── score_validation.json     # 스키마 + 마디/음/박자 검증 결과
├── score.svg                 # 모든 페이지를 세로로 이어 보는 미리보기
├── svg_pages/page_XX.svg
├── score.pdf
└── debug/ (segments, raw_measures, deduplicated_measures, missing_regions)
```

## GitHub Actions 로 실행

1. **Secret 등록** — repository → Settings → Secrets and variables → Actions → *New repository secret*
   - `GEMINI_API_KEY`: Gemini API 키 (`recognize=1` 일 때 필요, 코드/로그에 노출되지 않음)
   - `YOUTUBE_COOKIES` (선택): YouTube 가 GitHub runner 를 봇으로 차단할 때 쓰는 Netscape 형식 cookies.txt 내용
2. **Actions** 탭 → **TAB Score Extraction** → **Run workflow**
3. 입력
   - `mode`: `extract`(영상 → 전체) / `render`(기존 score.json → SVG/PDF)
   - `video_url`: YouTube URL, 또는 `video_path`: repo 안의 영상 경로(예: `input/video.mp4`)
     - 둘 다 비어 있으면 오류, **둘 다 있으면 `video_path` 가 우선**(경고 출력)
   - `recognize`: `1`(Gemini 사용) / `0`(키 없이 구조만)
   - `expected_measures`: **비워 두면 됩니다**(선택)
   - `sample_sec` / `segment_sec` / `label_corr_min` / `measures_per_line`: 기본값 그대로 두면 됩니다
   - `score_json`: `render` 모드에서만 사용(repo 안 경로, 기본 `output/score.json`)
4. 실행이 끝나면 run 페이지 하단 **Artifacts → `tab-score-output`** 을 내려받습니다
   (`score.pdf`, `score.svg`, `score.json`, `raw_recognition.json`, `score_validation.json`, `debug/`, `run.log`). 실패해도 가능한 파일은 업로드됩니다(보관 14일).
5. run 의 **Summary** 탭에 요약 표(Expected/Raw/After dedup/Final/Missing/Validation/SVG/PDF)가 표시됩니다.

큰 영상 파일을 Git 에 커밋하는 것은 권장하지 않습니다. 기본은 `video_url` 입니다. `input/` 은 작은 영상용입니다.

`render` 모드로 기존 결과를 다시 그리려면 이전 artifact 의 `score.json` 을 repo 에 커밋(예: `input/score.json`)한 뒤 `score_json=input/score.json` 으로 실행하세요.

## 한계 / 주의

- 마디 번호는 시간·인접 순서로 1..N 을 새로 부여합니다. 영상에 적힌 번호는 `recognized_measure_number` 에 보조 정보로만 보존됩니다.
- `position` 은 모든 컬럼의 duration 이 확인되고 합이 박자와 정확히 같을 때만 계산합니다. 아니면 `null` 입니다.
- 마디 번호 라벨이 없고 내용이 완전히 같은 페이지로 넘어가는 영상은 정지 화면과 구분하지 못합니다.
- 실제 YouTube + Gemini 조합의 정확도는 영상마다 다릅니다. 로그의 `✂️`(추적 끊김)과 `possible_missing_regions` 를 확인하세요.
