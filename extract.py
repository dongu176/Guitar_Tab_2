"""extract.py - YouTube 기타 TAB 영상 → Score JSON → SVG/PDF

video → TAB 영역(구간별) → 프레임 추적(월드 좌표 마디선 레지스트리) → 마디 단위 raw 인식
      → raw_recognition.json → 중복 제거/순서 복원 → score.json → 검증 → 레이아웃 → SVG → PDF

사용법:
    python extract.py "https://www.youtube.com/watch?v=..."
    python extract.py ./local_video.mp4          # 로컬 파일도 가능
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

try:
    import yt_dlp
except ImportError:
    yt_dlp = None
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

import scorelib as sl


# ------------------------------------------------------------------
# 설정 (환경변수로 조절)
# ------------------------------------------------------------------
def _f(name, default):
    v = os.environ.get(name, "").strip()
    try:
        return float(v) if v else float(default)
    except ValueError:
        print(f"⚠️ 환경변수 {name}={v!r} 를 숫자로 읽을 수 없어 기본값 {default} 사용")
        return float(default)


def _i(name, default):
    return int(_f(name, default))


def _opt_int(name):
    """선택 입력. 없음/빈 값/잘못된 값 → None."""
    v = os.environ.get(name, "").strip()
    if not v:
        return None
    try:
        n = int(float(v))
    except ValueError:
        print(f"⚠️ 환경변수 {name}={v!r} 를 정수로 읽을 수 없어 무시합니다.")
        return None
    return n if n > 0 else None


OUT_DIR = os.environ.get("OUT_DIR", "output")
EXPECTED_MEASURES = _opt_int("EXPECTED_MEASURES")   # 선택 사항(검증 힌트). None 이면 개수 검증을 하지 않는다
SAMPLE_SEC = _f("SAMPLE_SEC", 0.15)              # 기본 샘플 간격. 스크롤이 빠르면 자동으로 촘촘해짐
SEGMENT_SEC = _f("SEGMENT_SEC", 60)               # TAB 영역을 따로 탐지하는 시간 구간 길이
VIDEO_HEIGHT = _i("VIDEO_HEIGHT", 1080)
TAB_TITLE = os.environ.get("TAB_TITLE", "").strip()
TAB_COMPOSER = os.environ.get("TAB_COMPOSER", "").strip()
TIME_SIG = os.environ.get("TIME_SIG", "").strip() or "4/4"      # 영상에서 박자표를 읽지 않으므로 가정값
MEASURES_PER_LINE = _i("MEASURES_PER_LINE", 0)    # 0 = 자동
RECOGNIZE = os.environ.get("RECOGNIZE", "1").strip() not in ("0", "false", "False", "no")
GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()
_env_model = os.environ.get("GEMINI_MODEL", "").strip()
MODEL_CANDIDATES = ([_env_model] if _env_model else []) + [
    "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash",
    "gemini-3.6-flash", "gemini-3-flash-preview",
]
AI_BOX_BUDGET_SEC = _f("AI_BUDGET_SEC", 150)
RECOG_SLEEP = _f("RECOG_SLEEP", 0.4)
RECOG_MAX_FAILS = _i("RECOG_MAX_FAILS", 6)

MATCH_MIN_SCORE = 0.6          # 프레임 간 겹침 상관 최소값
MAX_SCROLL_FRAC = 0.45         # 샘플 간 최대 스크롤(화면 폭 비율)
NO_STAFF_CLOSE_SEC = 1.0       # 오선이 이만큼 안 보이면 추적 구간 종료
MIN_SCROLL_PX_SEC = 20.0       # 이보다 느리면 '정지(페이지 넘김)'로 간주
MAX_BREAK_SEC = 1.5            # 구간 경계 중복 판정을 허용하는 최대 시간 간격
HASH_SAME = 0.30               # 같은 마디 모양으로 보는 최대 거리(1-상관)
MAX_FAILS = 3                  # 연속 매칭 실패가 이 횟수에 이르면 새 추적 구간
LABEL_CORR_MIN = _f("LABEL_CORR_MIN", 0.9)   # 마디 번호 라벨 영역 상관이 이보다 낮으면 화면 전환으로 간주

DBG = os.path.join(OUT_DIR, "debug")


class Stages:
    """[n/N] 단계 로그 + 단계별 소요 시간."""

    def __init__(self, total):
        self.total, self.t, self.name, self.times = total, None, None, {}

    def start(self, n, name):
        self.end()
        self.name, self.t = name, time.time()
        print(f"[{n}/{self.total}] {name}", flush=True)

    def end(self):
        if self.t is not None:
            dt = time.time() - self.t
            self.times[self.name] = round(dt, 1)
            print(f"    ↳ {dt:.1f}s", flush=True)
            self.t = None


try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def to_py(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=to_py)


def rel(path):
    return os.path.relpath(path, OUT_DIR).replace(os.sep, "/")


def save_ink_png(ink, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, 255 - ink)


# ------------------------------------------------------------------
# 1. 영상 다운로드 / 열기
# ------------------------------------------------------------------
def is_url(src):
    return src.lower().startswith(("http://", "https://"))


def _yt_opts(use_cookies):
    opts = {
        "format": f"bv*[height<={VIDEO_HEIGHT}][vcodec^=avc1]/bv*[height<={VIDEO_HEIGHT}]/b[height<={VIDEO_HEIGHT}]/b",
        "outtmpl": "video.%(ext)s", "noplaylist": True, "quiet": True, "no_warnings": True, "retries": 5,
    }
    if use_cookies:
        opts["cookiefile"] = "cookies.txt"
    return opts


def download_video(src):
    if os.path.exists(src):
        return src, os.path.splitext(os.path.basename(src))[0]
    if not is_url(src):
        raise RuntimeError(f"입력 파일을 찾을 수 없습니다: {src}")
    if yt_dlp is None:
        raise RuntimeError("yt-dlp 가 설치되어 있지 않습니다. (pip install yt-dlp)")
    try:
        print(f"yt-dlp version: {yt_dlp.version.__version__}")
    except Exception:
        pass
    for f in glob.glob("video.*"):
        os.remove(f)
    has_cookies = os.path.exists("cookies.txt") and os.path.getsize("cookies.txt") > 0
    # 쿠키가 있으면 먼저 쓰고, 실패하면 쿠키 없이 다시 시도한다(쿠키가 오히려 오류를 내는 경우가 있음)
    attempts = ([True] if has_cookies else []) + [False]
    errors, info = [], None
    for use_cookies in attempts:
        try:
            with yt_dlp.YoutubeDL(_yt_opts(use_cookies)) as ydl:
                info = ydl.extract_info(src, download=True)
            if use_cookies is False and has_cookies:
                print("ℹ️ 쿠키로는 실패해서 쿠키 없이 다운로드했습니다. (YOUTUBE_COOKIES 가 만료/무효일 수 있음)")
            break
        except Exception as e:
            msg = str(e).replace(src, "<input>")
            errors.append(f"{'with cookies' if use_cookies else 'without cookies'}: {msg[:300]}")
            print(f"⚠️ 다운로드 시도 실패 ({'쿠키 사용' if use_cookies else '쿠키 없음'}): {msg[:200]}")
            for f in glob.glob("video.*"):
                try:
                    os.remove(f)
                except OSError:
                    pass
    if info is None:
        raise RuntimeError(" | ".join(errors))
    files = [f for f in glob.glob("video.*") if not f.endswith((".part", ".ytdl"))]
    if not files:
        raise RuntimeError("영상 다운로드에 실패했습니다.")
    return files[0], (info or {}).get("title", "")


def open_video(path):
    cap = cv2.VideoCapture(path)
    ok, _ = cap.read() if cap.isOpened() else (False, None)
    if ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap
    print("⚠️ OpenCV가 코덱을 읽지 못해 ffmpeg로 변환합니다...")
    cap.release()
    conv = "video_h264.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-an", "-c:v", "libx264",
                    "-preset", "ultrafast", "-crf", "28", conv], check=True)
    cap = cv2.VideoCapture(conv)
    if not cap.isOpened():
        raise RuntimeError("영상 파일을 열 수 없습니다.")
    return cap


# ------------------------------------------------------------------
# Gemini 공용 호출 (모델 폐기/한도 시 다음 후보로)
# ------------------------------------------------------------------
_client = None
_model_idx = 0
_use_thinking = True
_ai_dead = False


def gemini_ready():
    global _client
    if not RECOGNIZE or _ai_dead or genai is None or not GEMINI_API_KEY:
        return False
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=60_000))
    return True


def gemini_json(images, prompt):
    """성공하면 파싱된 JSON, 일시 실패면 None. 사용 가능한 모델이 없으면 RuntimeError."""
    global _model_idx, _use_thinking, _ai_dead
    attempt = 0
    while True:
        if _model_idx >= len(MODEL_CANDIDATES):
            _ai_dead = True
            raise RuntimeError("사용 가능한 Gemini 모델/한도가 없습니다.")
        model = MODEL_CANDIDATES[_model_idx]
        try:
            cfg = dict(response_mime_type="application/json", temperature=0,
                       automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
            if _use_thinking:
                try:
                    cfg["thinking_config"] = types.ThinkingConfig(thinking_level="low")
                except Exception:
                    _use_thinking = False
            resp = _client.models.generate_content(
                model=model, contents=list(images) + [prompt],
                config=types.GenerateContentConfig(**cfg))
            return json.loads(resp.text)
        except json.JSONDecodeError:
            return None
        except Exception as e:
            msg, low = str(e), str(e).lower()
            if "404" in msg or "NOT_FOUND" in msg:
                print(f"⚠️ 모델 사용 불가: {model} → 다음 후보")
                _model_idx += 1
                attempt = 0
                continue
            if _use_thinking and "thinking" in low:
                _use_thinking = False
                continue
            if "429" in msg or "resource_exhausted" in low:
                attempt += 1
                if attempt >= 2:
                    print(f"⚠️ {model} 한도 초과 → 다음 모델")
                    _model_idx += 1
                    attempt = 0
                else:
                    time.sleep(10)
                continue
            if any(k in low for k in ("503", "unavailable", "504", "timeout", "timed out", "deadline")):
                attempt += 1
                if attempt >= 3:
                    _model_idx += 1
                    attempt = 0
                else:
                    time.sleep(10)
                continue
            print(f"⚠️ Gemini 예외: {msg[:120]}")
            return None


BOX_PROMPT = """
Identify the guitar TAB (tablature) score area in this video frame.
Return ONLY a JSON object with normalized coordinates (0 to 1000 scale) in this exact format:
{"has_tab": true, "box_2d": [ymin, xmin, ymax, xmax]}
If no TAB score is present, return {"has_tab": false, "box_2d": []}.
"""


def detect_box_ai(pil_img):
    small = pil_img.copy()
    small.thumbnail((1280, 1280))
    data = gemini_json([small], BOX_PROMPT)
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        return None
    box = data.get("box_2d", [])
    if data.get("has_tab") and len(box) == 4:
        ymin, xmin, ymax, xmax = [float(v) for v in box]
        if ymax > ymin and xmax > xmin:
            return [ymin, xmin, ymax, xmax]
    return None


# ------------------------------------------------------------------
# 2. 이미지 처리: 잉크 추출 / 오선 / 마디선
# ------------------------------------------------------------------
def ink_from_bgr(bgr, video_h):
    m = bgr.min(axis=2)
    k = max(9, int(round(video_h * 0.014))) | 1
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    top = cv2.morphologyEx(m, cv2.MORPH_TOPHAT, kern)
    ink = np.clip((top.astype(np.float32) - 18) / 60.0, 0, 1)
    ink[m < 90] = 0
    ink = (ink * 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats((ink > 90).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 6
    ink[~keep[lab]] = 0
    return ink


def staff_extent(ink, min_frac=0.5):
    prof = (ink > 100).mean(axis=1)
    rows = np.where(prof > min_frac)[0]
    if len(rows) == 0:
        return None
    centers, grp = [], [rows[0]]
    for r in rows[1:]:
        if r - grp[-1] <= 2:
            grp.append(r)
        else:
            centers.append(float(np.mean(grp)))
            grp = [r]
    centers.append(float(np.mean(grp)))
    if len(centers) < 4:
        return None
    best = None
    for i in range(len(centers) - 1):
        sp = centers[i + 1] - centers[i]
        j = i + 1
        while j + 1 < len(centers) and 0.7 * sp <= centers[j + 1] - centers[j] <= 1.4 * sp:
            j += 1
        n = j - i + 1
        if n >= 4 and (best is None or n > best[0]):
            best = (n, centers[i], centers[j])
    if best is None:
        return None
    return int(round(best[1])), int(round(best[2]))


def remove_staff_lines(ink):
    out = ink.copy()
    prof = (ink > 100).mean(axis=1)
    for r in np.where(prof > 0.5)[0]:
        out[max(0, r - 1):r + 2, :] = 0
    return out


def detect_bars(ink):
    """(오선 범위, 마디선 x 리스트). 오선이 없으면 (None, [])."""
    ext = staff_extent(ink, 0.5)
    if ext is None:
        return None, []
    top, bot = ext
    w = ink.shape[1]
    cols = (ink[top:bot + 1] > 100).mean(axis=0) > 0.85
    xs, run = [], []
    for x, v in enumerate(cols):
        if v:
            run.append(x)
        elif run:
            xs.append(float(np.mean(run)))
            run = []
    if run:
        xs.append(float(np.mean(run)))
    mdist = max(10.0, 0.008 * w)
    bars = []
    for x in xs:
        if x < 3 or x > w - 4:
            continue
        if not bars or x - bars[-1] > mdist:
            bars.append(x)
    return ext, bars


def crop_box(frame, box, pad=10):
    h, w = frame.shape[:2]
    ymin, xmin, ymax, xmax = box
    return frame[max(0, int(ymin / 1000 * h) - pad):min(h, int(ymax / 1000 * h) + pad),
                 max(0, int(xmin / 1000 * w) - pad):min(w, int(xmax / 1000 * w) + pad)]


def heuristic_box(frame):
    h, w = frame.shape[:2]
    ext = staff_extent(ink_from_bgr(frame, h), 0.5)
    if ext is None:
        return None
    top, bot = ext
    sh = max(1, bot - top)
    return [max(0, top - 0.7 * sh) / h * 1000, 0.0, min(h, bot + 0.9 * sh) / h * 1000, 1000.0]


# ------------------------------------------------------------------
# 3. TAB 영역 탐지: 시간 구간(window)마다 따로
# ------------------------------------------------------------------
def detect_windows(cap, fps, total):
    duration = total / fps
    n = max(1, int(np.ceil(duration / SEGMENT_SEC)))
    boxes, methods = [], []
    ai_t0 = time.time()
    prev = None
    for wi in range(n):
        t0, t1 = wi * SEGMENT_SEC, min(duration, (wi + 1) * SEGMENT_SEC)
        frames = []
        for p in (0.15, 0.4, 0.65, 0.9):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((t0 + (t1 - t0) * p) * fps))
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
        hbs = [b for b in (heuristic_box(fr) for fr in frames) if b]
        box, method = None, "none"
        if len(hbs) >= 2:
            box, method = [float(v) for v in np.median(np.array(hbs), axis=0)], "staff"
        elif hbs:
            box, method = hbs[0], "staff-1"
        if box is None and gemini_ready() and time.time() - ai_t0 < AI_BOX_BUDGET_SEC:
            found = []
            for fr in frames[:3]:
                try:
                    b = detect_box_ai(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                except RuntimeError as e:
                    print(f"⚠️ {e}")
                    break
                if b:
                    found.append(b)
            if found:
                box, method = [float(v) for v in np.median(np.array(found), axis=0)], "gemini"
        if box is not None and prev is not None and max(abs(np.array(box) - np.array(prev))) < 15:
            box, method = prev, method + "(same)"
        boxes.append(box)
        methods.append(method)
        if box is not None:
            prev = box
        if frames:
            vis = frames[0].copy()
            if box is not None:
                h, w = vis.shape[:2]
                cv2.rectangle(vis, (int(box[1] / 1000 * w), int(box[0] / 1000 * h)),
                              (int(box[3] / 1000 * w), int(box[2] / 1000 * h)), (0, 255, 0), 3)
            os.makedirs(os.path.join(DBG, "segments"), exist_ok=True)
            cv2.imwrite(os.path.join(DBG, "segments", f"window_{wi:02d}.jpg"), vis)
        print(f"  window {wi} ({t0:.0f}~{t1:.0f}s): {method} → {box}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not any(b is not None for b in boxes):
        return None, []
    for i in range(n):  # 못 찾은 구간은 가까운 구간의 박스를 빌려온다
        if boxes[i] is None:
            near = min((j for j, b in enumerate(boxes) if b is not None), key=lambda j: abs(j - i))
            boxes[i], methods[i] = boxes[near], "inherited"
    return boxes, methods


# ------------------------------------------------------------------
# 4. 프레임 간 이동량 추정 (후보 생성 + 마디선 일관성 검증 + 속도 사전확률)
# ------------------------------------------------------------------
def prep(ink):
    """(2D 블러 이미지, 열 방향 1D 프로파일). 1D 프로파일로 후보를 빠르게 찾고 2D로 검증한다."""
    img = cv2.GaussianBlur(remove_staff_lines(ink), (3, 3), 0).astype(np.float32)
    prof = cv2.GaussianBlur(img.sum(axis=0).reshape(1, -1), (0, 0), 1.5).astype(np.float32)
    return img, prof


def overlap_corr(pa, pb, dx):
    pa, pb = pa[0], pb[0]
    w = pa.shape[1]
    ow = w - dx
    if ow < 0.5 * w:
        return None
    A = pa[:, dx:dx + ow].ravel()
    B = pb[:, :ow].ravel()
    A = A - A.mean()
    B = B - B.mean()
    den = float(np.sqrt((A * A).sum() * (B * B).sum()))
    if den < 1e-3:
        return None
    return float((A * B).sum() / den)


def bar_agree(pbars, cbars, d, w):
    tol = max(3.0, 0.006 * w)
    mapped = [b - d for b in pbars if 3 <= b - d <= w - 3]
    cov = [c for c in cbars if c + d <= w - 3]
    if not mapped and not cov:
        return None, 0
    matched = sum(1 for m in mapped if any(abs(m - c) <= tol for c in cbars))
    um_p = len(mapped) - matched
    um_c = sum(1 for c in cov if not any(abs((c + d) - b) <= tol for b in pbars))
    return matched / max(1, matched + um_p + um_c), matched


def estimate_dx(pa, pbars, pb, cbars, expected):
    """이전 프레임(pa) → 현재 프레임(pb)의 가로 스크롤 픽셀 수. 실패 시 None.
    template 최고점 하나를 믿지 않고, 여러 후보를 마디선 일치도/속도로 검증해 고른다."""
    w = pa[0].shape[1]
    max_dx = int(w * MAX_SCROLL_FRAC)
    wt = int(w * 0.25)
    cands = {0}
    if expected is not None:
        e = int(round(expected))
        cands.update(d for d in range(e - 2, e + 3) if 0 <= d <= max_dx)
    for a in (0, int(0.35 * w), int(0.6 * w)):
        t = pb[1][:, a:a + wt]
        region = pa[1][:, a:a + max_dx + wt]
        if t.shape[1] < wt or t.std() < 1e-3 or region.shape[1] <= wt:
            continue
        res = np.nan_to_num(cv2.matchTemplate(np.ascontiguousarray(region), np.ascontiguousarray(t),
                                              cv2.TM_CCOEFF_NORMED)[0], nan=-1.0)
        picked = []
        for i in np.argsort(-res):
            if res[i] < 0.4 or len(picked) >= 4:
                break
            if all(abs(int(i) - q) > 3 for q in picked):
                picked.append(int(i))
        cands.update(d for d in picked if 0 <= d <= max_dx)
    for pbx in pbars:
        for cbx in cbars:
            d = int(round(pbx - cbx))
            if 0 <= d <= max_dx:
                cands.add(d)

    best, best_total = None, -1e9
    for d in sorted(cands):
        c = overlap_corr(pa, pb, d)
        bs, matched = bar_agree(pbars, cbars, d, w)
        if c is None and bs is None:
            continue
        total = (c if c is not None else 0.5) + 0.6 * (bs if bs is not None else 0.5)
        if expected is not None:
            total -= 0.25 * min(1.0, abs(d - expected) / max(6.0, 0.5 * abs(expected) + 3))
        total -= 0.02 * d / max(1, max_dx)           # 동점이면 작은 이동량 선호(반복 리프 모호성 완화)
        if total > best_total:
            best_total, best = total, (d, c, bs, matched)
    if best is None:
        return None
    d, c, bs, matched = best
    if c is not None:
        if c < MATCH_MIN_SCORE:
            return None
    elif not (bs is not None and bs >= 0.8 and matched >= 1):
        return None
    if bs is not None and bs < 0.5:
        return None
    return float(d)


def label_changed(ink_a, ink_b, dx, top):
    """오선 위쪽(마디 번호 라벨 영역)이 스크롤로 설명되지 않게 바뀌었는지.
    내용이 같은 페이지로 넘어가도(정지 화면과 구분 불가한 경우) 라벨은 달라지므로 화면 전환을 잡아낸다.
    라벨이 없거나 너무 작으면(잉크 25px 미만) 판단하지 않는다."""
    r = top - 3
    if r < 6:
        return False
    w = ink_a.shape[1]
    dx = int(dx)
    ow = w - dx
    if ow < 0.5 * w:
        return False
    A = ink_a[:r, dx:dx + ow].astype(np.float32)
    B = ink_b[:r, :ow].astype(np.float32)
    if max(int((A > 100).sum()), int((B > 100).sum())) < 25:
        return False
    A = cv2.GaussianBlur(A, (3, 3), 0).ravel()
    B = cv2.GaussianBlur(B, (3, 3), 0).ravel()
    A, B = A - A.mean(), B - B.mean()
    den = float(np.sqrt((A * A).sum() * (B * B).sum()))
    if den < 1e-3:
        return True            # 한쪽에만 라벨이 있음 → 변화
    return float((A * B).sum() / den) < LABEL_CORR_MIN


class Seg:
    """연속 스크롤되는 하나의 추적 구간. 마디선을 '월드 좌표'로 레지스트리에 등록한다.
    마디의 정체성은 내용이 아니라 월드 좌표 상의 (왼쪽 마디선, 오른쪽 마디선) 쌍이다."""

    def __init__(self, sid, box):
        self.id, self.box = sid, box
        self.X, self.vel, self.nupd = 0.0, 0.0, 0
        self.prev = None
        self.clusters = []         # [world_x, n]
        self.tracks = {}
        self.t_start = self.t_end = None
        self.n_frames = 0
        self.last_dx = 0.0
        self.w = None

    def _cluster(self, wx, tol):
        best, bd = None, 1e18
        for i, c in enumerate(self.clusters):
            d = abs(c[0] - wx)
            if d < bd:
                best, bd = i, d
        return best if best is not None and bd <= tol else None

    def observe(self, ink, ext, bars, idx, t):
        h, w = ink.shape
        top, bot = ext
        tol = max(8.0, 0.03 * w)
        snap = []
        for b in bars:
            ci = self._cluster(b + self.X, tol)
            if ci is not None:
                snap.append(self.clusters[ci][0] - (b + self.X))
        if len(snap) >= 2:
            self.X += float(np.median(snap))
        cids = []
        for b in bars:
            wx = b + self.X
            ci = self._cluster(wx, tol)
            if ci is None:
                self.clusters.append([wx, 1])
                ci = len(self.clusters) - 1
            else:
                c = self.clusters[ci]
                c[0] = (c[0] * c[1] + wx) / (c[1] + 1)
                c[1] += 1
            cids.append(ci)
        sh = bot - top
        y0, y1 = max(0, top - int(0.8 * sh)), min(h, bot + int(0.8 * sh))
        for i in range(len(bars) - 1):
            a, b = cids[i], cids[i + 1]
            if a == b:
                continue
            x0, x1 = bars[i], bars[i + 1]
            q = 1 - abs((x0 + x1) / 2 - w / 2) / (w / 2)
            tr = self.tracks.get((a, b))
            if tr is None:
                tr = {"a": a, "b": b, "first_t": t, "first_idx": idx, "first_x0": x0, "first_x1": x1,
                      "count": 0, "best_q": -9.0}
                self.tracks[(a, b)] = tr
            tr["last_t"], tr["last_x0"], tr["last_x1"] = t, x0, x1
            tr["count"] += 1
            if q > tr["best_q"]:
                xa, xb = max(0, int(x0) - 2), min(w, int(x1) + 3)
                tr.update(best_q=q, best_t=t, best_idx=idx,
                          bbox=[float(xa), float(y0), float(xb), float(y1)],
                          ink=ink[y0:y1, xa:xb].copy())

    def add(self, ink, ext, bars, idx, t):
        pa = prep(ink)
        if self.prev is None:
            self.t_start = t
        else:
            dt = t - self.prev["t"]
            expected = self.vel * dt if self.nupd >= 2 else None
            dx = estimate_dx(self.prev["pa"], self.prev["bars"], pa, bars, expected)
            if dx is None:
                return False
            if label_changed(self.prev["ink"], ink, dx, min(self.prev["top"], ext[0])):
                return False
            self.X += dx
            self.last_dx = dx
            if dt > 0:
                v = dx / dt
                self.vel = v if self.nupd == 0 else 0.6 * self.vel + 0.4 * v
            self.nupd += 1
        self.observe(ink, ext, bars, idx, t)
        self.prev = {"pa": pa, "bars": bars, "t": t, "ink": ink, "top": ext[0]}
        self.t_end = t
        self.n_frames += 1
        self.w = ink.shape[1]
        return True


# ------------------------------------------------------------------
# 5. 영상 전체 스캔 (적응형 샘플링)
# ------------------------------------------------------------------
def scan_video(cap, fps, windows):
    base = max(1, int(round(fps * SAMPLE_SEC)))
    step = base
    segs, cur = [], None
    no_staff_start, no_staff = None, []
    idx, next_idx, sid = 0, 0, 0
    fail_count = 0
    samples = 0
    last_t = 0.0
    while True:
        if not cap.grab():
            break
        if idx >= next_idx:
            ok, frame = cap.retrieve()
            next_idx = idx + step
            if ok:
                t = idx / fps
                last_t = t
                samples += 1
                box = windows[min(int(t / SEGMENT_SEC), len(windows) - 1)]
                crop = crop_box(frame, box)
                ink = ink_from_bgr(crop, frame.shape[0]) if crop.size > 0 else None
                ext, bars = detect_bars(ink) if ink is not None else (None, [])
                if ext is None:
                    if no_staff_start is None:
                        no_staff_start = t
                    if cur is not None and t - cur.t_end >= NO_STAFF_CLOSE_SEC:
                        segs.append(cur)
                        cur = None
                else:
                    if no_staff_start is not None:
                        if t - no_staff_start >= 2.0:
                            no_staff.append((no_staff_start, t))
                        no_staff_start = None
                    if cur is not None and cur.box is not box:
                        segs.append(cur)
                        cur = None
                    if cur is None:
                        cur = Seg(sid, box)
                        sid += 1
                        fail_count = 0
                        cur.add(ink, ext, bars, idx, t)
                        print(f"📌 구간 {cur.id} 시작 ({t:.1f}s)")
                    elif cur.add(ink, ext, bars, idx, t):
                        fail_count = 0
                        r = cur.last_dx / max(1, ink.shape[1])
                        if r > 0.15:
                            step = max(1, step // 2)
                        elif r < 0.04:
                            step = min(base, step * 2)
                    else:
                        # 일시적 글리치일 수 있으므로 같은 구간을 유지한 채 다음 프레임을 바로 다시 시도한다
                        fail_count += 1
                        next_idx = idx + 1
                        if fail_count >= MAX_FAILS:
                            segs.append(cur)
                            print(f"✂️ 추적 끊김 → 새 구간 ({t:.1f}s)")
                            cur = Seg(sid, box)
                            sid += 1
                            cur.add(ink, ext, bars, idx, t)
                            fail_count = 0
                            step = max(1, base // 2)
        idx += 1
    if cur is not None:
        segs.append(cur)
    segs = [s for s in segs if s.n_frames > 0]
    if no_staff_start is not None and last_t - no_staff_start >= 2.0:
        no_staff.append((no_staff_start, last_t))
    return segs, no_staff, {"frames_read": idx, "samples": samples, "duration": idx / fps}


# ------------------------------------------------------------------
# 6. Raw 마디 목록 (구간별, 월드 좌표 순서)
# ------------------------------------------------------------------
def small_vec(ink):
    return cv2.resize(ink.astype(np.float32), (96, 24), interpolation=cv2.INTER_AREA).ravel()


def vec_dist(a, b):
    if a.std() < 1e-3 or b.std() < 1e-3:
        return 1.0
    return float(1 - np.corrcoef(a, b)[0, 1])


def build_raw(segs):
    raws, breaks = [], []
    for seg in segs:
        trs = sorted(seg.tracks.values(), key=lambda tr: seg.clusters[tr["a"]][0])
        if not trs:
            continue
        widths = [tr["bbox"][2] - tr["bbox"][0] for tr in trs]
        med = float(np.median(widths))
        prev = None
        for tr, wd in zip(trs, widths):
            if wd < 0.25 * med:
                continue                         # 겹마디선/잡티로 생긴 가짜 마디
            raw = {
                "raw_id": f"raw_{len(raws) + 1:06d}",
                "segment_id": seg.id,
                "timestamp": float(tr["best_t"]), "frame_index": int(tr["best_idx"]),
                "t_first": float(tr["first_t"]), "t_last": float(tr["last_t"]),
                "observations": int(tr["count"]),
                "world_x0": float(seg.clusters[tr["a"]][0]), "world_x1": float(seg.clusters[tr["b"]][0]),
                "bbox": tr["bbox"], "width_ratio": float(wd / med),
                "_q": tr["best_q"], "_ink": tr["ink"], "_vec": small_vec(tr["ink"]),
                "_first_x0": tr["first_x0"], "_last_x0": tr["last_x0"],
                "_b": tr["b"], "_a": tr["a"],
            }
            if prev is not None and prev["_b"] != tr["a"]:
                gap = seg.clusters[tr["a"]][0] - seg.clusters[prev["_b"]][0]
                if gap > 0.5 * med:
                    breaks.append({"kind": "chain_break", "segment_id": seg.id,
                                   "after_raw": prev["raw_id"], "before_raw": raw["raw_id"],
                                   "t_from": prev["t_last"], "t_to": raw["t_first"],
                                   "gap_px": float(gap), "approx_missing": int(round(gap / med))})
            raws.append(raw)
            prev = raw
    return raws, breaks


# ------------------------------------------------------------------
# 7. Gemini 마디 인식 (보이는 것만, 추측 금지)
# ------------------------------------------------------------------
REC_PROMPT = """You are reading ONE measure of guitar TAB from an image (white background, black ink, 6 lines).
The TOP line is string 1 (highest pitch) and the BOTTOM line is string 6 (lowest).
Return ONLY JSON in exactly this format:
{"measure_label": <small measure number printed above the staff, or null>,
 "confidence": <0..1 overall>,
 "columns": [
   {"x": <horizontal center of this column as fraction of image width, 0..1>,
    "duration": "1/1"|"1/2"|"1/4"|"1/8"|"1/16"|"1/32"|null,
    "rest": false,
    "tie": null|"start"|"stop"|"continue",
    "notes": [{"string": 1-6, "fret": <integer or null>, "muted": false,
               "technique": null|"hammer_on"|"pull_off"|"slide_up"|"slide_down"|"bend"|"release"|"vibrato"|"harmonic"|"palm_mute"|"let_ring",
               "grace": false, "confidence": <0..1>}]}]}
Rules:
- Columns are ordered left to right. Notes on the same vertical line form one column (a chord).
- Report ONLY digits that are clearly visible. If a digit is unclear use "fret": null and a low confidence. NEVER guess.
- "X" on a line means muted: "muted": true and "fret": null.
- "duration" only if stems/beams/flags make it clear, otherwise null.
- If the image has no readable TAB content, return {"measure_label": null, "confidence": 0, "columns": []}.
"""


def prep_for_ai(ink):
    gray = 255 - ink
    s = float(np.clip(1000.0 / max(1, gray.shape[1]), 1.0, 3.0))
    img = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return Image.fromarray(img).convert("RGB")


def _clamp01(v, default=0.0):
    try:
        return float(min(1.0, max(0.0, float(v))))
    except Exception:
        return default


def columns_to_events(data, mnum, ts):
    """Gemini 결과를 검증/정규화해 events 로 만든다. 확신 없는 값은 null 로 둔다."""
    cols = data.get("columns") if isinstance(data, dict) else None
    if not isinstance(cols, list):
        return [], False
    mlen = sl.Fraction(ts["numerator"], ts["denominator"])
    cleaned = []
    for ci, col in enumerate(cols):
        if not isinstance(col, dict):
            continue
        x = _clamp01(col.get("x"), (ci + 0.5) / max(1, len(cols)))
        dur = col.get("duration")
        dur = dur if (isinstance(dur, str) and sl.valid_duration(dur)) else None
        tie = col.get("tie") if col.get("tie") in sl.TIES else None
        notes = []
        for n in (col.get("notes") or []):
            if not isinstance(n, dict):
                continue
            s = n.get("string")
            if not (isinstance(s, int) and not isinstance(s, bool) and 1 <= s <= 6):
                continue
            fr = n.get("fret")
            conf = _clamp01(n.get("confidence"), 0.3)
            if not (isinstance(fr, int) and not isinstance(fr, bool) and 0 <= fr <= 30):
                fr = None
                conf = min(conf, 0.35)
            tech = n.get("technique") if n.get("technique") in sl.TECHNIQUES else None
            notes.append({"string": s, "fret": fr, "muted": bool(n.get("muted")), "technique": tech,
                          "grace": bool(n.get("grace")), "confidence": conf})
        cleaned.append({"x": x, "duration": dur, "rest": bool(col.get("rest")), "tie": tie, "notes": notes})
    cleaned = [c for c in cleaned if c["rest"] or c["notes"]]
    # 모든 컬럼의 duration 이 확인되고 합이 박자와 정확히 같을 때만 position 을 계산한다(추측 금지)
    rhythm_known = bool(cleaned) and all(c["duration"] for c in cleaned)
    positions = [None] * len(cleaned)
    if rhythm_known:
        cur = sl.Fraction(0)
        for i, c in enumerate(cleaned):
            positions[i] = cur
            cur += sl.parse_frac(c["duration"])
        if cur != mlen:
            rhythm_known, positions = False, [None] * len(cleaned)
    events = []
    for ci, c in enumerate(cleaned):
        pos = sl.frac_str(positions[ci]) if positions[ci] is not None else None
        base = {"position": pos, "duration": c["duration"], "x_norm": c["x"], "tie": c["tie"]}
        if c["rest"] and not c["notes"]:
            events.append({"id": f"{mnum}_e{len(events) + 1}", "type": "rest", **base, "string": None,
                           "fret": None, "technique": None, "modifiers": [], "chord_id": None,
                           "confidence": 0.5})
            continue
        chord = f"{mnum}_c{ci + 1}" if len(c["notes"]) > 1 else None
        for n in c["notes"]:
            typ = "mute" if n["muted"] else ("grace_note" if n["grace"] else "note")
            events.append({"id": f"{mnum}_e{len(events) + 1}", "type": typ, **base,
                           "string": n["string"], "fret": None if n["muted"] else n["fret"],
                           "technique": n["technique"], "modifiers": [], "chord_id": chord,
                           "confidence": n["confidence"]})
    return events, rhythm_known


def recognize_all(raws, ts):
    fails, calls = 0, 0
    use_ai = RECOGNIZE and gemini_ready()
    if not use_ai:
        print("⚠️ 인식 비활성(RECOGNIZE=0 또는 GEMINI_API_KEY/라이브러리 없음) → 모든 마디를 'unrecognized'로 둡니다.")
    for i, r in enumerate(raws):
        r.update(recognized_measure_number=None, notes=[], confidence=0.0, status="unrecognized",
                 rhythm_known=False, recognition="skipped", recognition_error=None)
        img_path = os.path.join(DBG, "raw_measures", f"{r['raw_id']}.png")
        save_ink_png(r["_ink"], img_path)
        r["image_path"] = rel(img_path)
        if not use_ai:
            continue
        if fails >= RECOG_MAX_FAILS:
            r["recognition"], r["recognition_error"] = "failed", "too many consecutive Gemini failures"
            continue
        try:
            data = gemini_json([prep_for_ai(r["_ink"])], REC_PROMPT)
        except RuntimeError as e:
            print(f"⚠️ {e} → 남은 마디는 미인식")
            fails = RECOG_MAX_FAILS
            r["recognition"], r["recognition_error"] = "failed", str(e)
            continue
        calls += 1
        if isinstance(data, list):
            data = data[0] if data else None
        if not isinstance(data, dict):
            fails += 1
            r["recognition"], r["recognition_error"] = "failed", "empty or invalid Gemini response"
            print(f"⚠️ {r['raw_id']}: 인식 응답 없음/형식 오류 (fret=null, confidence=0 으로 보존)")
            continue
        fails = 0
        r["recognition"] = "ok"
        events, rk = columns_to_events(data, r["raw_id"], ts)
        label = data.get("measure_label")
        r["recognized_measure_number"] = label if isinstance(label, int) and not isinstance(label, bool) else None
        r["notes"], r["rhythm_known"] = events, rk
        if events:
            mean_c = float(np.mean([e["confidence"] for e in events]))
            r["confidence"] = round(mean_c * _clamp01(data.get("confidence"), 0.5), 3)
            r["status"] = "recognized" if all(e["fret"] is not None or e["type"] in ("mute", "rest")
                                              for e in events) else "partial"
        if (i + 1) % 10 == 0:
            print(f"  인식 {i + 1}/{len(raws)}")
        time.sleep(RECOG_SLEEP)
    failed = [r["raw_id"] for r in raws if r.get("recognition") == "failed"]
    if failed:
        print(f"⚠️ 인식 실패 마디 {len(failed)}개: {failed[:20]}")
    return calls, failed


# ------------------------------------------------------------------
# 8. 중복 제거 / 순서 복원
#    - 같은 구간 안: 월드 좌표 레지스트리 덕분에 마디당 raw 1개 (내용 비교 안 함)
#    - 구간 경계: '스크롤 중이었고(속도) + 시간 간격이 짧고 + 기하(위치)가 맞고 + 모양이 같을 때'만 중복
#      페이지 넘김(정지)처럼 기하 근거가 없으면 내용이 같아도 별개 마디로 유지
# ------------------------------------------------------------------
def resolve_overlaps(segs, raws):
    by_seg = {}
    for r in raws:
        by_seg.setdefault(r["segment_id"], []).append(r)
    segs = sorted((s for s in segs if by_seg.get(s.id)), key=lambda s: s.t_start)  # 마디 없는 글리치 구간은 건너뜀
    dropped, log = set(), []
    for A, B in zip(segs, segs[1:]):
        ra, rb = by_seg.get(A.id, []), by_seg.get(B.id, [])
        if not ra or not rb:
            continue
        dt = B.t_start - A.t_end
        entry = {"segment_a": A.id, "segment_b": B.id, "gap_sec": round(dt, 3), "vel_px_sec": round(A.vel, 1),
                 "matches": [], "decision": "none"}
        log.append(entry)
        if A.vel < MIN_SCROLL_PX_SEC or dt > MAX_BREAK_SEC or A.nupd < 2:
            entry["decision"] = "keep_both (정지/페이지 넘김 또는 간격 큼: 기하 근거 없음)"
            continue
        tail = [r for r in ra if r["t_last"] >= A.t_end - 1e-6]
        head = [r for r in rb if r["t_first"] <= B.t_start + 1e-6]
        medw = float(np.median([r["bbox"][2] - r["bbox"][0] for r in tail])) if tail else 0
        shift = A.vel * dt
        used = set()
        for h in head:
            for t in tail:
                if t["raw_id"] in used:
                    continue
                geo = abs((h["_first_x0"] + shift) - t["_last_x0"])
                if geo > 0.35 * medw:
                    continue
                dist = vec_dist(h["_vec"], t["_vec"])
                la, lb = t.get("recognized_measure_number"), h.get("recognized_measure_number")
                if dist > HASH_SAME or (la is not None and lb is not None and la != lb):
                    continue
                used.add(t["raw_id"])
                dropped.add(h["raw_id"])
                t["t_first"], t["t_last"] = min(t["t_first"], h["t_first"]), max(t["t_last"], h["t_last"])
                t["observations"] += h["observations"]
                if h["_q"] > t["_q"]:
                    for k in ("timestamp", "frame_index", "bbox", "_ink", "_vec", "_q", "notes", "confidence",
                              "status", "rhythm_known", "recognized_measure_number", "image_path"):
                        t[k] = h[k]
                entry["matches"].append({"kept": t["raw_id"], "dropped": h["raw_id"], "geo_px": round(geo, 1),
                                         "dist": round(dist, 3)})
                break
        if entry["matches"]:
            entry["decision"] = f"overlap_removed({len(entry['matches'])})"
    kept = [r for r in raws if r["raw_id"] not in dropped]
    drop_list = [r for r in raws if r["raw_id"] in dropped]
    return kept, drop_list, log


def build_score(final, ts, meta, expected):
    measures = []
    os.makedirs(os.path.join(DBG, "deduplicated_measures"), exist_ok=True)
    for n, r in enumerate(final, 1):
        p = os.path.join(DBG, "deduplicated_measures", f"m_{n:04d}.png")
        save_ink_png(r["_ink"], p)
        measures.append({
            "number": n,
            "recognized_measure_number": r.get("recognized_measure_number"),
            "status": r["status"], "rhythm_known": r["rhythm_known"],
            "source": {"timestamp_start": round(r["t_first"], 3), "timestamp_end": round(r["t_last"], 3),
                       "frame_index": r["frame_index"], "segment_id": r["segment_id"],
                       "raw_id": r["raw_id"], "image_path": rel(p), "bbox": r["bbox"]},
            "time_signature": dict(ts), "confidence": r["confidence"], "events": r["notes"]})
    return {
        "schema_version": sl.SCHEMA_VERSION,
        "metadata": {**meta, "expected_measures": expected, "detected_measures": len(measures),
                     "incomplete": (len(measures) != expected) if expected else None,
                     "time_signature_assumed": True},
        "instrument": {"type": "guitar", "strings": 6, "tuning": ["E", "A", "D", "G", "B", "E"],
                       "tuning_order": "low_to_high", "string_numbering": "1=highest (top TAB line)"},
        "score": {"tempo": None, "time_signature": dict(ts), "measures": measures},
    }


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------
def check_requirements():
    """RECOGNIZE=1 인데 Gemini 를 쓸 수 없으면 바로 명확하게 종료한다. RECOGNIZE=0 은 키 없이 실행된다."""
    if not RECOGNIZE:
        return
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is required when RECOGNIZE=1  (키 없이 테스트하려면 RECOGNIZE=0)")
        sys.exit(5)
    if genai is None:
        print("ERROR: google-genai 패키지가 설치되어 있지 않습니다. (pip install -r requirements.txt)")
        sys.exit(5)


def summarize(input_label, expected, raw_n, dedup_n, final_n, calls, failed, unrec, rep, svg_ok, pdf_ok, times):
    pf = lambda ok: "PASS" if ok else "FAIL"
    if expected:
        missing = max(0, expected - final_n)
        exp_s, miss_s = str(expected), str(missing)
    else:
        missing, exp_s, miss_s = None, "not provided", "not checked"
    if RECOGNIZE:
        rec_s = f"enabled ({calls} calls, {len(failed)} failed)"
    else:
        rec_s = "disabled (RECOGNIZE=0)"
    if rep.get("schema_valid") is None:
        val_s = "SKIPPED (jsonschema not installed)" if not rep["errors"] else "FAIL"
    else:
        val_s = pf(rep.get("ok", False))
    rows = [("Input", input_label), ("Expected measures", exp_s), ("Raw measures", raw_n),
            ("After dedup", dedup_n), ("Final measures", final_n), ("Missing measures", miss_s),
            ("Unrecognized measures", unrec), ("Recognition", rec_s), ("Validation", val_s),
            ("SVG", pf(svg_ok)), ("PDF", pf(pdf_ok))]
    print("\n" + "=" * 40 + "\nTAB EXTRACTION SUMMARY\n" + "=" * 40)
    for k, v in rows:
        print(f"\n{k}:\n{v}")
    print("\n" + "=" * 40)
    if times:
        print("Stage times: " + ", ".join(f"{k} {v}s" for k, v in times.items()))
    sl.write_step_summary("TAB Extraction Result", rows + [("Stage times", ", ".join(f"{k} {v}s" for k, v in times.items()))])
    return missing


def process(src):
    ts_n, ts_d = [int(v) for v in TIME_SIG.split("/")]
    ts = {"numerator": ts_n, "denominator": ts_d}
    check_requirements()
    input_label = "YouTube URL (hidden)" if is_url(src) else src
    st = Stages(6)
    if os.path.isdir(DBG):
        shutil.rmtree(DBG)
    for d in ("segments", "raw_measures", "deduplicated_measures", "missing_regions"):
        os.makedirs(os.path.join(DBG, d), exist_ok=True)
    print(f"Options: SAMPLE_SEC={SAMPLE_SEC} SEGMENT_SEC={SEGMENT_SEC} MEASURES_PER_LINE={MEASURES_PER_LINE} "
          f"RECOGNIZE={int(RECOGNIZE)} LABEL_CORR_MIN={LABEL_CORR_MIN} "
          f"EXPECTED_MEASURES={EXPECTED_MEASURES if EXPECTED_MEASURES else 'not provided'}")

    st.start(1, "Download video")
    try:
        path, vtitle = download_video(src)
        cap = open_video(path)
    except Exception as e:
        msg = str(e).replace(src, "<input>") if is_url(src) else str(e)
        print(f"ERROR: 영상 준비 실패: {msg[:600]}")
        low = msg.lower()
        if "needs to be reloaded" in low:
            print("  힌트: yt-dlp 가 오래되었거나 쿠키가 무효일 수 있습니다. YOUTUBE_COOKIES secret 을 지우거나 새 쿠키로 바꾸고,")
            print("        그래도 안 되면 영상을 repo 의 input/ 에 올려 video_path 로 실행하세요.")
        elif "sign in" in low or "bot" in low:
            print("  힌트: YouTube 가 GitHub runner IP 를 봇으로 차단했습니다. README 의 cookies 설정 또는 video_path 를 사용하세요.")
        else:
            print("  힌트: 영상이 비공개/삭제/지역 제한인지 확인하고, 안 되면 video_path 로 직접 올린 영상을 사용하세요.")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
          f"fps={fps:.1f}, frames={total}")

    st.start(2, "Detect TAB regions")
    windows, methods = detect_windows(cap, fps, total)
    if not windows:
        print("ERROR: 영상에서 TAB 악보 영역을 찾지 못했습니다.")
        sys.exit(2)

    st.start(3, "Track measures (adaptive sampling, world-coordinate barlines)")
    segs, no_staff, scan_stats = scan_video(cap, fps, windows)
    cap.release()
    if not segs:
        print("ERROR: 악보를 감지하지 못했습니다.")
        sys.exit(3)
    raws, breaks = build_raw(segs)
    print(f"  추적 구간 {len(segs)}개, raw 마디 {len(raws)}개, 샘플 {scan_stats['samples']}개")
    if not raws:
        print("ERROR: 마디를 하나도 찾지 못했습니다.")
        sys.exit(4)

    st.start(4, "Gemini recognition" if RECOGNIZE else "Recognition skipped (RECOGNIZE=0)")
    calls, failed = recognize_all(raws, ts)

    st.start(5, "Deduplicate, restore order, build score.json")
    segs_sorted = sorted(segs, key=lambda s: s.t_start)
    order = {s.id: i for i, s in enumerate(segs_sorted)}
    raws.sort(key=lambda r: (order[r["segment_id"]], r["world_x0"]))
    final, dropped, overlap_log = resolve_overlaps(segs, raws)
    for r in dropped:
        save_ink_png(r["_ink"], os.path.join(DBG, "deduplicated_measures", f"dropped_{r['raw_id']}.png"))

    meta = {"title": TAB_TITLE or vtitle or None, "artist": TAB_COMPOSER or None,
            "source_url": src if is_url(src) else None}
    score = build_score(final, ts, meta, EXPECTED_MEASURES)

    missing = list(breaks)
    for a, b in zip(segs_sorted, segs_sorted[1:]):
        if b.t_start - a.t_end > 3 * max(SAMPLE_SEC, 0.1):
            missing.append({"kind": "segment_gap", "t_from": a.t_end, "t_to": b.t_start,
                            "segment_before": a.id, "segment_after": b.id})
    for t0, t1 in no_staff:
        missing.append({"kind": "no_staff_visible", "t_from": t0, "t_to": t1})
    by_id = {r["raw_id"]: r for r in raws}
    for i, mg in enumerate(missing):
        for key in ("after_raw", "before_raw"):
            if key in mg and mg[key] in by_id:
                save_ink_png(by_id[mg[key]]["_ink"],
                             os.path.join(DBG, "missing_regions", f"gap_{i:02d}_{key}.png"))

    raw_json = {
        "schema_version": "1.0", "source": src if is_url(src) else os.path.basename(src),
        "expected_measures": EXPECTED_MEASURES,
        "video": {"fps": fps, "frames": total, **scan_stats},
        "windows": [{"index": i, "box_2d": b, "method": m} for i, (b, m) in enumerate(zip(windows, methods))],
        "segments": [{"segment_id": s.id, "t_start": s.t_start, "t_end": s.t_end, "frames": s.n_frames,
                      "velocity_px_sec": s.vel, "tracked_measures": len(s.tracks)} for s in segs_sorted],
        "raw": [{k: v for k, v in r.items() if not k.startswith("_")} for r in raws],
        "dropped_as_duplicate": [r["raw_id"] for r in dropped],
        "overlap_decisions": overlap_log, "possible_missing_regions": missing,
        "stats": {"gemini_calls": calls, "recognition_failed": failed, "recognition_enabled": RECOGNIZE},
    }
    dump_json(os.path.join(OUT_DIR, "raw_recognition.json"), raw_json)
    dump_json(os.path.join(OUT_DIR, "score.json"), score)

    st.start(6, "Validate and render SVG/PDF")
    rep = {"errors": [], "warnings": [], "schema_errors": [], "schema_valid": None, "ok": False, "stats": {}}
    svg_ok = pdf_ok = False
    try:
        rep = sl.render_all(score, OUT_DIR, MEASURES_PER_LINE, EXPECTED_MEASURES)
    except Exception as e:                      # 렌더 실패해도 raw/score 는 이미 저장되어 있다
        print(f"ERROR: 렌더링 실패: {type(e).__name__}: {e}")
    svg_ok = os.path.exists(os.path.join(OUT_DIR, "score.svg"))
    pdf_ok = os.path.exists(os.path.join(OUT_DIR, "score.pdf"))
    st.end()

    n_final = len(final)
    unrec = sum(1 for r in final if r["status"] == "unrecognized")
    missing_n = summarize(input_label, EXPECTED_MEASURES, len(raws), n_final, n_final, calls, failed, unrec,
                          rep, svg_ok, pdf_ok, st.times)
    if EXPECTED_MEASURES and n_final != EXPECTED_MEASURES:
        print(f"⚠️ expected {EXPECTED_MEASURES}, detected {n_final} → incomplete extraction")
        for mg in missing[:12]:
            extra = f" (segment {mg['segment_id']}, ≈{mg['approx_missing']}마디)" if "approx_missing" in mg else ""
            print(f"   possible missing region: {mg['kind']} {mg.get('t_from', 0):.1f}s ~ {mg.get('t_to', 0):.1f}s{extra}")
    if unrec:
        print(f"ℹ️ 미인식 마디 {unrec}개는 빈 마디(unrecognized)로 렌더링됩니다.")
    for w in rep["warnings"][:15]:
        print("⚠️", w)
    for e in rep["errors"][:15] + rep["schema_errors"][:10]:
        print("❌", e)
    print(f"완료: {OUT_DIR}/score.json, score.svg, score.pdf")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python extract.py <YouTube URL 또는 로컬 영상 경로>")
        sys.exit(1)
    process(sys.argv[1])
