"""scorelib.py - Score JSON 모델 / JSON Schema / 검증 / 레이아웃 / SVG / PDF
영상·AI 의존성이 없다. score.json 하나만 있으면 SVG/PDF를 다시 만들 수 있다.

규약
- string: 1 = 가장 높은 음(TAB 맨 위 줄), 6 = 가장 낮은 음(맨 아래 줄)
- instrument.tuning 은 낮은 줄 → 높은 줄 순서(E A D G B E)
- position / duration 은 "분자/분모" 문자열(부동소수점 사용 안 함). 모르면 null
- 확신 없는 값은 null + 낮은 confidence 로 보존한다(추측해서 채우지 않음)
"""
import json
import os
from fractions import Fraction
from xml.sax.saxutils import escape

SCHEMA_VERSION = "1.0"
TECHNIQUES = ["hammer_on", "pull_off", "slide_up", "slide_down", "bend", "release",
              "vibrato", "harmonic", "palm_mute", "let_ring"]
EVENT_TYPES = ["note", "rest", "mute", "dead_note", "grace_note"]
TIES = ["start", "stop", "continue"]
MEASURE_STATUS = ["recognized", "partial", "unrecognized"]


# ------------------------------------------------------------------
# 유리수 헬퍼
# ------------------------------------------------------------------
def parse_frac(s):
    if s is None:
        return None
    try:
        return Fraction(str(s))
    except Exception:
        return None


def frac_str(f):
    return f"{f.numerator}/{f.denominator}"


def _pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def valid_duration(s):
    f = parse_frac(s)
    return f is not None and f > 0 and f <= 4 and _pow2(f.denominator) and f.denominator <= 64


# ------------------------------------------------------------------
# JSON Schema
# ------------------------------------------------------------------
def build_schema():
    frac = {"type": ["string", "null"], "pattern": r"^\d+/[1-9]\d*$"}
    ts = {"type": "object", "required": ["numerator", "denominator"],
          "properties": {"numerator": {"type": "integer", "minimum": 1},
                         "denominator": {"type": "integer", "minimum": 1}}}
    event = {
        "type": "object",
        "required": ["type", "position", "duration", "string", "fret", "confidence"],
        "properties": {
            "id": {"type": "string"},
            "type": {"enum": EVENT_TYPES},
            "position": frac,
            "duration": frac,
            "x_norm": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "string": {"type": ["integer", "null"], "minimum": 1, "maximum": 6},
            "fret": {"type": ["integer", "null"], "minimum": 0, "maximum": 30},
            "technique": {"enum": TECHNIQUES + [None]},
            "modifiers": {"type": "array", "items": {"enum": TECHNIQUES}},
            "tie": {"enum": TIES + [None]},
            "chord_id": {"type": ["string", "null"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    source = {
        "type": "object",
        "properties": {
            "timestamp_start": {"type": ["number", "null"]},
            "timestamp_end": {"type": ["number", "null"]},
            "frame_index": {"type": ["integer", "null"]},
            "segment_id": {"type": ["integer", "null"]},
            "raw_id": {"type": ["string", "null"]},
            "image_path": {"type": ["string", "null"]},
            "bbox": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        },
    }
    measure = {
        "type": "object",
        "required": ["number", "source", "time_signature", "confidence", "events"],
        "properties": {
            "number": {"type": "integer", "minimum": 1},
            "recognized_measure_number": {"type": ["integer", "null"]},
            "status": {"enum": MEASURE_STATUS},
            "rhythm_known": {"type": "boolean"},
            "source": source,
            "time_signature": ts,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "events": {"type": "array", "items": event},
        },
    }
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Guitar TAB Score",
        "type": "object",
        "required": ["schema_version", "metadata", "instrument", "score"],
        "properties": {
            "schema_version": {"type": "string"},
            "metadata": {"type": "object", "properties": {
                "title": {"type": ["string", "null"]}, "artist": {"type": ["string", "null"]},
                "source_url": {"type": ["string", "null"]},
                "expected_measures": {"type": ["integer", "null"], "minimum": 1},
                "detected_measures": {"type": "integer", "minimum": 0},
                "incomplete": {"type": ["boolean", "null"]},
                "time_signature_assumed": {"type": "boolean"}}},
            "instrument": {
                "type": "object", "required": ["type", "strings", "tuning"],
                "properties": {"type": {"type": "string"},
                               "strings": {"type": "integer", "minimum": 1, "maximum": 8},
                               "tuning": {"type": "array", "items": {"type": "string"}}}},
            "score": {
                "type": "object", "required": ["time_signature", "measures"],
                "properties": {"tempo": {"type": ["number", "null"]},
                               "time_signature": ts,
                               "measures": {"type": "array", "items": measure}}},
        },
    }


def write_schema(path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_schema(), f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# 검증
# ------------------------------------------------------------------
def validate_score(score, expected=None):
    rep = {"schema_valid": None, "schema_errors": [], "errors": [], "warnings": [], "stats": {}}
    try:
        import jsonschema
        v = jsonschema.Draft7Validator(build_schema())
        errs = sorted(v.iter_errors(score), key=lambda e: [str(p) for p in e.path])
        rep["schema_errors"] = ["/".join(str(p) for p in e.path) + ": " + e.message for e in errs[:200]]
        rep["schema_valid"] = len(errs) == 0
    except ImportError:
        rep["warnings"].append("jsonschema 미설치 - 스키마 검증 생략 (pip install jsonschema)")

    err, warn = rep["errors"], rep["warnings"]
    measures = (score.get("score") or {}).get("measures") or []
    nums = [m.get("number") for m in measures]

    # --- Measure ---
    if nums != list(range(1, len(nums) + 1)):
        dup = sorted({n for n in nums if nums.count(n) > 1})
        if dup:
            err.append(f"중복된 마디 번호: {dup}")
        miss = [n for n in range(1, (max([n for n in nums if isinstance(n, int)] or [0])) + 1) if n not in nums]
        if miss:
            err.append(f"누락된 마디 번호: {miss[:30]}")
        if not dup and not miss:
            err.append("마디 번호가 순차적이지 않음")
    if expected and len(measures) != expected:
        warn.append(f"예상 {expected}마디 / 현재 {len(measures)}마디 (차이 {expected - len(measures)})")

    empty = [m.get("number") for m in measures if not m.get("events")]
    if empty:
        warn.append(f"이벤트가 없는(미인식/빈) 마디 {len(empty)}개: {empty[:30]}")

    prev_t = None
    for m in measures:
        t = (m.get("source") or {}).get("timestamp_start")
        if t is not None and prev_t is not None and t < prev_t - 1e-6:
            warn.append(f"마디 {m.get('number')}: timestamp 역전 ({t:.2f}s < {prev_t:.2f}s)")
        if t is not None:
            prev_t = t

    seen = {}
    for m in measures:
        s = m.get("source") or {}
        key = (s.get("segment_id"), s.get("frame_index"), tuple(s.get("bbox") or []))
        if s.get("frame_index") is not None:
            seen.setdefault(key, []).append(m.get("number"))
    for key, ns in seen.items():
        if len(ns) > 1:
            warn.append(f"마디 {ns} 가 같은 영상 소스(segment {key[0]}, frame {key[1]}, 같은 bbox)를 공유 - 중복 의심")

    # --- Note / Musical ---
    rhythm_unknown, bad_sum = [], []
    for m in measures:
        n = m.get("number")
        ts = m.get("time_signature") or (score.get("score") or {}).get("time_signature") or {}
        try:
            mlen = Fraction(int(ts.get("numerator", 4)), int(ts.get("denominator", 4)))
        except Exception:
            mlen = Fraction(1)
        evs = m.get("events") or []
        known = True
        for e in evs:
            s, fr, c = e.get("string"), e.get("fret"), e.get("confidence")
            if s is not None and not (isinstance(s, int) and 1 <= s <= 6):
                err.append(f"마디 {n}: string 범위 오류 {s}")
            if fr is not None and not (isinstance(fr, int) and 0 <= fr <= 30):
                err.append(f"마디 {n}: fret 범위 오류 {fr}")
            if not isinstance(c, (int, float)) or not (0 <= c <= 1):
                err.append(f"마디 {n}: confidence 범위 오류 {c}")
            p, d = e.get("position"), e.get("duration")
            if p is not None:
                pf = parse_frac(p)
                if pf is None or pf < 0 or pf >= mlen:
                    err.append(f"마디 {n}: position 오류 {p}")
            if d is not None and not valid_duration(d):
                err.append(f"마디 {n}: duration 오류 {d}")
            if p is None or d is None:
                if e.get("type") != "grace_note":
                    known = False
        if evs and not known:
            rhythm_unknown.append(n)
        elif evs and known:
            col = {}
            for e in evs:
                if e.get("type") == "grace_note":
                    continue
                pf, df = parse_frac(e.get("position")), parse_frac(e.get("duration"))
                if pf is None or df is None:
                    continue
                col[pf] = max(col.get(pf, Fraction(0)), df)
            cur, ok = Fraction(0), True
            for pf in sorted(col):
                if pf != cur:
                    ok = False
                    break
                cur = pf + col[pf]
            if not ok or cur != mlen:
                bad_sum.append(n)
    if rhythm_unknown:
        warn.append(f"리듬(position/duration)을 확인하지 못한 마디 {len(rhythm_unknown)}개 - 박자 합 검사 생략: {rhythm_unknown[:20]}")
    if bad_sum:
        warn.append(f"duration 합이 박자와 일치하지 않는 마디(자동 수정 안 함): {bad_sum}")

    low = [m.get("number") for m in measures if m.get("status") != "unrecognized"
           and isinstance(m.get("confidence"), (int, float)) and m["confidence"] < 0.5]
    if low:
        warn.append(f"신뢰도 0.5 미만 마디: {low[:30]}")

    rep["stats"] = {
        "measures": len(measures),
        "empty_measures": len(empty),
        "rhythm_unknown_measures": len(rhythm_unknown),
        "duration_mismatch_measures": len(bad_sum),
        "events": sum(len(m.get("events") or []) for m in measures),
    }
    rep["ok"] = (not err) and (rep["schema_valid"] is not False)
    return rep


# ------------------------------------------------------------------
# 레이아웃 (마디 = 독립 객체)
# ------------------------------------------------------------------
PAGE_W, PAGE_H, MARGIN = 1240, 1754, 70
STR_GAP = 13
LABEL_H = 30
STEM_H = 42
LINE_H = LABEL_H + 5 * STR_GAP + STEM_H + 26
FONT = "'Nanum Gothic','Noto Sans CJK KR','DejaVu Sans',Arial,sans-serif"
TECH_LABEL = {"hammer_on": "H", "pull_off": "P", "slide_up": "/", "slide_down": "\\",
              "bend": "b", "release": "r", "vibrato": "~", "harmonic": "Harm.",
              "palm_mute": "P.M.", "let_ring": "let ring"}
ARC_TECH = ("hammer_on", "pull_off", "slide_up", "slide_down")


def _ts_len(m, default):
    ts = m.get("time_signature") or default or {}
    try:
        return Fraction(int(ts.get("numerator", 4)), int(ts.get("denominator", 4)))
    except Exception:
        return Fraction(1)


def group_columns(m, default_ts):
    """이벤트를 가로 위치 기준 컬럼으로 묶는다. position이 있으면 박 위치, 없으면 영상의 x_norm."""
    mlen = _ts_len(m, default_ts)
    evs = m.get("events") or []
    items = []
    n = max(1, len(evs))
    for i, e in enumerate(evs):
        pf = parse_frac(e.get("position"))
        if pf is not None and mlen > 0:
            key = float(pf / mlen)
        elif e.get("x_norm") is not None:
            key = float(e["x_norm"])
        else:
            key = (i + 0.5) / n
        items.append((key, i, e))
    items.sort(key=lambda t: (t[0], t[1]))
    cols = []
    for key, i, e in items:
        cid = e.get("chord_id")
        if cols and (abs(key - cols[-1]["key"]) < 0.015 or (cid and cid == cols[-1]["chord"])):
            cols[-1]["events"].append(e)
        else:
            cols.append({"key": key, "chord": cid, "events": [e]})
    return cols


def _col_w(col):
    digits = 1
    for e in col["events"]:
        fr = e.get("fret")
        digits = max(digits, len(str(fr)) if fr is not None else 1)
    return 12 + 8 * digits


def measure_min_width(m, default_ts):
    cols = group_columns(m, default_ts)
    if not cols:
        return 130
    return max(110, 26 + sum(_col_w(c) + 6 for c in cols))


def layout_score(score, per_line=0):
    """마디 폭을 계산해 줄/페이지로 배치한다. per_line=0 이면 자동."""
    default_ts = score["score"].get("time_signature")
    content_w = PAGE_W - 2 * MARGIN
    maxn = per_line if per_line > 0 else 8
    limit = content_w / (0.8 if per_line > 0 else 1.0)
    lines, cur, cur_w = [], [], 0.0
    for m in score["score"]["measures"]:
        mw = measure_min_width(m, default_ts)
        if cur and (len(cur) >= maxn or cur_w + mw > limit):
            lines.append((cur, cur_w))
            cur, cur_w = [], 0.0
        cur.append((m, mw))
        cur_w += mw
    if cur:
        lines.append((cur, cur_w))

    meta = score.get("metadata") or {}
    header_h = 0
    if meta.get("title"):
        header_h += 70
    if meta.get("artist"):
        header_h += 36
    if header_h:
        header_h += 20

    pages, page, y = [], [], MARGIN + header_h
    bottom = PAGE_H - MARGIN - 30
    for idx, (ms, w) in enumerate(lines):
        last = idx == len(lines) - 1
        scale = content_w / w
        scale = min(scale, 1.15 if last else 2.2)
        if y + LINE_H > bottom:
            pages.append(page)
            page, y = [], MARGIN
        x = MARGIN
        placed = []
        for m, mw in ms:
            placed.append({"m": m, "x": x, "w": mw * scale})
            x += mw * scale
        page.append({"y": y, "measures": placed})
        y += LINE_H
    pages.append(page)
    return pages


# ------------------------------------------------------------------
# 드로잉 프리미티브 (SVG / PDF 공통)
# ------------------------------------------------------------------
class Canvas:
    def __init__(self):
        self.ops = []

    def line(self, x1, y1, x2, y2, w=1.0, color="#000"):
        self.ops.append(("line", x1, y1, x2, y2, w, color))

    def rect(self, x, y, w, h, fill="#000", stroke=None):
        self.ops.append(("rect", x, y, w, h, fill, stroke))

    def text(self, x, y, s, size=12, anchor="middle", bold=False, fill="#000"):
        self.ops.append(("text", x, y, str(s), size, anchor, bold, fill))

    def arc(self, x1, y1, x2, y2, rise=-12, w=1.0, color="#000"):
        self.ops.append(("arc", x1, y1, x2, y2, rise, w, color))


def _draw_measure(cv, m, x0, w, y0, default_ts, is_last_piece):
    st_top = y0 + LABEL_H
    ys = [st_top + i * STR_GAP for i in range(6)]
    st_bot = ys[-1]
    for y in ys:
        cv.line(x0, y, x0 + w, y, 0.9, "#222")
    cv.line(x0, ys[0], x0, st_bot, 1.3)
    cv.line(x0 + w, ys[0], x0 + w, st_bot, 2.6 if is_last_piece else 1.3)
    cv.text(x0 + 3, y0 + LABEL_H - 16, m.get("number", ""), 10, "start", False, "#444")

    cols = group_columns(m, default_ts)
    if not cols:
        cv.text(x0 + w / 2, st_top + 2.5 * STR_GAP + 4, "unrecognized", 11, "middle", False, "#999")
        return
    pad_l, pad_r = 14, 14
    s0, s1 = x0 + pad_l, x0 + w - pad_r
    xs = [s0 + c["key"] * (s1 - s0) for c in cols]
    gaps = [(_col_w(cols[i - 1]) + _col_w(cols[i])) / 2 + 2 for i in range(1, len(cols))]
    for i in range(1, len(xs)):
        xs[i] = max(xs[i], xs[i - 1] + gaps[i - 1])
    if xs[-1] > s1 and xs[-1] > s0:
        k = (s1 - s0) / (xs[-1] - s0)
        xs = [s0 + (x - s0) * k for x in xs]

    for ci, (col, cx) in enumerate(zip(cols, xs)):
        for e in col["events"]:
            t, s, fr = e.get("type"), e.get("string"), e.get("fret")
            if t == "rest":
                cv.rect(cx - 6, ys[2] - 2, 12, 5, "#000")
                continue
            if s is None:
                continue
            y = ys[s - 1]
            grace = t == "grace_note"
            if t in ("mute", "dead_note"):
                label = "X"
            elif fr is None:
                label = "?"
            else:
                label = str(fr)
            if e.get("technique") == "harmonic" or "harmonic" in (e.get("modifiers") or []):
                label = f"<{label}>"
            size = 9 if grace else 12
            cv.rect(cx - len(label) * 3.6 - 1, y - 6.5, len(label) * 7.2 + 2, 13, "#fff")
            cv.text(cx, y + 4.2, label, size, "middle", True, "#b00" if label == "?" else "#000")

            techs = ([e["technique"]] if e.get("technique") else []) + list(e.get("modifiers") or [])
            for ti, tname in enumerate(techs):
                if tname in ARC_TECH:
                    continue
                lab = TECH_LABEL.get(tname)
                if lab and tname != "harmonic":
                    cv.text(cx, st_top - 5 - 10 * ti, lab, 9, "middle", False, "#333")
            tie_start = e.get("tie") in ("start", "continue")
            arc_tech = e.get("technique") in ARC_TECH
            if arc_tech or tie_start:
                for nj in range(ci + 1, len(cols)):
                    nxt = [q for q in cols[nj]["events"] if q.get("string") == s]
                    if nxt:
                        cv.arc(cx + 5, y - 7, xs[nj] - 5, y - 7, -9, 1.0)
                        if arc_tech:
                            cv.text((cx + xs[nj]) / 2, y - 17, TECH_LABEL[e["technique"]], 8, "middle", False, "#333")
                        break

    # 줄기/빔: 리듬이 확인된 컬럼에만 그린다
    def dur(col):
        ds = [parse_frac(e.get("duration")) for e in col["events"] if e.get("type") != "grace_note"]
        ds = [d for d in ds if d is not None]
        return min(ds) if ds else None

    def beams(d):
        if d is None or d >= Fraction(1, 4):
            return 0
        return 1 if d >= Fraction(1, 8) else (2 if d >= Fraction(1, 16) else 3)

    stem_top, stem_bot = st_bot + 7, st_bot + 7 + STEM_H * 0.7
    levels = []
    for col, cx in zip(cols, xs):
        d = dur(col)
        if d is None:
            levels.append(None)
            continue
        if any(e.get("type") != "rest" for e in col["events"]):
            cv.line(cx, stem_top, cx, stem_bot, 1.0)
        levels.append(beams(d))
    i = 0
    while i < len(cols):
        if not levels[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(cols) and levels[j + 1]:
            j += 1
        for lv in range(1, 4):
            k = i
            while k <= j:
                if levels[k] >= lv:
                    e2 = k
                    while e2 + 1 <= j and levels[e2 + 1] >= lv:
                        e2 += 1
                    yb = stem_bot - (lv - 1) * 5
                    if e2 > k:
                        cv.line(xs[k], yb, xs[e2], yb, 2.2)
                    elif lv == 1:
                        cv.line(xs[k], stem_bot, xs[k] + 7, stem_bot - 6, 1.4)
                    k = e2 + 1
                else:
                    k += 1
        i = j + 1


def render_canvases(score, per_line=0):
    pages = layout_score(score, per_line)
    meta = score.get("metadata") or {}
    default_ts = score["score"].get("time_signature")
    total_measures = len(score["score"]["measures"])
    canvases = []
    last_number = score["score"]["measures"][-1]["number"] if total_measures else None
    for pi, page in enumerate(pages):
        cv = Canvas()
        if pi == 0:
            y = MARGIN
            if meta.get("title"):
                size = 44
                while len(str(meta["title"])) * size * 0.62 > PAGE_W - 2 * MARGIN and size > 22:
                    size -= 2
                cv.text(PAGE_W / 2, y + 44, meta["title"], size, "middle", True)
                y += 70
            if meta.get("artist"):
                cv.text(PAGE_W - MARGIN, y + 22, meta["artist"], 20, "end", False, "#222")
        for line in page:
            for pm in line["measures"]:
                _draw_measure(cv, pm["m"], pm["x"], pm["w"], line["y"], default_ts,
                              pm["m"].get("number") == last_number)
        cv.text(PAGE_W / 2, PAGE_H - MARGIN / 2, f"{pi + 1} / {len(pages)}", 11, "middle", False, "#777")
        canvases.append(cv)
    return canvases


# ------------------------------------------------------------------
# SVG
# ------------------------------------------------------------------
def _svg_ops(cv, dy=0):
    out = []
    for op in cv.ops:
        k = op[0]
        if k == "line":
            _, x1, y1, x2, y2, w, c = op
            out.append(f'<line x1="{x1:.1f}" y1="{y1 + dy:.1f}" x2="{x2:.1f}" y2="{y2 + dy:.1f}" stroke="{c}" stroke-width="{w}"/>')
        elif k == "rect":
            _, x, y, w, h, fill, stroke = op
            st = f' stroke="{stroke}"' if stroke else ""
            out.append(f'<rect x="{x:.1f}" y="{y + dy:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}"{st}/>')
        elif k == "text":
            _, x, y, s, size, anchor, bold, fill = op
            wt = ' font-weight="bold"' if bold else ""
            out.append(f'<text x="{x:.1f}" y="{y + dy:.1f}" font-size="{size}" text-anchor="{anchor}"{wt} fill="{fill}" font-family="{FONT}">{escape(s)}</text>')
        elif k == "arc":
            _, x1, y1, x2, y2, rise, w, c = op
            out.append(f'<path d="M{x1:.1f},{y1 + dy:.1f} Q{(x1 + x2) / 2:.1f},{(y1 + y2) / 2 + rise * 2 + dy:.1f} {x2:.1f},{y2 + dy:.1f}" fill="none" stroke="{c}" stroke-width="{w}"/>')
    return "\n".join(out)


def to_svg_pages(canvases):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 {PAGE_W} {PAGE_H}">'
            f'<rect width="{PAGE_W}" height="{PAGE_H}" fill="#fff"/>\n{_svg_ops(cv)}\n</svg>' for cv in canvases]


def to_svg_stacked(canvases):
    gap = 30
    h = len(canvases) * (PAGE_H + gap)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" height="{h}" viewBox="0 0 {PAGE_W} {h}">',
             f'<rect width="{PAGE_W}" height="{h}" fill="#e8e8e8"/>']
    for i, cv in enumerate(canvases):
        dy = i * (PAGE_H + gap)
        parts.append(f'<rect x="0" y="{dy}" width="{PAGE_W}" height="{PAGE_H}" fill="#fff"/>')
        parts.append(_svg_ops(cv, dy))
    parts.append("</svg>")
    return "\n".join(parts)


# ------------------------------------------------------------------
# PDF (reportlab 벡터 출력. 같은 Canvas 프리미티브에서 생성되므로 SVG와 동일한 모양)
# ------------------------------------------------------------------
_FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", None, "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", None),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0, "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0, "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", None, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", None),
    ("C:/Windows/Fonts/malgun.ttf", None, "C:/Windows/Fonts/malgunbd.ttf", None),
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", 0, "/System/Library/Fonts/AppleSDGothicNeo.ttc", 0),
]


def _register_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    for reg, ri, bold, bi in _FONT_CANDIDATES:
        if os.path.exists(reg):
            try:
                pdfmetrics.registerFont(TTFont("ScoreFont", reg, subfontIndex=ri or 0))
                bp = bold if os.path.exists(bold) else reg
                pdfmetrics.registerFont(TTFont("ScoreFontBold", bp, subfontIndex=(bi or 0)))
                return "ScoreFont", "ScoreFontBold"
            except Exception:
                continue
    print("⚠️ 한글 폰트를 찾지 못해 Helvetica 사용 (한글 제목이 깨질 수 있음)")
    return "Helvetica", "Helvetica-Bold"


def _hex(c):
    from reportlab.lib.colors import HexColor
    c = c.strip()
    if len(c) == 4 and c[0] == "#":          # #rgb → #rrggbb (reportlab 은 3자리를 잘못 해석함)
        c = "#" + "".join(ch * 2 for ch in c[1:])
    return HexColor(c)


def write_pdf(canvases, path):
    from reportlab.pdfgen import canvas as rl
    reg, bold = _register_fonts()
    W, H = 595.276, 841.89
    k = W / PAGE_W
    c = rl.Canvas(path, pagesize=(W, H))
    c.setTitle("TAB Score")
    for cv in canvases:
        for op in cv.ops:
            kind = op[0]
            if kind == "line":
                _, x1, y1, x2, y2, w, col = op
                c.setStrokeColor(_hex(col))
                c.setLineWidth(w * k)
                c.line(x1 * k, H - y1 * k, x2 * k, H - y2 * k)
            elif kind == "rect":
                _, x, y, w, h, fill, stroke = op
                c.setFillColor(_hex(fill))
                if stroke:
                    c.setStrokeColor(_hex(stroke))
                c.rect(x * k, H - (y + h) * k, w * k, h * k, stroke=1 if stroke else 0, fill=1)
            elif kind == "text":
                _, x, y, s, size, anchor, b, fill = op
                c.setFillColor(_hex(fill))
                c.setFont(bold if b else reg, size * k)
                if anchor == "middle":
                    c.drawCentredString(x * k, H - y * k, s)
                elif anchor == "end":
                    c.drawRightString(x * k, H - y * k, s)
                else:
                    c.drawString(x * k, H - y * k, s)
            elif kind == "arc":
                _, x1, y1, x2, y2, rise, w, col = op
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + rise * 2
                p = c.beginPath()
                p.moveTo(x1 * k, H - y1 * k)
                # 2차 → 3차 베지어 변환
                p.curveTo((x1 + 2 / 3 * (cx - x1)) * k, H - (y1 + 2 / 3 * (cy - y1)) * k,
                          (x2 + 2 / 3 * (cx - x2)) * k, H - (y2 + 2 / 3 * (cy - y2)) * k,
                          x2 * k, H - y2 * k)
                c.setStrokeColor(_hex(col))
                c.setLineWidth(w * k)
                c.drawPath(p, stroke=1, fill=0)
        c.showPage()
    c.save()


# ------------------------------------------------------------------
# 통합 진입점
# ------------------------------------------------------------------
def render_all(score, out_dir, per_line=0, expected=None):
    os.makedirs(out_dir, exist_ok=True)
    write_schema(os.path.join(out_dir, "score.schema.json"))
    if expected is None:
        expected = (score.get("metadata") or {}).get("expected_measures")
    rep = validate_score(score, expected)
    with open(os.path.join(out_dir, "score_validation.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    if not score["score"]["measures"]:
        print("⚠️ 마디가 없어 렌더링을 건너뜁니다.")
        return rep
    cvs = render_canvases(score, per_line)
    with open(os.path.join(out_dir, "score.svg"), "w", encoding="utf-8") as f:
        f.write(to_svg_stacked(cvs))
    pages_dir = os.path.join(out_dir, "svg_pages")
    os.makedirs(pages_dir, exist_ok=True)
    for i, svg in enumerate(to_svg_pages(cvs), 1):
        with open(os.path.join(pages_dir, f"page_{i:02d}.svg"), "w", encoding="utf-8") as f:
            f.write(svg)
    write_pdf(cvs, os.path.join(out_dir, "score.pdf"))
    rep["pages"] = len(cvs)
    return rep


def write_step_summary(title, rows):
    """GitHub Actions 의 $GITHUB_STEP_SUMMARY 에 표를 추가한다. (로컬에서는 아무 일도 안 함)"""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"## {title}\n\n| Item | Result |\n|---|---:|\n")
            for k, v in rows:
                f.write(f"| {k} | {str(v).replace('|', '/')} |\n")
            f.write("\n")
    except OSError:
        pass
