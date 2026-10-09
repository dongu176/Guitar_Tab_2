"""합성 영상으로 파이프라인을 끝까지 검증하는 셀프테스트 (네트워크/Gemini 불필요).

    python tests/selftest.py                       # jsonschema 가 없으면 해당 테스트는 SKIP
    python tests/selftest.py --require-jsonschema  # CI: jsonschema 검증을 반드시 수행
"""
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, FPS = 1280, 720, 30
TOP = 400


def make_scroll(path, nm=40, mw=400, speed=150.0):
    rng = np.random.RandomState(1)
    riff = [(1, "7"), (3, "5"), (2, "6"), (4, "X")]
    cont = [riff if 8 <= m <= 21 else [(int(rng.randint(0, 6)), str(int(rng.randint(0, 13)))) for _ in range(4)]
            for m in range(nm)]
    world = np.zeros((H, nm * mw + W + 50, 3), np.uint8) + 30
    for i in range(6):
        cv2.line(world, (0, TOP + i * 20), (world.shape[1], TOP + i * 20), (255, 255, 255), 2)
    for m in range(nm + 1):
        x = m * mw + 100
        cv2.line(world, (x, TOP), (x, TOP + 100), (255, 255, 255), 2)
        cv2.putText(world, str(m + 1), (x + 4, TOP - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for m, c in enumerate(cont):
        for k, (s, t) in enumerate(c):
            cv2.putText(world, t, (m * mw + 140 + k * 85, TOP + s * 20 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(int(nm * mw / speed * FPS)):
        x = int(f / FPS * speed)
        fr = world[:, x:x + W].copy()
        if f == 900:                                   # 노이즈 프레임 1장
            fr[:] = rng.randint(0, 255, fr.shape, dtype=np.uint8)
        vw.write(fr)
    vw.release()


def make_flip(path, pages=12):
    riff = [(1, "7"), (3, "5"), (2, "6"), (4, "3")]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for p in range(pages):
        fr = np.zeros((H, W, 3), np.uint8) + 30
        for i in range(6):
            cv2.line(fr, (0, TOP + i * 20), (W, TOP + i * 20), (255, 255, 255), 2)
        for k in range(4):
            x = 60 + k * 400
            cv2.line(fr, (x, TOP), (x, TOP + 100), (255, 255, 255), 2)
            cv2.putText(fr, str(p * 3 + k + 1), (x + 4, TOP - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for k in range(3):
            for j, (s, t) in enumerate(riff):
                cv2.putText(fr, t, (100 + k * 400 + j * 85, TOP + s * 20 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
        for _ in range(60):
            vw.write(fr)
    vw.release()


def run(args, env_extra, cwd):
    env = {k: v for k, v in os.environ.items() if k not in ("EXPECTED_MEASURES", "GEMINI_API_KEY")}
    env.update(env_extra)
    p = subprocess.run([sys.executable] + args, env=env, cwd=cwd, capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def main():
    require_js = "--require-jsonschema" in sys.argv
    results = []

    def check(name, ok, detail=""):
        results.append((name, "PASS" if ok else "FAIL", detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    tmp = tempfile.mkdtemp(prefix="tabtest_")
    scroll, flip = os.path.join(tmp, "scroll.mp4"), os.path.join(tmp, "flip.mp4")
    print("synthetic videos →", tmp, flush=True)
    make_scroll(scroll)
    make_flip(flip)
    ext = os.path.join(ROOT, "extract.py")

    # TEST 1: RECOGNIZE=0, EXPECTED_MEASURES 없음
    o1 = os.path.join(tmp, "o1")
    rc, out = run([ext, scroll], {"RECOGNIZE": "0", "OUT_DIR": o1}, ROOT)
    sc = json.load(open(os.path.join(o1, "score.json"), encoding="utf-8")) if rc == 0 else {"score": {"measures": []}}
    n1 = len(sc["score"]["measures"])
    check("T1 RECOGNIZE=0, no expected measures", rc == 0 and n1 == 40
          and "not provided" in out and "not checked" in out, f"rc={rc} measures={n1}")
    check("T3 SVG/PDF created", all(os.path.getsize(os.path.join(o1, f)) > 0 for f in ("score.svg", "score.pdf"))
          if rc == 0 else False)
    nums = [m["number"] for m in sc["score"]["measures"]]
    check("T1b numbering 1..N, no duplicates", nums == list(range(1, len(nums) + 1)))

    # TEST 2: EXPECTED_MEASURES=40 (그리고 빈 문자열)
    o2 = os.path.join(tmp, "o2")
    rc, out = run([ext, scroll], {"RECOGNIZE": "0", "OUT_DIR": o2, "EXPECTED_MEASURES": "40"}, ROOT)
    check("T2 EXPECTED_MEASURES=40", rc == 0 and "Expected measures:\n40" in out and "Missing measures:\n0" in out,
          f"rc={rc}")
    o2b = os.path.join(tmp, "o2b")
    rc, out = run([ext, flip], {"RECOGNIZE": "0", "OUT_DIR": o2b, "EXPECTED_MEASURES": ""}, ROOT)
    check("T2b EXPECTED_MEASURES='' (empty string) + page-flip video = 36", rc == 0 and "Final measures:\n36" in out,
          f"rc={rc}")
    o2c = os.path.join(tmp, "o2c")
    rc, out = run([ext, flip], {"RECOGNIZE": "0", "OUT_DIR": o2c, "EXPECTED_MEASURES": "40"}, ROOT)
    check("T2c wrong expected → incomplete warning, no crash",
          rc == 0 and "incomplete extraction" in out and "Missing measures:\n4" in out, f"rc={rc}")

    # TEST 4: 재렌더링
    o4 = os.path.join(tmp, "o4")
    rc, out = run([os.path.join(ROOT, "render_score.py"), os.path.join(o1, "score.json"), "--out-dir", o4,
                   "--per-line", "4"], {}, ROOT)
    check("T4 render_score.py re-render", rc == 0 and os.path.getsize(os.path.join(o4, "score.pdf")) > 0, f"rc={rc}")

    # TEST 5: jsonschema
    try:
        import jsonschema
        schema = json.load(open(os.path.join(o1, "score.schema.json"), encoding="utf-8"))
        jsonschema.Draft7Validator.check_schema(schema)
        errs = list(jsonschema.Draft7Validator(schema).iter_errors(sc))
        rep = json.load(open(os.path.join(o1, "score_validation.json"), encoding="utf-8"))
        check("T5 jsonschema validation", not errs and rep.get("schema_valid") is True,
              f"errors={len(errs)} {errs[0].message if errs else ''}")
    except ImportError:
        if require_js:
            check("T5 jsonschema validation", False, "jsonschema not installed but required")
        else:
            results.append(("T5 jsonschema validation", "SKIP", "jsonschema not installed"))
            print("SKIP  T5 jsonschema validation (not installed)")

    # TEST 6: RECOGNIZE=1 + 키 없음 → 명확한 오류
    rc, out = run([ext, scroll], {"RECOGNIZE": "1", "OUT_DIR": os.path.join(tmp, "o6")}, ROOT)
    check("T6 RECOGNIZE=1 without key → clear error", rc == 5 and "GEMINI_API_KEY is required" in out, f"rc={rc}")

    # TEST 7: 없는 파일
    rc, out = run([ext, os.path.join(tmp, "nope.mp4")], {"RECOGNIZE": "0", "OUT_DIR": os.path.join(tmp, "o7")}, ROOT)
    check("T7 missing input file → error exit", rc == 1 and "ERROR" in out, f"rc={rc}")

    bad = [r for r in results if r[1] == "FAIL"]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed" + (f", FAILED: {[b[0] for b in bad]}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
