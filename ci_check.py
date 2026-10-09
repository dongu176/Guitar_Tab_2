"""파이프라인 산출물 검사. 필수 파일이 없거나 score.json 이 스키마 검증에 실패하면 종료 코드 1.
EXPECTED_MEASURES 가 없다는 이유로는 절대 실패하지 않는다.

사용법: python ci_check.py output [--render-only] [--allow-missing-jsonschema]
"""
import json
import os
import sys

REQUIRED = ["raw_recognition.json", "score.json", "score.schema.json", "score_validation.json",
            "score.svg", "score.pdf"]
REQUIRED_RENDER_ONLY = ["score.schema.json", "score_validation.json", "score.svg", "score.pdf"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0] if args else "output"
    allow_no_js = "--allow-missing-jsonschema" in sys.argv
    required = REQUIRED_RENDER_ONLY if "--render-only" in sys.argv else REQUIRED
    bad = []
    for name in required:
        p = os.path.join(out, name)
        ok = os.path.isfile(p) and os.path.getsize(p) > 0
        print(("PASS " if ok else "FAIL ") + name)
        if not ok:
            bad.append(f"missing or empty: {name}")
    if os.path.isfile(os.path.join(out, "score_validation.json")):
        rep = json.load(open(os.path.join(out, "score_validation.json"), encoding="utf-8"))
        sv = rep.get("schema_valid")
        if sv is True:
            print("PASS score.json schema validation (jsonschema)")
        elif sv is None and allow_no_js:
            print("SKIP score.json schema validation (jsonschema not installed)")
        else:
            bad.append("score.json schema validation failed or was not run: "
                       + "; ".join(rep.get("schema_errors", [])[:5]))
        for e in rep.get("errors", []):
            bad.append(f"validation error: {e}")
        for w in rep.get("warnings", [])[:10]:
            print("WARN", w)
    if bad:
        print("\nCHECK FAILED")
        for b in bad:
            print(" -", b)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
