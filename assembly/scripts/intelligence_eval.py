#!/usr/bin/env python3
"""Run and objectively grade the complete intelligence suite through asm-chat."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from intelligence_cases import CASES, Case


DEFAULT_ENDPOINT = "http://127.0.0.1:8080/chat"
DEFAULT_RELAY_HEALTH = "http://127.0.0.1:8081/"
TIMEOUT_SECONDS = 75


def query(endpoint: str, prompt: str) -> tuple[int | None, str, float, str | None]:
    request = urllib.request.Request(
        endpoint,
        data=prompt.encode("utf-8"),
        method="POST",
        headers={"Content-Type": "text/plain; charset=utf-8"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8", "replace")
            return response.status, body, time.perf_counter() - started, None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        return exc.code, body, time.perf_counter() - started, f"HTTPError: {exc.reason}"
    except Exception as exc:
        return None, "", time.perf_counter() - started, f"{type(exc).__name__}: {exc}"


def grade(case: Case, response: str, status: int | None) -> tuple[bool, str]:
    if status != 200:
        return False, f"expected HTTP 200, got {status}"
    actual = response.strip().replace("\r\n", "\n")
    expected = case.expected.replace("\r\n", "\n")
    if case.grader == "json":
        try:
            actual_value = json.loads(actual)
            expected_value = json.loads(expected)
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc.msg}"
        if actual_value != expected_value:
            return False, f"JSON mismatch: expected {expected!r}, got {actual!r}"
        # Compact output is an explicit part of these test instructions.
        if actual != expected:
            return False, f"format mismatch: expected {expected!r}, got {actual!r}"
        return True, "exact JSON match"
    if actual == expected:
        return True, "exact match"
    return False, f"expected {expected!r}, got {actual!r}"


def parse_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return selected


def read_relay_health(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.read().decode("utf-8", "replace").strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--relay-health", default=DEFAULT_RELAY_HEALTH)
    parser.add_argument("--ids", help="Comma-separated IDs or ranges, e.g. 1-20,44")
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    selected_ids = parse_ids(args.ids)
    selected = [case for case in CASES if selected_ids is None or case.id in selected_ids]
    if not selected:
        parser.error("no cases selected")

    run_started = datetime.now(timezone.utc)
    relay_health = read_relay_health(args.relay_health)
    print(f"Relay: {relay_health}", flush=True)
    results = []
    for index, case in enumerate(selected, 1):
        status, response, latency, error = query(args.endpoint, case.prompt)
        passed, grade_detail = grade(case, response, status)
        item = {
            "id": case.id,
            "difficulty": case.difficulty,
            "category": case.category,
            "source": case.source,
            "prompt": case.prompt,
            "expected": case.expected,
            "grader": case.grader,
            "http_status": status,
            "latency_seconds": round(latency, 3),
            "response": response,
            "passed": passed,
            "grade_detail": grade_detail,
            "error": error,
        }
        results.append(item)
        marker = "PASS" if passed else "FAIL"
        print(
            f"[{index:04d}/{len(selected):04d}] id={case.id:04d} {marker} "
            f"{case.category:<23} status={status!s:<4} latency={latency:6.2f}s",
            flush=True,
        )
        if not passed:
            print(f"    {grade_detail}", flush=True)

    finished = datetime.now(timezone.utc)
    passed_count = sum(item["passed"] for item in results)
    latencies = [item["latency_seconds"] for item in results]
    summary = {
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "total": len(results),
        "pass_rate": round(passed_count / len(results), 6),
        "latency_total_seconds": round(sum(latencies), 3),
        "latency_mean_seconds": round(statistics.mean(latencies), 3),
        "latency_median_seconds": round(statistics.median(latencies), 3),
        "latency_max_seconds": max(latencies),
        "by_category": {
            category: {
                "passed": sum(
                    item["passed"] and item["category"] == category for item in results
                ),
                "failed": sum(
                    (not item["passed"]) and item["category"] == category
                    for item in results
                ),
                "total": sum(item["category"] == category for item in results),
            }
            for category in sorted({item["category"] for item in results})
        },
    }
    payload = {
        "label": args.label,
        "started_at": run_started.isoformat(),
        "finished_at": finished.isoformat(),
        "endpoint": args.endpoint,
        "relay_health": relay_health,
        "summary": summary,
        "results": results,
    }

    output_dir = Path(__file__).resolve().parent.parent / "out" / "evals"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = run_started.strftime("%Y%m%dT%H%M%SZ")
    stem = f"intelligence-{len(CASES)}-{args.label}-{stamp}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Assembly agent intelligence evaluation",
        "",
        f"- Label: `{args.label}`",
        f"- Started: {payload['started_at']}",
        f"- Finished: {payload['finished_at']}",
        f"- Score: **{passed_count}/{len(results)}**",
        f"- Relay: `{relay_health}`",
        f"- Mean latency: {summary['latency_mean_seconds']:.3f}s",
        "",
    ]
    for item in results:
        verdict = "PASS" if item["passed"] else "FAIL"
        lines.extend(
            [
                f"## {item['id']:03d}. {item['category']} — {verdict}",
                "",
                f"Source family: {item['source']}",
                "",
                f"**Prompt:** {item['prompt']}",
                "",
                f"**Expected:** `{item['expected']}`",
                "",
                f"**Result:** HTTP {item['http_status']}; {item['latency_seconds']:.3f}s; {item['grade_detail']}",
                "",
                "**Response:**",
                "",
                "```text",
                item["response"],
                "```",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"JSON: {json_path}", flush=True)
    print(f"Markdown: {md_path}", flush=True)
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
