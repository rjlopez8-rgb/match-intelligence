"""First call to Microsoft Foundry, through the budget guard.

Needs two environment variables (set them on your computer, never in code or on GitHub):
    AZURE_OPENAI_ENDPOINT   e.g. https://rjlopez8-5433-resource.openai.azure.com
    AZURE_OPENAI_API_KEY    the key from the Foundry home page

Run:  python scripts/smoke_test.py gpt-6-luna-1 gpt-6-sol-1
Each call costs a fraction of a cent.
"""

from __future__ import annotations

import os
import sys
import time

from openai import OpenAI

from matchagents import BudgetExceeded, BudgetGuard


def main(deployments: list[str]) -> int:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    # accept the endpoint however it was copied from the portal
    for suffix in ("/openai/v1", "/openai"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
    key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    if not endpoint or not key:
        print("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY first (see the walkthrough).")
        return 2
    client = OpenAI(base_url=f"{endpoint}/openai/v1/", api_key=key)
    guard = BudgetGuard()
    print(f"Budget guard: ${guard.remaining():.2f} left of today's ${guard.cap:.2f} cap\n")
    for dep in deployments:
        try:
            guard.check(dep, expected_input_tokens=60, expected_output_tokens=200)
        except BudgetExceeded as exc:
            print(exc)
            return 3
        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": "You are a football commentator. One short sentence."},
            {"role": "user", "content": "Riverside lead Kingsport 1-0 after 74 minutes. Say something."},
        ]
        try:
            r = client.chat.completions.create(
                model=dep, messages=messages, max_completion_tokens=200, reasoning_effort="low"
            )
        except Exception as exc:  # some models do not accept reasoning_effort
            if "reasoning" not in str(exc).lower():
                raise
            r = client.chat.completions.create(model=dep, messages=messages, max_completion_tokens=200)
        dt = time.perf_counter() - t0
        u = r.usage
        cost = guard.record(dep, u.prompt_tokens, u.completion_tokens)
        print(f"{dep}: {dt:.1f}s, {u.prompt_tokens} in / {u.completion_tokens} out, ~${cost:.5f}")
        print(f'  "{(r.choices[0].message.content or "").strip()}"\n')
    print(guard.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["gpt-6-luna-1"]))
