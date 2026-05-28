"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time

    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    start_time = time.time()

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        latency = time.time() - start_time

        text = response.text or ""
        usage = {
            "input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
            "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
        }
        return text, latency, usage

    except (ImportError, Exception):
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model_inst = genai.GenerativeModel(model)
        config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens
        )
        response = model_inst.generate_content(prompt, generation_config=config)
        latency = time.time() - start_time

        text = response.text or ""
        try:
            input_tokens = model_inst.count_tokens(prompt).total_tokens
            output_tokens = model_inst.count_tokens(text).total_tokens
        except Exception:
            input_tokens = int(len(prompt.split()) * 1.5)
            output_tokens = int(len(text.split()) * 1.5)

        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        return text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    api_key = os.getenv("ANTHROPIC_API_KEY") or "mock-key"

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start_time

    text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": response.usage.input_tokens if response.usage else 0,
        "output_tokens": response.usage.output_tokens if response.usage else 0,
    }
    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    gpt4o_text, gpt4o_lat, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    gpt4o_cost = (
        gpt4o_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["input"] +
        gpt4o_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["output"]
    ) / 1_000_000

    mini_text, mini_lat, mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    mini_cost = (
        mini_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["input"] +
        mini_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["output"]
    ) / 1_000_000

    gemini_text, gemini_lat, gemini_usage = call_gemini(prompt, model=GEMINI_MODEL)
    gemini_cost = (
        gemini_usage["input_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["input"] +
        gemini_usage["output_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["output"]
    ) / 1_000_000

    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_lat,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"]
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_lat,
            "cost": mini_cost,
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"]
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_lat,
            "cost": gemini_cost,
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"]
        }
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    print("\n\033[94m==============================================")
    print("🤖 Vin Smart Future — Intelligent Chat Assistant")
    print("Powered by Google Gemini 2.5 Flash")
    print("Type 'quit' or 'exit' to end the session.")
    print("==============================================\033[0m\n")

    history = []

    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("\033[93mGoodbye!\033[0m")
                break

            messages_to_send = history[-6:] + [{"role": "user", "text": user_input}]
            print("\033[92mGemini:\033[0m ", end="", flush=True)

            full_reply = ""
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=GEMINI_API_KEY)
                formatted_contents = []
                for msg in messages_to_send:
                    formatted_contents.append(
                        types.Content(
                            role=msg["role"],
                            parts=[types.Part.from_text(text=msg["text"])]
                        )
                    )

                response_stream = client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=formatted_contents
                )
                for chunk in response_stream:
                    chunk_text = chunk.text or ""
                    print(chunk_text, end="", flush=True)
                    full_reply += chunk_text

            except (ImportError, Exception):
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                model_inst = genai.GenerativeModel(GEMINI_MODEL)

                legacy_contents = []
                for msg in messages_to_send:
                    legacy_contents.append({
                        "role": msg["role"] if msg["role"] == "user" else "model",
                        "parts": [msg["text"]]
                    })

                response_stream = model_inst.generate_content(legacy_contents, stream=True)
                for chunk in response_stream:
                    chunk_text = chunk.text or ""
                    print(chunk_text, end="", flush=True)
                    full_reply += chunk_text

            print("\n")
            history.append({"role": "user", "text": user_input})
            history.append({"role": "model", "text": full_reply})

        except KeyboardInterrupt:
            print("\n\033[93mSession interrupted. Goodbye!\033[0m")
            break
        except Exception as e:
            print(f"\n\033[91m[Error Calling API]: {e}\033[0m\n")


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
    if last_exception:
        raise last_exception


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for prompt in prompts:
        try:
            comp = compare_models(prompt)
            comp["prompt"] = prompt
            results.append(comp)
        except Exception as e:
            results.append({
                "prompt": prompt,
                "gpt4o": {"response": f"Error: {e}", "latency": 0.0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0},
                "gpt4o_mini": {"response": f"Error: {e}", "latency": 0.0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0},
                "gemini_flash": {"response": f"Error: {e}", "latency": 0.0, "cost": 0.0, "input_tokens": 0, "output_tokens": 0}
            })
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    table_lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for r in results:
        prompt = r["prompt"]
        if len(prompt) > 40:
            prompt = prompt[:37] + "..."

        models = [
            ("GPT-4o",       r.get("gpt4o")),
            ("GPT-4o-Mini",  r.get("gpt4o_mini")),
            ("Gemini-Flash", r.get("gemini_flash"))
        ]

        for name, data in models:
            if not data:
                continue

            resp  = data.get("response", "").replace("\n", " ")
            lat   = data.get("latency", 0.0)
            in_t  = data.get("input_tokens", 0)
            out_t = data.get("output_tokens", 0)
            cost  = data.get("cost", 0.0)

            if len(resp) > 50:
                resp = resp[:47] + "..."

            table_lines.append(
                f"| {prompt} | {name} | {resp} | {lat:.2f}s | {in_t}/{out_t} | ${cost:.6f} |"
            )

    return "\n".join(table_lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")