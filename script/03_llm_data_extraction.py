#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: LLM-based Data Extraction
===================================
Extracts structured antibiotic resistance data from Markdown articles
using large language models.

Two backends are supported:
  1. "openai"    - any OpenAI-compatible HTTP API (OpenAI, DeepSeek,
                   local vLLM/Ollama servers, etc.). Requires an API key.
  2. "codebuddy" - drives the local CodeBuddy CLI (`codebuddy -p ...`) as
                   the LLM engine, authenticated via a CODEBUDDY_API_KEY
                   you supply (not the interactively-logged-in session).
                   Handy for quick testing with a CodeBuddy API key.

Usage (OpenAI-compatible API):
    python 03_llm_data_extraction.py \
        --input ../data_demo/2_convert_md_pmc \
        --output ../data_demo/3_response_pmc \
        --prompt ../script/prompt/p_resistance_extract.txt \
        --prompt-merge ../script/prompt/p_resistance_merge.txt \
        --model deepseek-v3 \
        --api-url https://api.deepseek.com/v1/chat/completions \
        --api-key $LLM_API_KEY

Usage (CodeBuddy backend, reuses your current CLI login session - no API key needed):
    python 03_llm_data_extraction.py \
        --input ../data_demo/2_convert_md_pmc \
        --output ../data_demo/3_response_pmc \
        --prompt ../script/prompt/p_resistance_extract.txt \
        --prompt-merge ../script/prompt/p_resistance_merge.txt \
        --llm-backend codebuddy \
        --model claude-sonnet-5

Usage (CodeBuddy backend, with an explicit static API key instead of session login):
    export CODEBUDDY_API_KEY="your-codebuddy-api-key"
    # China edition also needs:
    export CODEBUDDY_INTERNET_ENVIRONMENT=internal
    python 03_llm_data_extraction.py \
        --input ../data_demo/2_convert_md_pmc \
        --output ../data_demo/3_response_pmc \
        --prompt ../script/prompt/p_resistance_extract.txt \
        --prompt-merge ../script/prompt/p_resistance_merge.txt \
        --llm-backend codebuddy \
        --model claude-sonnet-5

Environment Variables:
    LLM_API_KEY                    - API key for the LLM service (required for "openai" backend)
    LLM_API_URL                    - API endpoint URL (optional, can use --api-url)
    LLM_MODEL                      - Model name (optional, can use --model)
    LLM_BACKEND                    - "openai" (default) or "codebuddy" (optional, can use --llm-backend)
    CODEBUDDY_API_KEY              - CodeBuddy API key (OPTIONAL for "codebuddy" backend; omit it to
                                      reuse the current CLI login session instead, e.g. iOA SSO)
    CODEBUDDY_INTERNET_ENVIRONMENT - "internal" (China edition) / "ioa" (iOA enterprise) /
                                      unset (International). Only relevant when using a static API key.
"""

import re
import os
import time
import json
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Optional

import requests

# === Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# Core functions
# ==============================================================================

def remove_references(content: str) -> str:
    """Remove the references section from article content, preserving supplementary."""
    ref_pattern = r"(#+\s*(■\s*)?\[?(References|Reference|REFERENCES|LITERATURE CITED|Notes and references)\]?[^\n]*)"
    post_ref_patterns = [
        r"(#+\s*SUPPLEMENTARY[^\n]*)",
        r"(#+\s*Appendix[^\n]*)",
        r"(#+\s*Tables[^\n]*)",
    ]

    ref_match = re.search(ref_pattern, content, re.IGNORECASE)
    if not ref_match:
        return content

    start_pos = ref_match.start()
    end_pos = None
    for p in post_ref_patterns:
        match = re.search(p, content[ref_match.end():], re.IGNORECASE)
        if match:
            end_pos = ref_match.end() + match.start()
            break

    return content[:start_pos] + (content[end_pos:] if end_pos else "")


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 word ≈ 1.5 tokens)."""
    return int(len(text.split()) * 1.5)


def call_llm_api(messages: list, config: dict) -> Optional[str]:
    """
    Send a request to an OpenAI-compatible chat API.
    
    Args:
        messages: List of message dicts [{"role": ..., "content": ...}]
        config: Dict with keys: api_url, api_key, model, temperature, max_tokens
    
    Returns:
        Response text or None on failure.
    """
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
    }

    try:
        response = requests.post(config["api_url"], headers=headers, json=payload, timeout=300)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"API error {response.status_code}: {response.text[:500]}")
            return None
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return None


def call_codebuddy_cli(messages: list, config: dict, retries: int = 2) -> Optional[str]:
    """
    Send a request via the local `codebuddy` CLI non-interactive mode (`-p`).

    Two authentication modes are supported:
      1. Session auth (recommended when running inside an already-logged-in
         CodeBuddy IDE/CLI session, e.g. iOA SSO): simply leave
         CODEBUDDY_API_KEY / --codebuddy-api-key unset. The CLI subprocess
         will inherit the current shell's login session and authenticate
         automatically, exactly like running `codebuddy -p "..."` by hand.
      2. Static API Key auth: pass CODEBUDDY_API_KEY (or --codebuddy-api-key)
         explicitly. This OVERRIDES session auth in the subprocess env, so
         only use it if you have a valid CodeBuddy API key for the target
         environment (internal/ioa/international) - an invalid/mismatched
         key will cause 401 even if session auth would have worked.

    Optional env vars (can also be passed via config / CLI flags, see main()):
        CODEBUDDY_API_KEY              - your CodeBuddy API key (optional; omit to use session auth)
        CODEBUDDY_INTERNET_ENVIRONMENT - "internal" (China) / "ioa" (iOA) / unset (Intl)

    Args:
        messages: List of message dicts [{"role": ..., "content": ...}]
        config: Dict with keys: model, codebuddy_api_key,
                codebuddy_internet_environment, codebuddy_base_url, timeout

    Returns:
        Response text or None on failure.
    """
    # `codebuddy -p` takes a single prompt; flatten system+user messages.
    prompt = "\n\n".join(m["content"] for m in messages if m.get("content"))

    # Inherit the current environment (including any active login session)
    # by default. Only override with a static API key / custom env / base
    # URL if the user explicitly provided one - this way, running inside an
    # already-authenticated session (e.g. iOA SSO) keeps working out of the
    # box without requiring a separate API key.
    env = os.environ.copy()
    if config.get("codebuddy_api_key"):
        env["CODEBUDDY_API_KEY"] = config["codebuddy_api_key"]
    if config.get("codebuddy_internet_environment"):
        env["CODEBUDDY_INTERNET_ENVIRONMENT"] = config["codebuddy_internet_environment"]
    if config.get("codebuddy_base_url"):
        env["CODEBUDDY_BASE_URL"] = config["codebuddy_base_url"]

    cmd = [
        "codebuddy", "-p",
        "--permission-mode", "bypassPermissions",
        "--tools", "",  # pure text extraction task; no built-in tools needed
    ]
    if config.get("model"):
        cmd += ["--model", config["model"]]

    timeout = config.get("timeout", 300)
    for attempt in range(1, retries + 1):
        logger.info(f"  → Invoking codebuddy CLI (attempt {attempt}/{retries}, "
                    f"prompt≈{len(prompt)} chars, timeout={timeout}s)...")
        t0 = time.time()
        try:
            result = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"codebuddy CLI call timed out after {time.time() - t0:.0f}s "
                         f"(attempt {attempt}/{retries})")
            continue
        except FileNotFoundError:
            logger.error("`codebuddy` CLI not found on PATH. Please install/configure it first.")
            return None

        elapsed = time.time() - t0
        if result.returncode != 0:
            logger.error(f"codebuddy CLI exited with code {result.returncode} after {elapsed:.0f}s "
                         f"(attempt {attempt}/{retries}): {result.stderr[:500]}")
        else:
            output = result.stdout.strip()
            if output:
                logger.info(f"  ← codebuddy CLI responded in {elapsed:.0f}s")
                return output
            logger.warning(f"codebuddy CLI returned empty output after {elapsed:.0f}s "
                            f"(attempt {attempt}/{retries}). stderr: {result.stderr[:300]}")

        if attempt < retries:
            time.sleep(5 * attempt)

    return None


def call_llm(messages: list, config: dict) -> Optional[str]:
    """Dispatch to the configured LLM backend ("openai" or "codebuddy")."""
    if config.get("backend") == "codebuddy":
        return call_codebuddy_cli(messages, config)
    return call_llm_api(messages, config)


def extract_single_pass(content: str, prompt: str, config: dict) -> Optional[str]:
    """Single-pass extraction for short articles."""
    messages = [
        {"role": "system", "content": "You are an expert in information extraction from scientific literature."},
        {"role": "user", "content": f"The following is a scientific article:\n{content}\n\n{prompt}"},
    ]
    return call_llm(messages, config)


def extract_chunked_merge(content: str, prompt_extract: str, prompt_merge: str, config: dict) -> Optional[str]:
    """Multi-chunk extraction + merge for long articles."""
    chunk_size = 120000  # characters
    all_responses = []

    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        messages = [
            {"role": "system", "content": "You are an expert in information extraction from scientific literature."},
            {"role": "user", "content": f"The following is a scientific article:\n{chunk}\n\n{prompt_extract}"},
        ]
        logger.info(f"  Extracting chunk {i // chunk_size + 1}")
        time.sleep(3)
        response = call_llm(messages, config)
        if response:
            all_responses.append(response)
        else:
            logger.warning(f"  Chunk {i // chunk_size + 1} failed")

    if not all_responses:
        return None

    # Merge step
    merged_text = "\n".join(all_responses)
    messages = [
        {"role": "system", "content": "You are an expert in information extraction from scientific literature."},
        {"role": "user", "content": f"Provided extracted data:\n'''\n{merged_text}\n'''\n\n{prompt_merge}"},
    ]
    logger.info("  Merging extracted chunks...")
    time.sleep(3)
    return call_llm(messages, config)


# ==============================================================================
# Main pipeline
# ==============================================================================

def process_paper(md_path: str, output_dir: str, prompt_extract: str, prompt_merge: str, config: dict) -> bool:
    """Process a single paper: read → remove refs → extract → save."""
    paper_id = Path(md_path).stem

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove references section
    content = remove_references(content)

    # Guard: skip LLM call entirely if the article body is (near-)empty.
    # This typically means Step 1/2 failed to fetch/convert the full-text
    # PDF (e.g. only a title + supplementary files were downloaded), and
    # feeding an empty document to the LLM tends to produce hallucinated
    # / garbage output instead of a clean "no data" result.
    min_chars = config.get("min_content_chars", 200)
    if len(content.strip()) < min_chars:
        logger.warning(f"  ✗ Skipped {paper_id}: article body too short "
                        f"({len(content.strip())} chars < {min_chars}) — likely missing "
                        f"fulltext.pdf from Step 1. Not calling LLM.")
        return False

    token_est = estimate_tokens(content) + config["max_tokens"]

    # Choose extraction strategy
    if token_est > 65000 and prompt_merge:
        logger.info(f"  Long article ({token_est} est. tokens) → chunked extraction")
        result = extract_chunked_merge(content, prompt_extract, prompt_merge, config)
    else:
        result = extract_single_pass(content, prompt_extract, config)

    if result:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"response_{paper_id}.csv")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        logger.info(f"  ✓ Saved: {output_path}")
        return True
    else:
        logger.warning(f"  ✗ No response for {paper_id}")
        return False


def main():
    parser = argparse.ArgumentParser(description="LLM-based antibiotic resistance data extraction")
    parser.add_argument("--input", required=True, help="Directory containing .md files")
    parser.add_argument("--output", required=True, help="Output directory for extraction results")
    parser.add_argument("--prompt", required=True, help="Path to extraction prompt file")
    parser.add_argument("--prompt-merge", default=None, help="Path to merge prompt file (for long articles)")
    parser.add_argument("--model", default=None, help="LLM model name (or set LLM_MODEL env var)")
    parser.add_argument("--api-url", default=None, help="API endpoint URL (or set LLM_API_URL env var)")
    parser.add_argument("--api-key", default=None, help="API key (or set LLM_API_KEY env var)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (default: 0.1)")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max output tokens (default: 8192)")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay between API calls in seconds")
    parser.add_argument("--llm-backend", choices=["openai", "codebuddy"], default=None,
                        help="LLM backend to use: 'openai' (default, HTTP API) or "
                             "'codebuddy' (local CodeBuddy CLI). Or set LLM_BACKEND env var.")
    parser.add_argument("--codebuddy-api-key", default=None,
                        help="CodeBuddy API key for the 'codebuddy' backend "
                             "(or set CODEBUDDY_API_KEY env var)")
    parser.add_argument("--codebuddy-internet-environment", default=None,
                        help="'internal' (China edition) / 'ioa' (iOA enterprise) / unset (International). "
                             "Required alongside the CodeBuddy API key "
                             "(or set CODEBUDDY_INTERNET_ENVIRONMENT env var)")
    parser.add_argument("--codebuddy-base-url", default=None,
                        help="Override CodeBuddy API endpoint, for third-party model services "
                             "(or set CODEBUDDY_BASE_URL env var)")
    args = parser.parse_args()

    # === Resolve configuration ===
    backend = args.llm_backend or os.environ.get("LLM_BACKEND", "openai")
    config = {
        "backend": backend,
        "api_url": args.api_url or os.environ.get("LLM_API_URL"),
        "api_key": args.api_key or os.environ.get("LLM_API_KEY"),
        "model": args.model or os.environ.get("LLM_MODEL", "deepseek-v3" if backend == "openai" else "claude-sonnet-5"),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "codebuddy_api_key": args.codebuddy_api_key or os.environ.get("CODEBUDDY_API_KEY"),
        "codebuddy_internet_environment": args.codebuddy_internet_environment or os.environ.get("CODEBUDDY_INTERNET_ENVIRONMENT"),
        "codebuddy_base_url": args.codebuddy_base_url or os.environ.get("CODEBUDDY_BASE_URL"),
    }

    if backend == "codebuddy":
        if config["codebuddy_api_key"]:
            logger.info(f"Using CodeBuddy CLI backend (model={config['model']}, "
                        f"env={config['codebuddy_internet_environment'] or 'international/default'}, "
                        f"auth=static API key)")
        else:
            logger.info(f"Using CodeBuddy CLI backend (model={config['model']}) "
                        f"— no CODEBUDDY_API_KEY set, will rely on the current CLI login "
                        f"session (e.g. iOA SSO) for authentication.")
    else:
        if not config["api_url"]:
            logger.error("API URL not provided. Use --api-url or set LLM_API_URL environment variable.")
            logger.error("Examples:")
            logger.error("  OpenAI:    https://api.openai.com/v1/chat/completions")
            logger.error("  DeepSeek:  https://api.deepseek.com/v1/chat/completions")
            logger.error("  Local:     http://localhost:8000/v1/chat/completions")
            exit(1)

        if not config["api_key"]:
            logger.error("API key not provided. Use --api-key or set LLM_API_KEY environment variable.")
            exit(1)

    # === Load prompts ===
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt_extract = f.read()

    prompt_merge = None
    if args.prompt_merge:
        with open(args.prompt_merge, "r", encoding="utf-8") as f:
            prompt_merge = f.read()

    # === Get papers to process ===
    md_files = sorted([
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith(".md")
    ], key=os.path.getsize)

    # Check already processed
    done_papers = set()
    if os.path.exists(args.output):
        for root, dirs, files in os.walk(args.output):
            for f in files:
                if f.startswith("response_") and f.endswith(".csv"):
                    done_papers.add(f.replace("response_", "").replace(".csv", ""))

    pending = [f for f in md_files if Path(f).stem not in done_papers]
    logger.info(f"Total papers: {len(md_files)} | Already done: {len(done_papers)} | Pending: {len(pending)}")

    # === Process ===
    success, failed = 0, 0
    for i, md_path in enumerate(pending):
        paper_id = Path(md_path).stem
        logger.info(f"[{i+1}/{len(pending)}] Processing: {paper_id}")

        ok = process_paper(md_path, args.output, prompt_extract, prompt_merge, config)
        if ok:
            success += 1
        else:
            failed += 1

        time.sleep(args.delay)

    logger.info(f"\nExtraction complete. Success: {success} | Failed: {failed}")


if __name__ == "__main__":
    main()
