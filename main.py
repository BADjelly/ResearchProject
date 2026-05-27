import json
import time
import csv
import os
import requests
from tqdm import tqdm
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

headers = {
  "Authorization": "Bearer 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",
  "Accept": "application/json"
}

#OLLAMA_URL = "http://localhost:11434/api/generate"
#OLLAMA_URL = "http://192.168.178.127:11434/api/generate" #iPhone

MODELS = [
    "google/gemma-3n-e4b-it",
    "google/gemma-3n-e2b-it",
    # "gemma4:e2b",
    # "gemma4:e4b",
    # "gemma4:26b",
]

REPETITIONS = 3

SCENARIO_FILE = "res/scenarios.json"
PROFILE_FILE = "res/profiles.json"

OUTPUT_FILE = "results.csv"

START_INDEX = 0
END_INDEX = None   # None = until end

# #Laptop:
# START_INDEX = 0
# END_INDEX = 5000

# #Mac Mini:
# START_INDEX = 5000
# END_INDEX = None

TEMPERATURE = 0.2

# SEED = 42
# random.seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

with open(SCENARIO_FILE, "r", encoding="utf-8") as f:
    scenarios = json.load(f)

with open(PROFILE_FILE, "r", encoding="utf-8") as f:
    profiles = json.load(f)


# ============================================================
# PROMPT BUILDERS
# ============================================================

PHASE1_LAYOUTS = [
    "instructions_scenario_options",
    "instructions_options_scenario",
    "scenario_options_instructions",
    "options_scenario_instructions"
]

PHASE2_LAYOUTS = [
    "instructions_profile_scenario_options",
    "instructions_profile_options_scenario",
    "instructions_scenario_profile_options",
    "instructions_scenario_options_profile",
    "instructions_options_profile_scenario",
    "instructions_options_scenario_profile",
    "profile_scenario_options_instructions",
    "profile_options_scenario_instructions",
    "scenario_profile_options_instructions",
    "scenario_options_profile_instructions",
    "options_profile_scenario_instructions",
    "options_scenario_profile_instructions"
]

OPTION_KEYS = [
    "option_a_strong",
    "option_a_moderate",
    "option_a_mild",
    "option_b_mild",
    "option_b_moderate",
    "option_b_strong"
]


def build_options_text(scenario):

    letters = ["A", "B", "C", "D", "E", "F"]

    lines = []

    for i, key in enumerate(OPTION_KEYS):
        lines.append(
            f"{letters[i]}. {scenario[key]}"
        )

    return "\n".join(lines)


def build_profile_text(profile):

    return f"""
        Psychological profile (Maslow needs):
        
        Physiological: {profile['Physiological']}
        Safety: {profile['Safety']}
        Belonging: {profile['Belonging']}
        Esteem: {profile['Esteem']}
        Self-actualization: {profile['Self-actualization']}
        
        Higher values indicate stronger motivational pressure.
        """.strip()


def build_prompt(
        scenario,
        profile=None,
        layout=None):

    scenario_text = scenario["scenario_text"]

    options_text = build_options_text(scenario)

    profile_text = ""

    if profile:
        profile_text = build_profile_text(profile)

    sections = {
        "instructions": f"""
            You are participating in a behavioural prediction study.
        
            Your task:
            
            Predict which action a human would most likely choose.
            
            Choose exactly one option:
            A, B, C, D, E, or F.
            
            Do not explain.
            
            Return only one letter.
            """.strip(),

        "profile": f"""
            PSYCHOLOGICAL PROFILE:
            
            {profile_text}
            """.strip(),

        "scenario": f"""
            SCENARIO:
            
            {scenario_text}
            """.strip(),

        "options": f"""
            OPTIONS:
            
            {options_text}
            """.strip()
    }

    parts = []

    for section_name in layout.split("_"):
        if section_name == "profile" and not profile:
            continue
        parts.append(sections[section_name])

    prompt = "\n\n".join(parts)

    return prompt.strip()


# ============================================================
# NVIDIA
# ============================================================

def call_nvidia(model, prompt):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8, # 512 is maybe too high for max output tokens
        "temperature": TEMPERATURE,
        "top_p": 0.70,
        "frequency_penalty": 0.00,
        "presence_penalty": 0.00,
        "stream": False
    }

    response = requests.post(invoke_url, headers=headers, json=payload, timeout=(30, 120))

    response.raise_for_status() # maybe needs to be deleted

    data = response.json()

    # print(data)

    text = data["choices"][0]["message"]["content"].strip()

    return text

def call_nvidia_with_retry(model, prompt, max_retries=5, base_sleep=1.0):

    for attempt in range(max_retries):

        try:

            return call_nvidia(model, prompt)

        except requests.exceptions.HTTPError as e:

            status = None

            if e.response is not None:

                status = e.response.status_code

            # retry only on transient errors

            if status in [429, 500, 502, 503, 504]:

                sleep_time = base_sleep * (2 ** attempt)  # exponential backoff

                print(f"[Retry {attempt+1}/{max_retries}] HTTP {status}. Sleeping {sleep_time:.1f}s...")

                time.sleep(sleep_time)

                continue

            # non-retryable error

            raise

        except requests.exceptions.RequestException as e:

            # network-level failure

            sleep_time = base_sleep * (2 ** attempt)

            print(f"[Retry {attempt+1}/{max_retries}] Network error: {e}. Sleeping {sleep_time:.1f}s...")

            time.sleep(sleep_time)

    raise RuntimeError("Max retries exceeded for NVIDIA API call")


# ============================================================
# PARSING
# ============================================================

VALID = {"A", "B", "C", "D", "E", "F"}


def parse_choice(text):

    text = text.strip().upper()

    for c in text:
        if c in VALID:
            return c

    return "INVALID"


# ============================================================
# SAVE
# ============================================================

def init_csv():

    if Path(OUTPUT_FILE).exists():
        return

    with open(
            OUTPUT_FILE,
            "w",
            newline="",
            encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "trial_index",
            "trial_id",
            "model",
            "scenario_index",
            "profile_id",
            "baseline",
            "prompt_order",
            "repetition",
            "raw_output",
            "parsed_choice"
        ])


# def save_result(row):
#
#     with open(
#             OUTPUT_FILE,
#             "a",
#             newline="",
#             encoding="utf-8") as f:
#
#         writer = csv.writer(f)
#
#         writer.writerow(row)

def calculate_total_prompts():

    scenario_count = len(scenarios)
    profile_count = len(profiles)
    model_count = len(MODELS)

    phase1_total = model_count * scenario_count * REPETITIONS * len(PHASE1_LAYOUTS)

    phase2_total = model_count * scenario_count * profile_count * REPETITIONS * len(PHASE2_LAYOUTS)

    return phase1_total + phase2_total

def build_trial_id(
        model,
        scenario_index,
        profile_id,
        baseline,
        prompt_order,
        repetition):

    return (
        f"{model}|"
        f"{scenario_index}|"
        f"{profile_id}|"
        f"{baseline}|"
        f"{prompt_order}|"
        f"{repetition}"
    )

def load_completed_trials():

    completed = set()

    if not os.path.exists(OUTPUT_FILE):
        return completed

    with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:
            completed.add(
                row["trial_id"]
            )

    print(
        f"Found {len(completed)} completed trials."
    )

    return completed


# ============================================================
# EXPERIMENT
# ============================================================

def run():

    init_csv()

    completed_trials = load_completed_trials()

    total_prompts = calculate_total_prompts()

    print(f"\nTotal prompts: {total_prompts}")

    global_prompt_index = 0

    with tqdm(
            total=total_prompts,
            initial=len(completed_trials),
            desc="Experiment",
            unit="prompt",
            bar_format=(
                    "\033[92m"
                    "{desc}: "
                    "{percentage:6.2f}%|"
                    "{bar}"
                    "| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                    "\033[0m"
            )
    ) as pbar:

        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as out_f:
            resultswriter = csv.writer(out_f)

            for model in MODELS:

                # ==================================================
                # PHASE 1 — BASELINE ONLY
                # ==================================================

                print("\nStarting PHASE 1: BASELINE")


                for s_idx, scenario in enumerate(scenarios):

                    for rep in range(REPETITIONS):

                        for order_name in PHASE1_LAYOUTS:

                            trial_id = build_trial_id(
                                model=model,
                                scenario_index=s_idx,
                                profile_id="",
                                baseline=True,
                                prompt_order=order_name,
                                repetition=rep
                            )

                            if global_prompt_index < START_INDEX:
                                global_prompt_index += 1
                                continue

                            if END_INDEX is not None and global_prompt_index >= END_INDEX:
                                return

                            if trial_id not in completed_trials:

                                pbar.set_postfix({
                                    "phase": "baseline",
                                    "model": model,
                                    "scenario": s_idx,
                                    "rep": rep
                                })

                                prompt = build_prompt(
                                    scenario,
                                    profile=None,
                                    layout=order_name
                                )

                                raw = call_nvidia_with_retry(model, prompt)

                                choice = parse_choice(raw)

                                # save_result([
                                #     global_prompt_index,
                                #     trial_id,
                                #     model,
                                #     s_idx,
                                #     "",
                                #     True,
                                #     order_name,
                                #     rep,
                                #     raw,
                                #     choice
                                # ])

                                resultswriter.writerow([
                                    global_prompt_index,
                                    trial_id,
                                    model,
                                    s_idx,
                                    "",
                                    True,
                                    order_name,
                                    rep,
                                    raw,
                                    choice
                                ])

                                completed_trials.add(
                                    trial_id
                                )

                                time.sleep(0.1)

                                pbar.update(1)

                            global_prompt_index += 1

                # ==================================================
                # PHASE 2 — PROFILES
                # ==================================================

                print("\nStarting PHASE 2: PSYCHOLOGICAL PROFILES")

                for profile in profiles:

                    for s_idx, scenario in enumerate(scenarios):

                        profile_id = profile[
                            "profile_id"
                        ]

                        for rep in range(REPETITIONS):

                            for order_name in PHASE2_LAYOUTS:

                                trial_id = build_trial_id(
                                    model=model,
                                    scenario_index=s_idx,
                                    profile_id=profile_id,
                                    baseline=False,
                                    prompt_order=order_name,
                                    repetition=rep
                                )

                                if global_prompt_index < START_INDEX:
                                    global_prompt_index += 1
                                    continue

                                if END_INDEX is not None and global_prompt_index >= END_INDEX:
                                    return

                                if trial_id not in completed_trials:

                                    pbar.set_postfix({
                                        "phase": "profile",
                                        "model": model,
                                        "scenario": s_idx,
                                        "profile": profile_id,
                                        "rep": rep
                                    })

                                    prompt = build_prompt(
                                        scenario,
                                        profile=profile,
                                        layout=order_name
                                    )

                                    raw = call_nvidia_with_retry(model, prompt)

                                    choice = parse_choice(raw)

                                    # save_result([
                                    #     global_prompt_index,
                                    #     trial_id,
                                    #     model,
                                    #     s_idx,
                                    #     profile_id,
                                    #     False,
                                    #     order_name,
                                    #     rep,
                                    #     raw,
                                    #     choice
                                    # ])

                                    resultswriter.writerow([
                                        global_prompt_index,
                                        trial_id,
                                        model,
                                        s_idx,
                                        profile_id,
                                        False,
                                        order_name,
                                        rep,
                                        raw,
                                        choice
                                    ])

                                    completed_trials.add(
                                        trial_id
                                    )

                                    time.sleep(0.1)

                                    pbar.update(1)

                                global_prompt_index += 1

if __name__ == "__main__":
    run()