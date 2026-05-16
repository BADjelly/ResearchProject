import json
import uuid
import random
import time
import requests
import csv
import os
from tqdm import tqdm
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
#OLLAMA_URL = "http://192.168.178.127:11434/api/generate" #iPhone

MODELS = [
    "gemma4:e2b",
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

SEED = 42
random.seed(SEED)


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
    "scenario_options",
    "options_scenario"
]

PHASE2_LAYOUTS = [
    "profile_scenario_options",
    "profile_options_scenario",
    "scenario_profile_options",
    "scenario_options_profile",
    "options_profile_scenario",
    "options_scenario_profile"
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

    body = "\n\n".join(parts)


    prompt = f"""
        You are participating in a behavioural prediction study.
        
        Your task:
        
        Predict which action a human would most likely choose.
        
        Choose exactly one option:
        A, B, C, D, E, or F.
        
        Do not explain.
        
        Return only one letter.
        
        {profile_text}
        
        {body}
        """

    return prompt.strip()


# ============================================================
# OLLAMA
# ============================================================

def call_ollama(model, prompt):

    payload = {
        "model": model,
        "prompt": prompt,

        # important:
        # no memory between calls
        # no context passed

        "stream": False,
        "thinking": False,

        "options": {
            "temperature": TEMPERATURE
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    text = data["response"].strip()

    return text


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


def save_result(row):

    with open(
            OUTPUT_FILE,
            "a",
            newline="",
            encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(row)

def calculate_total_prompts():

    scenario_count = len(scenarios)
    profile_count = len(profiles)
    model_count = len(MODELS)

    prompt_orders = 2

    prompts_per_order = 1 + profile_count
    prompts_per_repetition = prompt_orders * prompts_per_order

    total = (
        model_count
        * scenario_count
        * REPETITIONS
        * prompts_per_repetition
    )

    return total

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
                    "{bar} "
                    "| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                    "\033[0m"
            )
    ) as pbar:

        # ==================================================
        # PHASE 1 — BASELINE ONLY
        # ==================================================

        print("\nStarting PHASE 1: BASELINE")

        for model in MODELS:

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

                            raw = call_ollama(
                                model,
                                prompt
                            )

                            choice = parse_choice(raw)

                            save_result([
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

        for model in MODELS:

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

                                raw = call_ollama(
                                    model,
                                    prompt
                                )

                                choice = parse_choice(raw)

                                save_result([
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