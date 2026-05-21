import json
import uuid
import random
import time
import requests
import csv
import os
import threading
from tqdm import tqdm
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

#OLLAMA_URL = "http://localhost:11434/api/generate"
#OLLAMA_URL = "http://192.168.178.127:11434/api/generate" #iPhone

OLLAMA_HOSTS = [
    {
        "name": "Laptop1",
        "url": "http://localhost:11434/api/generate",
    },
    # {
    #     "name": "Laptop2",
    #     "url": "http://localhost:11434/api/generate",
    # },
    {
        "name": "MacMini",
        "url": "http://192.168.178.57:11434/api/generate",
    },
]

MODELS = [
    "gemma4:e2b",
    # "gemma4:e4b",
    # "gemma4:26b",
]

REPETITIONS = 3

SCENARIO_FILE = "res/scenarios.json"
PROFILE_FILE = "res/profiles.json"

OUTPUT_FILE = "results.csv"

# START_INDEX = 0
# END_INDEX = None   # None = until end

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
        scenario_first=True):

    scenario_text = scenario["scenario_text"]

    options_text = build_options_text(scenario)

    profile_text = ""

    if profile:
        profile_text = build_profile_text(profile)

    if scenario_first:

        body = f"""
SCENARIO:

{scenario_text}

OPTIONS:

{options_text}
"""

    else:

        body = f"""
OPTIONS:

{options_text}

SCENARIO:

{scenario_text}
"""

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

def call_ollama(host, model, prompt):

    payload = {
        "model": model,
        "prompt": prompt,

        # important:
        # no memory between calls
        # no context passed

        "stream": False,

        "options": {
            "temperature": TEMPERATURE
        }
    }

    response = requests.post(
        host["url"],
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
    completed_buffer[row[0]] = row

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

def load_processed_trial_indexes():

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
                int(row["trial_index"])
            )

    # print(f"Found {len(completed)} completed trial indexes.")

    return completed


trials_in_work_Lock = threading.Lock()
trials_in_work = set()

def worker(host): #running on each thread
    global_prompt_index = 0

    # ==================================================
    # PHASE 1 — BASELINE ONLY
    # ==================================================

    print("\n[Thread " + host["name"] + "] Starting PHASE 1: BASELINE")

    for model in MODELS:

        for s_idx, scenario in enumerate(scenarios):

            for rep in range(REPETITIONS):

                for order in [True, False]:

                    order_name = (
                        "scenario_first"
                        if order
                        else "options_first"
                    )

                    trial_id = build_trial_id(
                        model=model,
                        scenario_index=s_idx,
                        profile_id="",
                        baseline=True,
                        prompt_order=order_name,
                        repetition=rep
                    )

                    trials_in_work_Lock.acquire()
                    if trial_id not in completed_trials and trial_id not in trials_in_work:
                        trials_in_work.add(
                            trial_id
                        )
                        trials_in_work_Lock.release()
                        # progress_bar.set_postfix({
                        #     "phase": "baseline",
                        #     "model": model,
                        #     "scenario": s_idx,
                        #     "rep": rep
                        # })

                        prompt = build_prompt(
                            scenario,
                            profile=None,
                            scenario_first=order
                        )

                        raw = call_ollama(
                            host,
                            model,
                            prompt
                        )

                        choice = parse_choice(raw)

                        trials_in_work.remove(trial_id)

                        completed_trials.add(trial_id)

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

                        time.sleep(0.1)

                        # progress_bar.update(1)
                    else:
                        trials_in_work_Lock.release()

                    global_prompt_index = global_prompt_index + 1

    # ==================================================
    # PHASE 2 — PROFILES
    # ==================================================

    print("\n[Thread " + host["name"] + "] Starting PHASE 2: PSYCHOLOGICAL PROFILES")

    for model in MODELS:

        for profile in profiles:

            for s_idx, scenario in enumerate(scenarios):

                profile_id = profile[
                    "profile_id"
                ]

                for rep in range(REPETITIONS):

                    for order in [True, False]:

                        order_name = (
                            "scenario_first"
                            if order
                            else "options_first"
                        )

                        trial_id = build_trial_id(
                            model=model,
                            scenario_index=s_idx,
                            profile_id=profile_id,
                            baseline=False,
                            prompt_order=order_name,
                            repetition=rep
                        )

                        trials_in_work_Lock.acquire()
                        if trial_id not in completed_trials and trial_id not in trials_in_work:
                            trials_in_work.add(
                                trial_id
                            )
                            trials_in_work_Lock.release()
                            # progress_bar.set_postfix({
                            #     "phase": "profile",
                            #     "model": model,
                            #     "scenario": s_idx,
                            #     "profile": profile_id,
                            #     "rep": rep
                            # })

                            prompt = build_prompt(
                                scenario,
                                profile=profile,
                                scenario_first=order
                            )

                            raw = call_ollama(
                                host,
                                model,
                                prompt
                            )

                            choice = parse_choice(raw)

                            trials_in_work.remove(trial_id)

                            completed_trials.add(trial_id)

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

                            time.sleep(0.1)

                            # progress_bar.update(1)
                        else:
                            trials_in_work_Lock.release()

                        global_prompt_index = global_prompt_index + 1


# ============================================================
# EXPERIMENT
# ============================================================

completed_buffer = {}

def run():
    global completed_trials

    init_csv()

    completed_trials = load_completed_trials()

    processed_trial_indexes = load_processed_trial_indexes()

    total_prompts = calculate_total_prompts()

    print(f"\nTotal prompts: {total_prompts}")

    # Start threads for each host
    threads = []
    for host in OLLAMA_HOSTS:
        # Using `args` to pass positional arguments
        t = threading.Thread(target=worker, args=(host,))
        threads.append(t)
        t.start()

    with open(
            OUTPUT_FILE,
            "a",
            newline="",
            encoding="utf-8") as f:

        writer = csv.writer(f)

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
        ) as progress_bar:
            prev_completed_trials = len(completed_trials)
            current_completed_trials = prev_completed_trials
            next_write_index = 0
            while next_write_index in processed_trial_indexes:
                next_write_index += 1
            while current_completed_trials < total_prompts:
                progress_bar.update(current_completed_trials-prev_completed_trials)
                prev_completed_trials = current_completed_trials
                time.sleep(5)
                current_completed_trials = len(completed_trials)
                while next_write_index in completed_buffer:
                    writer.writerow(completed_buffer[next_write_index])
                    del completed_buffer[next_write_index]
                    processed_trial_indexes.add(next_write_index)
                    while next_write_index in processed_trial_indexes:
                        next_write_index += 1

    # Wait for all threads to finish
    for t in threads:
        t.join()

if __name__ == "__main__":
    run()