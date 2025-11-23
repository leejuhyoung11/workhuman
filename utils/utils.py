import signal
import tiktoken
import os, json
from dotenv import load_dotenv

load_dotenv()
enc = tiktoken.get_encoding("cl100k_base")
CONFIG_PATH = "./config/llm_providers.json"

def chunk_awards(rec_id, awards_list, max_tokens=40000):
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for award_idx, award in enumerate(awards_list):
        title = award.get("title", "").strip()
        message = award.get("message", "").strip()

        award_text = (
            f"{award_idx}#{title}|{message}\n\n"
        )

        award_tokens = len(enc.encode(award_text))

        # if adding this text exceeds limit → finalize current chunk
        if current_tokens + award_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = award_text
            current_tokens = award_tokens
        else:
            current_chunk += award_text
            current_tokens += award_tokens


    if current_chunk.strip():
        chunks.append(current_chunk)

    save_chunks(rec_id, chunks)

    return chunks



def save_chunks(employee_id, chunks, output_dir="output"):
    

    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"employee_{employee_id}_award_chunks.jsonl")

    with open(file_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} chunks to {file_path}")
    return file_path


def save_employee_signals(rec_id: int, results: dict, folder: str = "output"):
    os.makedirs(folder, exist_ok=True)

    save_path = f"{folder}/employee_{rec_id}_keywords.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Saved] {save_path}")
    return save_path

def save_clustering_result(rec_id: int, results, is_vp, folder: str = "output"):
    folder_path = f"{folder}/{is_vp}"
    os.makedirs(folder_path, exist_ok=True)

    save_path = f"{folder_path}/employee_{rec_id}_clustering_result.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Saved] {save_path}")
    return save_path

def save_clustering_result(rec_id: int, results, is_vp, folder: str = "output"):
    folder_path = f"{folder}/{is_vp}"
    os.makedirs(folder_path, exist_ok=True)

    save_path = f"{folder_path}/employee_{rec_id}_clustering_result.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Saved] {save_path}")
    return save_path

def save_final_result(results, is_vp, folder: str = "output"):
    folder_path = f"{folder}/{is_vp}"
    os.makedirs(folder_path, exist_ok=True)

    save_path = f"{folder_path}/pattern_results.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Saved] {save_path}")
    return save_path


def parse_json_from_llm(text):
    clean = (
        text.replace("```json", "")
            .replace("```", "")
            .strip()
    )
    return json.loads(clean)

def extract_phrase_set(signal_json):
    phrases = []

    for award in signal_json.values():
        for group in award.values():
            phrases.extend(group)

    return list(set(phrases)) 

def merge_signal_set(folder):
    if not os.path.exists(folder):
        print(f"[WARN] Folder not found: {folder}")
        return set()

    if not os.path.isdir(folder):
        print(f"[WARN] '{folder}' is not a directory.")
        return set()

    signal_set = set()

    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read JSON: {fpath} ({e})")
            continue
        pattern_list = list(data.keys())
        signal_set.update(pattern_list)

    return list(signal_set)
    



def load_provider_settings(provider: str):
    provider = provider.lower()

    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    if provider not in cfg:
        raise ValueError(f"No provider config for '{provider}'")

    block = cfg[provider]

    # Build env var: OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY
    env_var = f"{provider.upper()}_API_KEY"
    api_key = os.environ.get(env_var)

    if not api_key:
        raise EnvironmentError(f"Missing environment variable: {env_var}")

    return {
        "provider": provider,
        "model": block["model"],
        "temperature": block["temperature"],
        "max_tokens": block["max_tokens"],
        "api_key": api_key
    }