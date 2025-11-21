import tiktoken
import os, json

enc = tiktoken.get_encoding("cl100k_base")

def chunk_awards(rec_id, awards_list, max_tokens=1500):
    """
    awards_list: [{'title': ..., 'message': ...}, ...]
    returns: list of text chunks
    """
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


def save_employee_keywords(rec_id: int, results: dict, folder: str = "output"):
    os.makedirs(folder, exist_ok=True)

    save_path = f"{folder}/employee_{rec_id}_keywords.json"

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