from typing import List, Dict, Any
import json
from concurrent.futures import ThreadPoolExecutor

from src.models.provider_factory import LLMProviderFactory

from utils.utils import parse_json_from_llm, save_final_result, save_taxonomy

class GlobalCluster:

    def __init__(self, provider, model, temperature, api_key):
        self.llm = LLMProviderFactory.create(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=4000
        )
    
    def dedupligate_signals(self, signal_list, is_vp):
        print(f"Deduplicating process...")
        prompt = self._build_deduplicate_prompt(signal_list)
        raw = self.llm.call(prompt)
        # print(" response:", raw)

        try:
            parsed_json =  parse_json_from_llm(raw)
        except Exception as e:
            print("JSON PARSE ERROR:", e)
            parsed_json = {}

        save_final_result(parsed_json, is_vp)

        return 


    def generate_canonical_taxonomy(self, vp_path, non_vp_path):
        with open(vp_path, "r", encoding="utf-8") as f:
            vp = json.load(f)
        with open(non_vp_path, "r", encoding="utf-8") as f:
            non_vp = json.load(f)

        seen = set()
        taxonomy = {}

        for name, content in vp.items():
            if name in seen:
                continue
            seen.add(name)
            taxonomy[name] = content.get("summary", "")

        for name, content in non_vp.items():
            if name in seen:
                continue
            seen.add(name)
            taxonomy[name] = content.get("summary", "")

        save_taxonomy(taxonomy)
        
        return taxonomy


    def _build_deduplicate_prompt(self, signal_list):
        return f"""
        You will perform **SEMANTIC DEDUPLICATION** on a list of behavior cluster names.

        IMPORTANT:
        - These names already represent meaningful behavioral themes.
        - Your sole objective is **semantic consolidation**, NOT reinterpreting employees.
        - You MUST reduce the total number of themes as much as reasonably possible.

        ===============================================================
        GOAL — MINIMIZE the Number of Canonical Themes
        ===============================================================
        If two names show ANY meaningful conceptual overlap, even partial overlap:
        - You MUST merge them.
        - When uncertain, ALWAYS merge rather than separate.
        - Do NOT leave similar clusters as separate.

        Treat the following as equivalent:
        - wording differences (“leadership” vs “leading”)
        - scope variations (“program” vs “initiative”)
        - level of detail differences (“planning & vision” vs “strategic planning”)
        - domain focus variations (“team”, “cross-team”, “cross-functional”)

        ===============================================================
        STEP 1 — Deduplication Rules (STRONG)
        ===============================================================
        You MUST merge names together if they share ANY of:
        - similar strategic intent
        - overlapping leadership concepts
        - related planning or influence ideas
        - related execution themes
        - similar communication concepts
        - similar customer or stakeholder concepts

        Do NOT:
        - generate new themes
        - broaden meaning beyond what aliases share
        - split a theme into sub-themes
        - keep names separate unless their meanings are CLEARLY different

        ===============================================================
        STEP 2 — Canonical Name Selection
        ===============================================================
        For each merged group:
        - Select ONE canonical name.
        - Choose the MOST GENERAL name that still captures all aliases.
        - Avoid overly specific words.
        - Avoid obscure or narrow names.

        ===============================================================
        STEP 3 — Canonical Summary
        ===============================================================
        Write a neutral 1–2 sentence summary capturing ONLY the meaning shared by all aliases.

        ===============================================================
        STEP 4 — Output Format (JSON ONLY)
        ===============================================================
        {{
        "<canonical_name>": {{
            "aliases": ["..."],
            "summary": "..."
        }}
        }}

        Return ONLY a JSON object.
        DO NOT include any text before or after the JSON.
        DO NOT include summaries, bullet points, or explanations.
        If you output anything outside JSON, the system will fail.
        ===============================================================
        NOW DEDUPLICATE THE FOLLOWING NAMES:
        ===============================================================
        {signal_list}
            """

    
    