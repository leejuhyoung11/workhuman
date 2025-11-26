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

    def generate_difference_taxonomy(self, vp_path, non_vp_path):
        with open(vp_path, "r", encoding="utf-8") as f:
            vp = json.load(f)
        with open(non_vp_path, "r", encoding="utf-8") as f:
            non_vp = json.load(f)

        prompt = self._build_difference_prompt(vp, non_vp)
        raw = self.llm.call(prompt)
        # print(" response:", raw)

        try:
            parsed_json =  parse_json_from_llm(raw)
        except Exception as e:
            print("JSON PARSE ERROR:", e)
            parsed_json = {}

        


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


    def _build_difference_prompt(self, vp, non_vp):
        return f"""
        You are given two sets of behavioral cluster names:

        - Group A = treatment group clusters
        - Group B = control group clusters

        Do NOT assume anything about the groups.
        Do NOT infer seniority, performance, role level, or importance.
        Do NOT treat either group as superior or inferior.
        Both lists must be evaluated equally.

        Your task is to perform a purely semantic comparison and consolidation.

        =====================================================================
        STEP 1 — Semantic Understanding
        =====================================================================
        You must analyze all cluster names from Group A and Group B and determine:

        1. Which clusters express the SAME underlying behavioral concept.
        2. Which clusters are DISTINCT in meaning.
        3. Where two names differ in wording but are semantically overlapping.

        Important:
        - Use SEMANTIC interpretation only, not string matching.
        - Even if names are different, treat them as equivalent if the meaning aligns.
        - If two concepts partially overlap, MERGE them.
        - If uncertain, MERGE rather than split.

        =====================================================================
        STEP 2 — Build Canonical Competency Categories
        =====================================================================
        Using all clusters from BOTH groups:

        Create a unified, deduplicated set of canonical categories.

        For each canonical category:
        - Choose a clear, general, 2–5 word category name.
        - Do NOT generate abstract categories beyond what is implied in the data.
        - Do NOT split a concept into multiple categories.

        Each canonical category must contain:
        1. "aliases" → all original cluster names (from both groups) that semantically match
        2. "summary" → a 1–2 sentence explanation of the shared competency meaning

        =====================================================================
        STEP 3 — Identify semantic group relationships
        =====================================================================
        For each canonical category, classify it into:

        - "both_groups": appears semantically in BOTH treatment and control  
        - "treatment_only": appears ONLY in treatment (no semantic equivalent in control)  
        - "control_only": appears ONLY in control (no semantic equivalent in treatment)

        Important:
        - This determination must be SEMANTIC, not based on name overlap.
        - Two differently worded clusters count as "both_groups" if they express the same idea.

        =====================================================================
        STEP 4 — Output Format (JSON only)
        =====================================================================
        Return a single JSON object in this exact structure:

        {
        "<canonical_category>": {
            "aliases": [ ... all original cluster names ... ],
            "summary": "1–2 sentence meaning summary.",
            "group_presence": "both_groups" | "treatment_only" | "control_only"
        },
        ...
        }

        NO commentary.  
        NO markdown.  
        JSON only.

        =====================================================================
        NOW ANALYZE THE FOLLOWING:
        =====================================================================

        Group A (treatment):
        {vp}

        Group B (control):
        {non_vp}
        """

    
    