import os
import json
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from string import Template
from openai import OpenAI

# === 1. Load API key và khởi tạo client ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# === 2. Load criteria config ===
with open("criteria_config.json", "r", encoding="utf-8") as f:
    criteria_cfg = json.load(f)

# === 3. Load prompt template ===
with open("prompt_template.txt", "r", encoding="utf-8", errors="ignore") as f:
    raw = f.read()
template = Template(raw)

# === 4. Load full logs.json ===
with open("log.json", "r", encoding="utf-8") as f:
    logs = json.load(f)

def call_llm(prompt: str) -> dict:
    """Gọi GPT và parse JSON response, strip code fences nếu có."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = resp.choices[0].message.content

    # Loại bỏ code fences
    if content.strip().startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": content}

def map_score(raw_score):
    """
    Map raw_score in {1.0, 0.5, 0.0, None} to:
      - None (N/A) -> 0.5
      - 1.0 -> 1.0
      - 0.5 -> 0.8
      - 0.0 -> 0.8
    """
    if raw_score is None:
        return 0.5
    # ensure float
    try:
        s = float(raw_score)
    except:
        return 0.5
    if s >= 1.0:
        return 1.0
    # both PARTIAL (0.5) and NOT OK (0.0) map to 0.8
    return 0.8

records = []

# === 5. Đánh giá từng log ===
for log in tqdm(logs, desc="Đánh giá logs"):
    tool1 = " → ".join(log.get("tool_chain_first", []))
    tool2 = " → ".join(log.get("tool_loop_second", []))
    answer = "\n".join(log.get("final_answer", []))

    # Build prompt
    prompt = template.substitute(
        query=log["query"],
        tool1=tool1,
        tool2=tool2,
        answer=answer
    )

    result = call_llm(prompt)

    record = {
        "query": log["query"],
        "tool_chain_first": tool1,
        "tool_loop_second": tool2
    }
    total_weighted = 0.0
    total_weights  = 0.0

    # Parse each criterion
    for code, cfg in criteria_cfg.items():
        entry = result.get(code, {})
        raw = entry.get("score")
        note = entry.get("note", "")

        # Map sang scale mới
        mapped = map_score(raw)

        record[f"{code}_score"] = mapped
        record[f"{code}_note"]  = note

        # Cộng vào overall weighted
        w = cfg["weight"]
        total_weighted += mapped * w
        total_weights  += w

    # Compute Overall_score
    overall = total_weighted / total_weights if total_weights > 0 else None
    record["Overall_score"] = overall

    records.append(record)

# === 6. Xuất ra CSV & Excel ===
df = pd.DataFrame(records)
# df.to_csv("evaluation_results.csv", index=False, encoding="utf-8-sig")
df.to_excel("evaluation_results.xlsx", index=False)

print("✅ Hoàn tất đánh giá. Kết quả ở evaluation_results.csv và evaluation_results.xlsx")
