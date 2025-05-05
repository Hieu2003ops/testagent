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

# === 4. Load logs (5 samples của bạn) ===
with open("log.json", "r", encoding="utf-8") as f:
    logs = json.load(f)

def call_llm(prompt: str) -> dict:
    """Gọi GPT và parse JSON response, tự động strip code fences."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user", "content": prompt}],
        temperature=0
    )
    content = resp.choices[0].message.content

    # --- XỬ LÝ code fences ---
    # Nếu response bắt đầu bằng ``` thì loại bỏ fence đầu và cuối
    if content.strip().startswith("```"):
        lines = content.splitlines()
        # bỏ dòng fence đầu (``` hoặc ```json)
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # bỏ fence cuối
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines)

    # Bây giờ thử parse JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Nếu vẫn lỗi, trả về raw để debug tiếp
        return {"error": content}

records = []

for i, log in enumerate(tqdm(logs, desc="Đánh giá logs")):
    # 5. Chuẩn bị prompt
    tool1 = " → ".join(log.get("tool_chain_first", []))
    tool2 = " → ".join(log.get("tool_loop_second", []))
    answer = "\n".join(log.get("final_answer", []))
    prompt = template.substitute(
        query=log["query"],
        tool1=tool1,
        tool2=tool2,
        answer=answer
    )

    # Debug 1 sample đầu
    if i == 0:
        print("=== DEBUG PROMPT ===")
        print(prompt)
        print("====================")

    # 6. Gọi LLM
    result = call_llm(prompt)

    # Debug raw result
    if i == 0:
        print("=== DEBUG RESULT ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("====================")

    # 7. Đưa kết quả vào record
    record = {
        "query": log["query"],
        "tool_chain_first": tool1,
        "tool_loop_second": tool2
    }
    total_weighted = 0.0
    total_weights  = 0.0

    for code, cfg in criteria_cfg.items():
        entry = result.get(code, {})
        raw_score = entry.get("score")
        note = entry.get("note", "")

        # Ép score về float nếu cần
        score = None
        if isinstance(raw_score, (int, float)):
            score = float(raw_score)
        elif isinstance(raw_score, str):
            try:
                score = float(raw_score)
            except:
                score = None

        record[f"{code}_score"] = score if score is not None else ""
        record[f"{code}_note"]  = note

        if score is not None:
            w = cfg["weight"]
            total_weighted += score * w
            total_weights  += w

    # 8. Lấy Overall từ result hoặc fallback
    overall = None
    if "Overall_score" in result:
        overall = result["Overall_score"]
    elif "Overall" in result:
        ov = result["Overall"]
        try:
            overall = float(ov)
        except:
            overall = None
    elif total_weights > 0:
        overall = total_weighted / total_weights

    record["Overall_score"] = overall
    records.append(record)

# 9. Xuất ra CSV & Excel
df = pd.DataFrame(records)
df.to_excel("evaluation_results_full.xlsx", index=False)

print("✅ Hoàn tất đánh giá. Kết quả ở evaluation_results.csv và evaluation_results.xlsx")
