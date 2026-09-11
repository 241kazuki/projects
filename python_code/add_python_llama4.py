import os
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging

logging.set_verbosity_error()

MODEL_ID = "llm-jp/llm-jp-4-8b-thinking"
INPUT_CSV = "/home/s22t324/project/result.csv"
OUTPUT_CSV = "add_python_result_llama4.csv"

print(f"=== モデル {MODEL_ID} をロード中... ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

instructs = """あなたはコード変換器です。

入力として以下が与えられます。

1. 問題文
2. なでしこコード

あなたの仕事は、
問題文とコードの両方を参考にしながら、
入力されたなでしこコードと
完全に同じ動作をする
Python 3 のコードを生成することです。

## 厳守事項

- Pythonコードのみを出力してください
- 説明は禁止
- コメントは禁止
- markdown記法は禁止
- ``` や ```python を出力しない
- 標準入力・標準出力を使用する
- 元コードと完全に同じ動作にする
- 問題文の仕様を必ず反映する
- 不明な場合でも可能な限り推測して実装する

## 出力
Pythonコードのみ"""

try:
    df_input = pd.read_csv(INPUT_CSV, dtype=str)
except Exception as e:
    print(f"CSV読み込みエラー: {e}")
    exit()

results_list = []

print("\n=== Pythonコード生成処理の開始 ===")
for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Python生成"):
    filename = row.get("filename", "")
    problem = row["generated_problem"]
    original_code = row["original_code"]

    try:
        user_content = f"# 問題文\n{problem}\n\n# なでしこコード\n{original_code}"

        chats = [
            {"role": "system", "content": instructs},
            {"role": "user", "content": user_content},
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            chats,
            tokenize=False,
            add_generation_prompt=True,
        )

        encoded_input = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            return_attention_mask=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **encoded_input,
                do_sample=False,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        output = tokenizer.decode(
            output_ids[0][encoded_input["input_ids"].size(1):],
            skip_special_tokens=True
        )

        # 思考プロセスタグやマークダウンブロックの削除
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
        output = output.replace("```python", "").replace("```", "").strip()

        if output == "":
            continue

        results_list.append({
            "filename": filename,
            "generated_problem": problem,
            "original_code": original_code,
            "python_code": output
        })

    except Exception as e:
        print(f"Error generating for row: {e}")
        continue
    finally:
        if 'encoded_input' in locals():
            del encoded_input
        if 'output_ids' in locals():
            del output_ids
        torch.cuda.empty_cache()

if results_list:
    df_result = pd.DataFrame(results_list)
    df_result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n処理が完了しました。結果を {OUTPUT_CSV} に保存しました。")
else:
    print("生成されたデータがありませんでした。")