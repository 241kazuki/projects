import torch
import pandas as pd
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
from peft import PeftModel

logging.set_verbosity_error()

# --- 1. セットアップ ---
BASE_MODEL_ID = "Qwen/Qwen3-8B"
PEFT_MODEL_ID = "lorenzocazzador/coder-grpo-qwen3-8b"

input_csv = "/mnt/data/s26g375/project/result.csv"
output_csv = "add_python_result_qwen3.csv"

print("=== モデルおよびPEFTアダプターのロード中... ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained(PEFT_MODEL_ID, trust_remote_code=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)

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

def apply_chat_template_safe(messages, tokenizer, add_generation_prompt=True):
    """Qwen3の思考モードを明示的にOFFにする。テンプレートが未対応の場合はフォールバック。"""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

def strip_thinking(text):
    """<think>ブロックを除去。閉じタグがない(=生成が思考の途中で打ち切られた)場合は
    有効な出力が得られていないとみなし空文字を返す。"""
    if "<think>" in text and "</think>" not in text:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

try:
    df_input = pd.read_csv(input_csv, dtype=str)
except Exception as e:
    print(f"CSV読み込みエラー: {e}")
    exit()

results_list = []
incomplete_count = 0

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

        formatted_prompt = apply_chat_template_safe(chats, tokenizer)

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
                pad_token_id=tokenizer.eos_token_id
            )

        output = tokenizer.decode(
            output_ids[0][encoded_input["input_ids"].size(1):],
            skip_special_tokens=True
        )

        # 思考ブロックの除去（未完了なら空文字になる）
        output = strip_thinking(output)
        # マークダウンの除去処理
        output = output.replace("```python", "").replace("```", "").strip()

        if output == "":
            incomplete_count += 1
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

if incomplete_count:
    print(f"注意: {incomplete_count} 件で思考が完了せず出力をスキップしました（max_new_tokens不足の可能性）。")

if results_list:
    df_result = pd.DataFrame(results_list)
    df_result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n処理が完了しました。結果を {output_csv} に保存しました。")
else:
    print("生成されたデータがありませんでした。")