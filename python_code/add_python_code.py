import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from transformers.utils import logging

# transformers のログ非表示
logging.set_verbosity_error()

# --- 1. セットアップ ---
model_id = "elyza/Llama-3-ELYZA-JP-8B"

# デバイス確認
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# --- 2. トークナイザーとモデル読み込み ---
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)

# --- 3. AIへの指示 ---
instructs = """
あなたはコード変換器です。

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
Pythonコードのみ
"""

# --- 4. CSV読み込み ---
input_csv = "/home/s22t324/project/result.csv"

try:

    # CSVに以下3列がある前提
    # filename, generated_problem, original_code

    df_input = pd.read_csv(
        input_csv,
        dtype=str
    )

except Exception as e:
    print(f"CSV読み込みエラー: {e}")
    exit()

# --- 5. 結果保存用 ---
results_list = []

# --- 6. 各コードを処理 ---
for _, row in df_input.iterrows():

    filename = row["filename"]
    problem = row["generated_problem"]
    original_code = row["original_code"]

    try:

        # ユーザー入力作成
        user_content = f"""
# 問題文
{problem}

# なでしこコード
{original_code}
"""

        # チャット形式
        chats = [
            {
                "role": "system",
                "content": instructs
            },
            {
                "role": "user",
                "content": user_content
            },
        ]

        # プロンプト生成
        formatted_prompt = tokenizer.apply_chat_template(
            chats,
            tokenize=False,
            add_generation_prompt=True,
        )

        # トークナイズ
        encoded_input = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            return_attention_mask=True
        )

        token_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        # 推論
        with torch.no_grad():

            output_ids = model.generate(
                token_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),

                do_sample=False,

                max_new_tokens=512,

                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ],

                pad_token_id=tokenizer.eos_token_id
            )

        # デコード
        output = tokenizer.decode(
            output_ids.tolist()[0][token_ids.size(1):],
            skip_special_tokens=True
        )

        # markdown除去
        output = output.replace("```python", "")
        output = output.replace("```", "")
        output = output.strip()

        # 空出力除外
        if output == "":
            continue

        # 保存
        results_list.append({
            "filename": filename,
            "generated_problem": problem,
            "original_code": original_code,
            "python_code": output
        })

    except Exception:
        continue

# --- 7. CSV保存 ---
if results_list:

    df_result = pd.DataFrame(results_list)

    output_csv = "add_python_result.csv"

    df_result.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

else:
    print("No data processed.")