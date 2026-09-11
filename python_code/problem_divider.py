import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os

# --- 1. セットアップ（モデルとデバイスの準備） ---

# モデルID
model_id = "elyza/Llama-3-ELYZA-JP-8B"

# デバイスの確認 (GPU > MPS > CPU)
device = torch.device('cuda') if torch.cuda.is_available() else \
         torch.device('mps') if torch.backends.mps.is_available() else \
         torch.device('cpu')
print(f"Using device: {device}")

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_id)

# モデルのロード
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

print(f"Model '{model_id}' loaded successfully.")

# --- 2. ヒント生成のための「入力情報」を定義 ---

# LLMに与える「問題文」
problem_statement = """
問題文：身長と体重を入力してください。
身長はセンチメートル、体重はキログラムで入力してください。
入力された身長と体重を用いて、BMIを計算し、肥満、低体重、普通体重のいずれに該当するかを判定し、結果を表示してください。
"""

# LLMに「お手本」として与える完成形コード
target_code = """
# 身長と体重を尋ねる
「身長(cm)は？」と尋ねて身長cmに代入。
「体重(kg)は？」と尋ねて体重kgに代入。
# BMIを計算
BMI=体重kg÷(身長cm/100)^2
# 肥満かどうか判定
もし、BMIが25以上ならば
　　判定＝「肥満」
違えば、もし、BMIが18.5未満ならば
　　判定＝「低体重」
違えば
　　判定＝「普通体重」
ここまで。
# 結果を表示
「BMI={BMI} / 判定={判定}」を表示。
"""

# ★修正点：
# プロンプト内のアルゴリズム（1.入力, 2.計算, 3.判定, 4.表示）と
# 参照する「コード分析結果」の番号と内容が一致するように修正。
analysis_result = """
###言語：
日本語プログラミング

###コードの目的：
ユーザーに身長と体重を尋ね、BMIを計算し、肥満かどうかを判定して結果を表示するプログラムです。

###重要な部分（分割単位）：

###1. 入力と代入
「身長(cm)は？」と尋ねて身長cmに代入。
「体重(kg)は？」と尋ねて体重kgに代入。
この部分は、ユーザーに身長と体重を尋ねて、変数に代入する部分です。

###2. BMIの計算
BMI=体重kg÷(身長cm/100)^2
この部分は、BMIを計算するロジックです。

###3. 肥満かどうかの判定
もし、BMIが25以上ならば
    判定＝「肥満」
違えば、もし、BMIが18.5未満ならば
    判定＝「低体重」
違えば
    判定＝「普通体重」
ここまで。
この部分は、BMIに応じた判定の条件分岐が含まれています。

###4. 結果の表示
「BMI={BMI} / 判定={判定}」を表示。
この部分は、計算されたBMIと判定結果を表示する部分です。
"""
target_language = "日本語プログラミング（なでしこ、プロデルなどを想定）"


# --- 3. LLMへの詳細な「指示書（プロンプト）」を作成 ---
# ★修正点：
# 参照する analysis_result の番号と内容が一致するように、
# 思考プロセス（アルゴリズム）の指示を修正。
instructs_step_by_step = f"""
あなたは、プログラミング教育の専門家です。
「### 解答例コード」と「### コード分析結果」に基づき、コードを段階的なステップに分割します。

### あなたが実行すべき思考プロセス（アルゴリズム）：

1.  **ステップ 1（入力）:** 「### コード分析結果」の「**1. 入力と代入**」のコード断片を特定し、ステップ 1として出力します。
2.  **ステップ 2（計算）:** 「### コード分析結果」の「**2. BMIの計算**」のコード断片を特定します。**ステップ 1のコード全文**に、その計算コードを追加して、ステップ 2として出力します。
3.  **ステップ 3（判定）:** 「### コード分析結果」の「**3. 肥満かどうかの判定**」のコード断片を特定します。**ステップ 2のコード全文**に、その判定コードを追加して、ステップ 3として出力します。
4.  **ステップ 4（表示）:** 「### コード分析結果」の「**4. 結果の表示**」のコード断片を特定します。**ステップ 3のコード全文**に、その表示コードを追加して、最終ステップとして出力します。

### 厳守すべきルール：
* **言語の厳守**：絶対にPythonコードを生成してはいけません。「### 解答例コード」と「### コード分析結果」で使われている日本語プログラミングの構文を**そのまま使用**してください。
* **全文コピーの徹底**：ステップ 2以降は、必ず**直前のステップのコード全文**を含めてください。
* **完了の徹底**：分析結果のステップがすべて完了するまで、生成を中断しないでください。

### 入力情報：
**対象言語：** {target_language}
**問題文：** {problem_statement}
**解答例コード：**
{target_code}

**コード分析結果（これが各ステップの単位）：**
{analysis_result}
---

### 出力テンプレート：
（あなたは、今から以下の形式で「ステップ 1」から生成を開始します）

---
### ステップ [番号]: [このステップの目的]

[実行可能なコードブロック全体]


**解説：**
[このステップのコードに関する簡単な説明]

**(ステップ 2以降のみ) ★今回の追加/変更点：**
[直前のステップから追加・変更されたコード断片]

"""

# --- 4. LLMによる推論（段階的ヒント生成）の実行 ---

# チャットテンプレートの準備
chats = [
    { "role": "system", "content": instructs_step_by_step},
]

# プロンプトをチャット形式にフォーマット
formatted_prompt = tokenizer.apply_chat_template(
    chats,
    tokenize=False,
    add_generation_prompt=True,
)

# フォーマットされたプロンプトをエンコード（トークン化）
encoded_input = tokenizer(
    formatted_prompt,
    return_tensors="pt",
    return_attention_mask=True
)

token_ids = encoded_input["input_ids"]
attention_mask = encoded_input["attention_mask"]

print("\nStarting model inference to generate stepwise hints (v5, do_sample=False)...")

# 推論の実行
with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        # ランダム性を排除し、プロンプトのアルゴリズムに厳密に従わせる
        do_sample=False, 
        max_new_tokens=4096, 
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ],
        pad_token_id=tokenizer.eos_token_id
    )

# --- 5. 結果のデコードと出力 ---

# output_idsから入力プロンプト部分を除去し、デコード
output_text = tokenizer.decode(
    output_ids.tolist()[0][token_ids.size(1) :], 
    skip_special_tokens=True
)

# 生成された段階的ヒントを出力
print("\n--- Generated Stepwise Hints ---")
print(output_text)
print("--- End of Output ---")

print("\nScript finished.")