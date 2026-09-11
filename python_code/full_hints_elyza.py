import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd # CSVの読み書きに追加
import os

# --- 1. セットアップ（モデルロードなど） ---

# モデルElyza8b
model_id = "elyza/Llama-3-ELYZA-JP-8B"

# デバイスの確認
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# トークナイザーとモデルの準備
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

# --- 2. 既存CSVの読み込みとループ処理（★ここからが主な変更点） ---

input_csv = "result.csv" 

try:
    df = pd.read_csv(input_csv)
    # 既存の 'blanked_code' 列があれば削除し、作り直す
    if 'blanked_code' in df.columns:
        df = df.drop(columns=['blanked_code'])
        print(f"既存の 'blanked_code' 列を削除し、再生成します。")

except FileNotFoundError:
    print(f"エラー: {input_csv} が見つかりません。")
    print("先に他のスクリプトを実行して、'generated_problem' と 'analysis_result' が含まれた result.csv を作成してください。")
    exit()

# 結果を保存するための新しい列（リスト）を準備
blanked_code_results = []

print(f"{input_csv} から {len(df)} 件のデータを読み込みました。空欄補充問題の生成を開始します...")

# CSVの各行をループ処理
for index, row in df.iterrows():
    # ★CSVから3つの必須データを取得
    problem_statement = row['generated_problem']
    intext_code = row['original_code']
    analysis_result = row['analysis_result']
    
    filename = row.get('filename', f"Row {index}")
    print(f"--- Generating blanks for: {filename} ({index + 1}/{len(df)}) ---")

    try:
        # --- 4. プロンプトをループ内で動的に生成 ---
        instructs_code_fill = f"""
あなたは、提供されたプログラムコード（###入力コードブロック）に対し、**「###達成すべき問題」**と**「###事前のコード分析」**の内容に厳密に従い、元のコードの構造を維持しながら空欄化する専門家です。

###指示：
1. 「###達成すべき問題」と「###事前のコード分析」の内容を徹底的に比較・参照します。
2. 「###事前のコード分析」で示されている**すべての重要な部分（例： `###1.`、`###2.`、`###3.`）の各ブロックから、それぞれ最低1つ以上**、学習上重要だと判断した箇所を空欄（`___`）にしてください。
3. 空欄にする対象は、問題文の仕様に対応する「数値」（例: `100`, `25`, `18.5`）、「演算子」（例: `÷`, `^`, `以上`）、「制御キーワード」（例: `もし`）、「文字列」（例: `「肥満」`, `「BMI=...」`）などです。
4. 空欄の数は合計で5箇所から最大10箇所程度を目標とします。
5. コードの構造は完全に維持してください。
6. 出力は、空欄化されたコードブロックのみとしてください。

###達成すべき問題：
{problem_statement}

###事前のコード分析：
{analysis_result}

###入力コードブロック：
{intext_code}

###出力（空欄化されたコードブロック）：
"""

        # チャットテンプレート
        chats_code_fill = [
            { "role": "system", "content": instructs_code_fill},
        ]

        # テンプレート適用とエンコード
        formatted_prompt_code_fill = tokenizer.apply_chat_template(
            chats_code_fill,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded_input_code_fill = tokenizer(
            formatted_prompt_code_fill,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True, # 非常に長いプロンプトに対応
            max_length=8192  # モデルの最大長（Llama3は8k）
        )

        token_ids_code_fill = encoded_input_code_fill["input_ids"]
        attention_mask_code_fill = encoded_input_code_fill["attention_mask"]

        # 推論の実行
        with torch.no_grad():
            output_ids_code_fill = model.generate(
                token_ids_code_fill.to(model.device),
                attention_mask=attention_mask_code_fill.to(model.device),
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                max_new_tokens=1024, # 空欄コードの出力なので512でも十分かもしれません
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ],
                pad_token_id=tokenizer.eos_token_id
            )
        
        # デコード
        output_code_fill = tokenizer.decode(output_ids_code_fill.tolist()[0][token_ids_code_fill.size(1) :], skip_special_tokens=True)
        
        # 結果をリストに追加
        blanked_code_results.append(output_code_fill)

    except Exception as e:
        print(f"エラーが発生しました ({filename}): {e}")
        blanked_code_results.append(f"ERROR: {e}") # エラーが発生してもリストの長さを合わせる

# --- 3. 新しい列をDataFrameに追加してCSVに上書き保存 ---

if len(blanked_code_results) == len(df):
    df['blanked_code'] = blanked_code_results
    
    # ★ result.csv に上書き保存
    df.to_csv(input_csv, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 全ての空欄補充問題の生成が完了しました。")
    print(f"'{input_csv}' に 'blanked_code' 列を追加して上書き保存しました。")
else:
    print(f"\n⚠️ 処理が失敗しました。CSVの行数と結果の数が一致しません。")
    print(f"(CSV行数: {len(df)}, 生成結果数: {len(blanked_code_results)})")