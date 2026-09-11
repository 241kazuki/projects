import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import pandas as pd

# --- 1. セットアップ ---
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

# プロンプトの準備
instructs = """
あなたは、経験豊富なプログラミング教材の作成者である。
あなたの目的は、入力された「模範解答」のコードを分析し、学習者がそのコードと全く同じ動作をするプログラムをゼロから作成できるように、「詳細な仕様がすべて含まれた、自然で論理的な問題文（お題）」を生成することである。

入力されるコードは「日本語プログラミング言語（なでしこ）」である。

### 指示：
1.  入力されたコードの動作（入力、計算、制御構造、出力など）を完全に再現するための「仕様」を漏れなく抽出すること。
2.  抽出した仕様は、箇条書きにはせず、一つの連続した自然な文章として統合すること。
3.  学習者がアルゴリズムの論理構造（処理の手順）を明確にイメージできるよう、処理の順番に沿って記述すること。
4.  学習者に対する「～するプログラムを作成してください」という明確な作成指示にすること。
5.  重要な制約1：元のコード内の具体的な「変数名」や「配列」「辞書」といった特定のデータ構造の専門用語は極力避け、「データ」「一覧」「対応表」など、初学者が直感的に理解しやすい一般的な表現に置き換えること。
6.  重要な制約2：Pythonなど他言語の構文（while, for, importなど）を連想させる表現は避け、「条件を満たす間繰り返す」「〇回繰り返す」といった日本語として自然な表現を用いること。
7.  生成した問題文の難易度を（初級）または（中級）で示すこと。

### 入力コード（模範解答）：
[ここにコードが挿入される]

### 出力形式：
（「問題文：」などの見出しは不要である。以下の形式のみを出力すること）
（[初級/中級]） [抽出したすべての仕様が自然な文章として統合された作成指示]
"""

# --- 2. CSVファイルの読み込みと処理のループ化 ---

# 入力CSVファイルのパス
input_csv = "/home/s22t324/project/transposed_nako3_sample.csv"

# CSVを読み込む
try:
    df_input = pd.read_csv(input_csv, header=None, names=['filename', 'code'])
except Exception as e:
    print(f"CSVの読み込み中にエラーが発生しました: {e}")
    exit()

# 全ての結果を保存するためのリスト
results_list = []

print(f"Found {len(df_input)} codes in CSV. Starting processing...")

# 各行に対して処理を繰り返す
for index, row in df_input.iterrows():
    filename = row['filename']
    intext = row['code']
    
    print(f"--- Processing ({index+1}/{len(df_input)}): {filename} ---")
    
    try:
        # チャットテンプレート
        chats = [
            { "role": "system", "content": instructs},
            { "role": "user", "content": intext},
        ]

        # テンプレート適用とエンコード
        formatted_prompt = tokenizer.apply_chat_template(
            chats,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded_input = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            return_attention_mask=True
        )
        token_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        # 推論の実行
        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                max_new_tokens=2048,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ],
                pad_token_id=tokenizer.eos_token_id
            )
        
        # デコードして結果を取得
        output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
        
        # 結果をリストに追加
        results_list.append({
            'filename': filename,
            'original_code': intext,
            'generated_problem': output
        })

    except Exception as e:
        print(f"An error occurred while processing {filename}: {e}")

# --- 3. 全ての結果をCSVファイルに保存 ---

if results_list:
    df_result = pd.DataFrame(results_list)
    df_result.to_csv("result.csv", index=False, encoding='utf-8-sig')
    print("\n✅ All data processed successfully. Results saved to result.csv")
else:
    print("\n⚠️ No data was processed. result.csv was not created.")