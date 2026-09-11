import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd # CSVの読み書きに追加
import os

# --- 1. セットアップ（変更なし） ---

# モデルElyza8b
model_id = "elyza/Llama-3-ELYZA-JP-8B"

# デバイスの確認
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_id)
# モデルのロード
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

# プロンプトの準備（変更なし）
instructs = """あなたは、提供されたプログラムコードの構造と目的を深く理解し、その**重要な部分**を抜き出す専門家です。以下の###指示に従い、###入力されたコードの内容から、特に重要な役割を果たしている部分を特定し、その要点を簡潔な日本語で説明してください。

###指示：
1. **コードの言語は、常に日本語プログラミングとしてください。**（言語判定は不要です）
2. **コードの全体的な目的**を理解し、その目的を達成するために**中核となるロジック**や**重要な処理**を特定してください。
3. 特定した重要な部分を**コードブロック形式**で抜き出し、その**機能や役割**を簡潔に説明してください。
4. **重要な計算式**、**条件分岐**、**ループ**、**入出力処理**など、プログラムの挙動を決定づける部分に焦点を当ててください。
5. **計算、条件分岐、ループ、入出力**など、**異なる役割を持つロジックは、それぞれ別の「重要な部分」として**抜き出してください。
6. 抽出する要点は**最大で3つ**に絞り込んでください。
7. 抽出した要点は、プログラムの目的達成に対する**重要度が最も高いものから順に**並べてください。
8. それぞれの要点について**2〜3行で簡潔に**説明してください。
9. 出力は以下の例に従ってください。

###言語：
日本語プログラミング

###コードの目的：
[コードの全体的な目的]

###重要な部分：
以下は、特に重要な部分を抜き出したコードブロックとその説明です。（★**重要度が高い順に1から番号が振られています**）

###1. [最も重要な部分のタイトル]
[コードブロック]

[説明文]

###2. [次に重要な部分のタイトル]
[コードブロック]

[説明文]

###3. [3番目に重要な部分のタイトル]
[コードブロック]

[説明文]
"""

# --- 2. 既存CSVの読み込みとループ処理（★変更） ---

input_csv = "/home/s22t324/project/rdr_data.csv" 

try:
    df = pd.read_csv(input_csv)
    # ★変更：既存の analysis_result 列があっても削除せず、メッセージだけ表示
    if 'analysis_result' in df.columns:
        print(f"'{input_csv}' には 'analysis_result' 列が既に存在します。")

except FileNotFoundError:
    print(f"エラー: {input_csv} が見つかりません。")
    print("先に問題文生成スクリプトを実行して、result.csvを作成してください。")
    exit()

# 結果を保存するための新しい列（リスト）を準備
analysis_results = []

print(f"{input_csv} から {len(df)} 件のコードを読み込みました。分析を開始します...")

# CSVの各行をループ処理（変更なし）
for index, row in df.iterrows():
    # ★CSVの 'original_code' 列を入力として使用
    intext = row['original_code'] 
    
    # 'filename' 列があるか確認し、ログ出力に使用
    filename = row.get('filename', f"Row {index}")
    print(f"--- Analyzing: {filename} ({index + 1}/{len(df)}) ---")

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
            return_attention_mask=True,
            truncation=True, # 長いコードに対応
            max_length=4096  # モデルのコンテキスト上限に応じて調整
        )

        token_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        # 推論の実行
        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                do_sample=True,
                temperature=0.5,
                top_p=0.7,
                max_new_tokens=2048,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ],
                pad_token_id=tokenizer.eos_token_id
            )
        
        output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
        
        # 結果をリストに追加
        analysis_results.append(output)

    except Exception as e:
        print(f"エラーが発生しました ({filename}): {e}")
        analysis_results.append(f"ERROR: {e}") # エラーが発生してもリストの長さを合わせる

# --- 3. 新しい列をDataFrameに追加してCSVに上書き保存（★変更） ---

if len(analysis_results) == len(df):
    
    # ★変更：新しい列名を動的に決定する
    new_column_name = 'analysis_result'
    counter = 1
    
    # df.columns (既存の列名リスト) に new_column_name が存在する限り、
    # 末尾に数字を付けた名前 (analysis_result_1, analysis_result_2, ...) を試行する
    while new_column_name in df.columns:
        new_column_name = f'analysis_result_{counter}'
        counter += 1
        
    print(f"\n新しい分析結果を '{new_column_name}' 列として追加します。")
    
    # 決定した列名でDataFrameに新しい列を追加
    df[new_column_name] = analysis_results
    
    # ★ result.csv に上書き保存
    df.to_csv(input_csv, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 全てのコードの分析が完了しました。")
    print(f"'{input_csv}' に '{new_column_name}' 列を追加して上書き保存しました。")

else:
    print(f"\n⚠️ 処理が失敗しました。CSVの行数と結果の数が一致しません。")
    print(f"(CSV行数: {len(df)}, 生成結果数: {len(analysis_results)})")