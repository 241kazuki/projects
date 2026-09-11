import os
import pandas as pd
import requests
import re

# --- 1. セットアップ ---
# OllamaのAPIエンドポイントとモデル名を指定
OLLAMA_API_URL = "http://127.0.0.1:1146de0/api/chat"
OLLAMA_MODEL = "deepseek-coder-v2" # 検証に用いるDeepSeekモデル名に変更すること

print(f"Using Ollama API at {OLLAMA_API_URL} with model {OLLAMA_MODEL}")

# プロンプトの準備（要件欠落を防ぐ改善版）
instructs = """
あなたは、経験豊富なプログラミング教材の作成者である。
あなたの目的は、入力された「模範解答」のコードを分析し、学習者がそのコードと全く同じ動作をするプログラムをゼロから作成できるように、「詳細な仕様がすべて含まれた、自然で論理的な問題文（お題）」を生成することである。

入力されるコードは「日本語プログラミング言語（なでしこ）」である。

### 指示：
1. 入力されたコードの動作（入力、計算、制御構造、出力など）を完全に再現するための「仕様」を漏れなく抽出すること。
2. 抽出した仕様は、箇条書きにはせず、一つの連続した自然な文章として統合すること。
3. 学習者がアルゴリズムの論理構造（処理の手順）を明確にイメージできるよう、処理の順番に沿って記述すること。
4. 学習者に対する「～するプログラムを作成してください」という明確な作成指示にすること。
5. 【重要】「配列」や「辞書」といった他言語特有のデータ構造の専門用語は極力避けるが、元のコード内で使用されている「具体的な数値（金額、回数、しきい値など）」「計算式」「データの項目名（エサ代、体力など）」は、抽象化せずに必ず問題文に明記すること。
6. 【重要】出力に関する指示では、プログラムが表示すべき文字列の完全なフォーマット（例：「毎年〇〇円必要です。」「義なる者はたとえ七度倒れても必ず立ち上がる」など）を一言一句違わず指定すること。
7. Pythonなど他言語の構文（while, for, importなど）を連想させる表現は避け、「条件を満たす間繰り返す」「〇回繰り返す」といった日本語として自然な表現を用いること。
8. 生成した問題文の難易度を（初級）または（中級）で示すこと。

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
        # Ollama APIへのペイロード
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": instructs},
                {"role": "user", "content": f"{intext}"}
            ],
            "stream": False,
            "options": {
                "temperature": 0.6,
                "top_p": 0.9
            }
        }
        
        # APIリクエストの実行
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        
        # 結果の取得
        output = response.json().get("message", {}).get("content", "")
        
        # DeepSeek-R1などの推論モデルを使用した場合の<think>タグを除去
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
        
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
    output_file = "result_ollama_deepseek.csv"
    df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ All data processed successfully. Results saved to {output_file}")
else:
    print("\n⚠️ No data was processed. CSV was not created.")