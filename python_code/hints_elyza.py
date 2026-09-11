import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# モデルElyza8b
model_id = "elyza/Llama-3-ELYZA-JP-8B"

# デバイスの確認
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# トークナイザーとモデルの準備
tokenizer = AutoTokenizer.from_pretrained(model_id)
# NameError/ValueError回避のため、標準のロード方法を使用
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
# --- 1. 達成すべき問題文---
problem_statement = """
問題文：身長と体重を入力してください。
身長はセンチメートル、体重はキログラムで入力してください。
入力された身長と体重を用いて、BMIを計算し、肥満、低体重、普通体重のいずれに該当するかを判定し、結果を表示してください。
"""

# --- 2. 処理対象の元のコードブロック ---
intext_code = """
# 身長と体重を尋ねる --- (*1)
    「身長(cm)は？」と尋ねて身長cmに代入。
    「体重(kg)は？」と尋ねて体重kgに代入。
    # BMIを計算 --- (*2)
    BMI=体重kg÷(身長cm/100)^2
    # 肥満かどうか判定 --- (*3)
    もし、BMIが25以上ならば
    　　判定＝「肥満」
    違えば、もし、BMIが18.5未満ならば
    　　判定＝「低体重」
    違えば
    　　判定＝「普通体重」
    ここまで。
    # 結果を表示 --- (*4)
    「BMI={BMI} / 判定={判定}」を表示。
"""

# --- 3. コード分析結果 ---
analysis_result_for_target_code = """
###言語：
日本語プログラミング

###コードの目的：
ユーザーに身長と体重を尋ね、BMIを計算し、肥満かどうかを判定し、結果を表示するプログラムです。

###重要な部分：
以下は、特に重要な部分を抜き出したコードブロックとその説明です。（★**重要度が高い順に1から番号が振られています**）

###1. BMIの計算
BMI=体重kg÷(身長cm/100)^2

この部分は、プログラムの核心部分で、BMIを計算し、肥満かどうかを判定するロジックです。BMIの計算式と肥満かどうかの判定条件が記述されています。
判定:ステートメント的コードブロック

###2. 条件分岐と判定の設定
もし、BMIが25以上ならば
    判定＝「肥満」
違えば、もし、BMIが18.5未満ならば
    判定＝「低体重」
違えば
    判定＝「普通体重」

この部分は、BMIの値に応じて、肥満かどうかを判定する条件分岐のロジックです。BMIの値が25以上、18.5未満、のどちらにも該当しない場合の「普通体重」判定も含まれています。
判定:構造的コードブロック

###3. 結果の表示
「BMI={BMI} / 判定={判定}」

この部分は、最後に結果を表示するためのコードです。BMIと判定結果を組み合わせて、ユーザーに結果を表示します。
判定:ステートメント的コードブロック
"""

# --- 4. 新しいプロンプト  ---
instructs_code_fill = f"""
あなたは、提供されたプログラムコード（###入力コード）に対し、**「###達成すべき問題」と「###事前のコード分析」**の内容に厳密に従い、元のコードの構造を維持しながら空欄化する専門家です。

###指示：
1.「###達成すべき問題」と「###事前のコード分析」の内容を徹底的に比較・参照します。
2.「###事前のコード分析」にある**「判定」（構造的/ステートメント的）**に基づいて、空欄化の対象を決定します。
    a.「判定:構造的コードブロック」の場合：分析ブロック内のコード断片から、制御ワード（例:もし,違えば,の間）とその判定式（例: BMIが25以上）を空欄化の**ターゲット**とします。
    b.「判定:ステートメント的コードブロック」の場合：分析ブロック内のコード断片から、式や文字列に含まれる数値（例: 100）や演算子（例: ÷, ^）を空欄化の**ターゲット**とします。
3.「###事前のコード分析」のすべての重要な部分（例： ###1.、###2.、###3.）を対象とし、**ステップ2で決定したターゲットと一致する箇所**を「###入力コード」から探し出し、空欄（___）にしてください。
4.各分析ブロック（###1., ###2., ###3.）に対応する箇所のうち、それぞれ最低1つ以上を空欄化してください。
5.空欄の数は合計で5箇所から最大10箇所程度を目標とします。
6.コードブロック内の構造は完全に維持してください。
7.出力は空欄化したコードだけにしてください

###達成すべき問題： {problem_statement}

###事前のコード分析： {analysis_result_for_target_code}

###入力コード： {intext_code}

###出力（空欄化されたコードブロック）：
"""

# チャットテンプレート
chats_code_fill = [
    { "role": "system", "content": instructs_code_fill},
]

# apply_chat_template でフォーマットされた文字列を取得
formatted_prompt_code_fill = tokenizer.apply_chat_template(
    chats_code_fill,
    tokenize=False,
    add_generation_prompt=True,
)

# エンコードと推論の準備
encoded_input_code_fill = tokenizer(
    formatted_prompt_code_fill,
    return_tensors="pt",
    return_attention_mask=True
)

token_ids_code_fill = encoded_input_code_fill["input_ids"]
attention_mask_code_fill = encoded_input_code_fill["attention_mask"]

# 推論の実行
print("\nStarting model inference for code block blanking (Prioritizing Structural Blocks)...")
with torch.no_grad():
    output_ids_code_fill = model.generate(
        token_ids_code_fill.to(model.device),
        attention_mask=attention_mask_code_fill.to(model.device),
        do_sample=True,   # サンプリングを有効化
        temperature=0.5, # 創造性を適度に上げる
        top_p=0.9,       # 選択肢の多様性を許容
        max_new_tokens=512,
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ],
        pad_token_id=tokenizer.eos_token_id
    )
    
# output_idsから入力プロンプト部分を除去し、デコード
output_code_fill = tokenizer.decode(output_ids_code_fill.tolist()[0][token_ids_code_fill.size(1) :], skip_special_tokens=True)

# 出力
print("\n--- Generated Blanked Code Block (Prioritizing Structural Blocks) ---\n")
print(output_code_fill)
print("\n--- End of Output ---\n")