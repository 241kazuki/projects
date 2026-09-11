import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    dtype="auto", # 'torch_dtype' を 'dtype' に変更
    device_map="auto",
)

# プロンプトの準備（★「重要な計算式」を明示的に指示）
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

###例：
###入力されたコード：
もし　「こんにちは」と言う
そうでなければ　「さようなら」と言う
終わり

###出力例：
###言語：
日本語プログラミング

###コードの目的：
与えられた条件に応じて、異なるメッセージを出力するプログラムです。

###重要な部分：
以下は、特に重要な部分を抜き出したコードブロックとその説明です。（★**重要度が高い順に1から番号が振られています**）

###1. 条件分岐
もし　「こんにちは」と言う
そうでなければ　「さようなら」と言う

この部分は、プログラムの挙動を決定づける条件分岐のロジックです。「もし」の条件が真であれば「こんにちは」と表示し、そうでなければ「さようなら」と表示します。
"""

# --- 処理対象の単一入力 ---
intext = """
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
# -------------------------

# チャットテンプレート
chats = [
    { "role": "system", "content": instructs},
    { "role": "user", "content": intext},
]

# apply_chat_template でフォーマットされた文字列を取得
formatted_prompt = tokenizer.apply_chat_template(
    chats,
    tokenize=False,
    add_generation_prompt=True,
)

# その後、tokenizer を使ってエンコードし、input_ids と attention_mask を辞書で取得
encoded_input = tokenizer(
    formatted_prompt,
    return_tensors="pt",
    return_attention_mask=True
)

token_ids = encoded_input["input_ids"]
attention_mask = encoded_input["attention_mask"]

# 推論の実行
print("\nStarting model inference for the single input...")
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

# 出力
print("\n--- LLM Analysis Output ---\n")
print(output)
print("\n--- End of Output ---\n")