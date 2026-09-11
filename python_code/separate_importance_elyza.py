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
    dtype="auto",
    device_map="auto",
)

# --- ステップ2：分類専用のプロンプト ---
# 入力されたテキスト全体を読み、分類行を「追記」するように指示
instructs_classifier = """あなたは、プログラミングコードの分析結果をレビューする専門家です。
入力されたテキスト（「###言語：」から始まる）を読み、その中の「###重要な部分：」セクションに含まれる**各項目**（###1., ###2., ...）を分析してください。

以下の###定義に基づき、各項目の「コードブロック」と「説明文」を評価し、**「重要度の分類：[構造的 または ステートメント的]」**という行を、**各説明文の直後**に追加してください。

###定義：
* **構造的**：プログラム全体の**流れ（フロー）**を制御する部分。**タイマー（「XX毎」）、ループ（「繰り返す」）、条件分岐（「もし」）、関数定義（「●XXとは」）**がこれにあたる。
* **ステートメント的**：構造の中で実行される個々の具体的な処理。**計算式、代入、描画命令（「線描画」「円描画」など）、入出力**がこれにあたる。

###指示：
1. 入力テキスト全体を読み込みます。
2. 「###重要な部分：」セクションを見つけます。
3. 「###1. ...」「###2. ...」など、すべての項目に対して分類を行います。
4. **入力テキストをそのまま再現し、各説明文の末尾に分類結果の行だけを追加**してください。他の部分は絶対に変更しないでください。

###例（入力）：
###言語：
日本語プログラミング
###コードの目的：
合計点を計算し、合格か不合格かを判定します。
###重要な部分：
...
###1. 合格・不合格の判定
[コードブロック]
もし 合計 >= 80 なら ...
[説明文]
この部分は、計算結果に基づきプログラムの最終的な動作を決定する条件分岐です。

###2. 合計点の計算
[コードブロック]
合計 = 点数A + 点B
[説明文]
この部分は、判定の基準となる「合計」の値を算出する具体的な計算処理です。

###例（出力）：
###言語：
日本語プログラミング
###コードの目的：
合計点を計算し、合格か不合格かを判定します。
###重要な部分：
...
###1. 合格・不合格の判定
[コードブロック]
もし 合計 >= 80 なら ...
[説明文]
この部分は、計算結果に基づきプログラムの最終的な動作を決定する条件分岐です。
**重要度の分類**：構造的

###2. 合計点の計算
[コードブロック]
合計 = 点数A + 点B
[説明文]
この部分は、判定の基準となる「合計」の値を算出する具体的な計算処理です。
**重要度の分類**：ステートメント的
"""
# -----------------------------------------------

# --- ★処理対象の入力 (ステップ1の出力をここに貼り付ける) ---
intext = """
###言語：
日本語プログラミング

###コードの目的：
数当てゲームを無限ループで繰り返し、ユーザーが正しい答えを推測するまで点数を加算し、結果を表示するプログラムです。

###重要な部分：
以下は、特に重要な部分を抜き出したコードブロックとその説明です。（★**重要度が高い順に1から番号が振られています**）

###1. 無限ループの定義
# ゲームを無限ループでずっと繰り返す --- (*1)
永遠の間繰り返す

この部分は、プログラムの最も重要な部分で、無限ループを定義しています。(*1)

###2. 推測値の生成と条件分岐
　　答え＝2の乱数
　　「数当てゲーム。0か1を入力して」と尋ねて推測値に代入。
　　もし答えが推測値ならば
　　　　点数＝点数＋1
　　　　結果＝「当たり⭐」
　　違えば
　　　　結果＝「はずれ😭」

この部分は、推測値を生成し、ユーザーの入力と比較して、当たりかはずれかを判定し、点数を加算する重要なロジックです。

###3. 結果表示と二択の提示
　　「{結果}。点数は{点数}点。続けますか？」

この部分は、結果を表示し、ユーザーに続けるかどうかを尋ねる二択を提示する重要な部分です。
"""
# -----------------------------------------------


print("\n--- Starting Step 2: Classification (Bulk Mode) ---")

# チャットテンプレート
chats = [
    { "role": "system", "content": instructs_classifier},
    { "role": "user", "content": intext}, # ステップ1の出力をそのまま入力
]

# プロンプトのフォーマット
formatted_prompt = tokenizer.apply_chat_template(
    chats,
    tokenize=False,
    add_generation_prompt=True,
)

# エンコード
encoded_input = tokenizer(
    formatted_prompt,
    return_tensors="pt",
    return_attention_mask=True
)
token_ids = encoded_input["input_ids"]
attention_mask = encoded_input["attention_mask"]

# 推論の実行 (分類タスク)
with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        do_sample=True,
        temperature=0.3, # 分類タスクなので低めの温度
        top_p=0.7,
        max_new_tokens=2048, # 全体を再構成するため長めに確保
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ],
        pad_token_id=tokenizer.eos_token_id
    )

# デコードして出力
classification_output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
print("\n--- LLM Analysis Output (Classified) ---\n")
print(classification_output.strip())
print("\n--- End of Output ---\n")