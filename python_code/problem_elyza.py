import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#モデルElyza8b
model_id = "elyza/Llama-3-ELYZA-JP-8B"

# デバイスの確認
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(device)

# トークナイザーとモデルの準備
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

# プロンプトの準備
instructs ="""
あなたは、経験豊富なプログラミング教材の作成者です。
あなたの目的は、入力された「模範解答」のコードを分析し、**そのコードと全く同じ動作をするプログラムをゼロから作成させる**ために必要な、「**詳細な仕様がすべて含まれた、自然な文章の問題文（お題）**」を生成することです。

入力されるコードは「日本語プログラミング言語」であることを前提とします。

### 指示：
1.  入力された「模範解答」のコードを分析し、その動作を完全に再現するために必要な「仕様」を**すべて**抽出してください。
    * （入力要求、計算式、分岐条件、出力形式など）
2.  抽出したすべての仕様を、**箇条書きのリストにはせず**、一つの連続した「問題文」の文章内に**自然に溶け込ませてください**。
3.  完成した問題文は、**ユーザーに対する「～してください」という明確な作成指示**になっていなければなりません。
4.  **重要な制約**：コード内の具体的な変数名は避け、仕様を記述してください。
5.  生成した問題文の難易度を（初級）または（中級）で示してください。

### 入力コード（模範解答）：
[ここに模範解答となる日本語プログラミングのコードを貼り付ける]

### 出力形式：
（「問題文：」や「仕様：」などの余計な見出しを含めず、生成された問題文と難易度のみを出力してください）
問題文
（[初級/中級]） [抽出したすべての仕様が、自然な文章として一つに統合された作成指示]

"""
# ユーザの質問文 (ここにコードを入力してください)
intext = """
# 変数の初期化 --- (*1)
TODOは[]
保存キーは「最小TODO」
タイトルは「
─────────────────────────────
📝 最小TODO v2
─────────────────────────────
すべきことを入力してください（キャンセルで終了）
数字を入力するとTODOを削除します。
--- 以下TODO一覧 ---
」

# 以前保存したデータがあれば読み込む --- (*2)
もし、保存キーが存在するならば
　　保存キーを読んで、TODOに代入。
ここまで。

# キャンセルが押されるまで永遠に繰り返す
永遠の間繰り返す
　　# 表示内容を作成する --- (*3)
　　TODO項目列挙してTODO一覧に代入。
　　「{タイトル}{TODO一覧}」を尋ねてVに代入。
　　もし、Vが空ならば、抜ける。
　　# 追加か削除か判定(数列なら削除) --- (*4)
　　もし((Vを数列判定)＝はい)ならば
　　　　TODOのVを配列削除。
　　違えば
　　　　TODOにVを配列追加。
　　ここまで。
　　TODOを保存キーに保存。# --- (*5)
ここまで。

●TODO項目列挙とは # --- (*6)
　　S=「」
　　TODOを反復
　　　　S=S&「📌 {対象キー}: {対象}{改行}」
　　ここまで。
　　それはS
ここまで。

"""

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
output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
# 出力
print(output)