import pandas as pd

# --- 設定：閾値（案A: 40% / 15%） ---
THRESH_LEVEL1 = 0.40  # これ以上なら難易度1（簡単）
THRESH_LEVEL2 = 0.15  # これ以上〜0.40未満なら難易度2（普通）

target_file = 'nadshiko3_blanks_data.csv'

try:
    # 1. データ読み込み
    print(f"Loading {target_file}...")
    try:
        df = pd.read_csv(target_file, encoding='cp932')
    except UnicodeDecodeError:
        df = pd.read_csv(target_file, encoding='utf-8-sig')

    # 2. 空欄率の計算（オリジナル vs レベル1）
    def calculate_ratio(row):
        original = str(row['original_code'])
        l1 = str(row['branks_level1'])
        
        # ハイフンを除去して「ヒントとして残っている文字数」をカウント
        l1_clean = l1.replace('-', '')
        len_orig = len(original)
        
        if len_orig == 0: return 0.0
        
        # 空欄になった文字数 = オリジナル - 残存
        diff = len_orig - len(l1_clean)
        
        # 空欄率
        return max(0, diff) / len_orig

    df['blank_ratio_l1'] = df.apply(calculate_ratio, axis=1)

    # 3. 難易度と理由の判定（ユーザー様のロジックを反映）
    def judge_difficulty_and_reason(ratio):
        ratio_percent = ratio * 100
        
        if ratio >= THRESH_LEVEL1:
            difficulty = 1
            reason = (f"空欄率が{ratio_percent:.1f}%と高く、問題文などから容易に推測できる"
                      f"「簡単な穴埋め箇所」が多くを占めているため、初級レベルと判定されます。")
            
        elif ratio >= THRESH_LEVEL2:
            difficulty = 2
            reason = (f"空欄率が{ratio_percent:.1f}%であり、問題文から推測可能な箇所が"
                      f"標準的な割合で含まれているため、中級レベルと判定されます。")
            
        else:
            difficulty = 3
            reason = (f"空欄率が{ratio_percent:.1f}%と低く、問題文から容易に推測できる"
                      f"簡単な箇所が少ないため、上級レベルと判定されます。")
            
        return f"難易度: {difficulty}\n理由: {reason}"

    # 判定の適用
    df['elyza_judgment'] = df['blank_ratio_l1'].apply(judge_difficulty_and_reason)

    # 4. 上書き保存
    df.to_csv(target_file, index=False, encoding='utf-8-sig')
    print(f"処理が完了しました。{target_file} に保存されました。")
    

except Exception as e:
    print(f"エラーが発生しました: {e}")