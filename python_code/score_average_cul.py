import pandas as pd

def calculate_metrics_average(csv_file_path):
    print(f"Reading file: {csv_file_path}...")
    
    # CSVファイルの読み込み
    # エラー回避のため、utf-8-sig (BOM付き) と cp932 (Shift-JIS) の両方を試行
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file_path, encoding='cp932')
        except UnicodeDecodeError:
            print("エラー: ファイルの読み込みに失敗しました。エンコーディングを確認してください。")
            return

    # 集計対象の列名（CSVの実際のヘッダー名に合わせています）
    target_cols = ['content_match', 'content_similarity']
    
    # 列の存在確認
    if not all(col in df.columns for col in target_cols):
        print(f"エラー: 指定された列 {target_cols} がCSV内に見つかりません。")
        print(f"現在の列名: {df.columns.tolist()}")
        return

    # 1. 全体の平均値を計算
    print("\n=== 全体の平均値 (Overall Average) ===")
    overall_mean = df[target_cols].mean()
    print(overall_mean)

    # 2. レベルごとの平均値を計算（level列がある場合）
    if 'level' in df.columns:
        print("\n=== レベルごとの平均値 (Average per Level) ===")
        level_mean = df.groupby('level')[target_cols].mean()
        print(level_mean)
        
        # 結果をCSVとして保存したい場合は以下のコメントアウトを外してください
        # level_mean.to_csv("average_scores_summary.csv")
    else:
        print("\n注意: 'level'列がないため、レベル別集計はスキップされました。")

if __name__ == "__main__":
    # ファイル名を指定して実行
    target_file = "final_evaluation_results.csv"
    calculate_metrics_average(target_file)