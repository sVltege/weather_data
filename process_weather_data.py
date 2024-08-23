import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

def aggregate_data(df, period='D'):
    if period == 'all':
        return df
    elif period == 'pentad':
        # 半旬（5日）ごとに集約
        return df.groupby(pd.Grouper(freq='5D')).mean()
    else:
        return df.groupby(pd.Grouper(freq=period)).mean()

def process_weather_data(source_dir, output_dir, aggregation_period='D'):
    # CSVファイルの読み込み
    csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
    dataframes = {}
    categorical_vars = []
    continuous_vars = []

    for file in csv_files:
        df = pd.read_csv(os.path.join(source_dir, file), index_col='datetime', parse_dates=True)
        var_name = file.split('.')[0]
        dataframes[var_name] = df
        if 'description' in file:
            categorical_vars.append(var_name)
        else:
            continuous_vars.append(var_name)

    print("Categorical variables:", categorical_vars)
    print("Continuous variables:", continuous_vars)

    # 各場所に対して処理を行う
    for location in dataframes[continuous_vars[0]].columns:
        # カテゴリカル変数のOne-Hotエンコーディング
        categorical_data = pd.concat([dataframes[var][location] for var in categorical_vars], axis=1)
        encoder = OneHotEncoder(sparse_output=False)
        onehot_data = encoder.fit_transform(categorical_data)
        onehot_columns = encoder.get_feature_names_out()
        onehot_df = pd.DataFrame(onehot_data, index=categorical_data.index, 
                                 columns=onehot_columns)

        # 連続変数の標準化（MinMaxScalerを使用）
        continuous_data = pd.concat([dataframes[var][location] for var in continuous_vars], axis=1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(continuous_data)
        scaled_df = pd.DataFrame(scaled_data, index=continuous_data.index, columns=continuous_vars)

        # 状態ベクトルの作成
        state_vector = pd.concat([onehot_df, scaled_df], axis=1)

        # 状態ベクトルの正規化（すでに非負なので、そのまま使用可能）
        state_vector_normalized = state_vector.div(state_vector.sum(axis=1), axis=0)

        # データの集約
        state_vector_aggregated = aggregate_data(state_vector_normalized, aggregation_period)

        # 結果をCSVファイルとして保存
        output_file = os.path.join(output_dir, f'{location}_state_vector_{aggregation_period}.csv')
        state_vector_aggregated.to_csv(output_file)

    print("処理が完了しました。")

# スクリプトの実行
if __name__ == "__main__":
    source_dir = 'source_data'  # CSVファイルが格納されているディレクトリ
    output_dir = 'output_data'  # 出力ファイルを保存するディレクトリ
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 集約期間を指定（'D'：日毎、'H'：時間毎、'W'：週毎、'M'：月毎、'pentad'：半旬毎、'all'：集約なし）
    aggregation_period = 'D'
    
    process_weather_data(source_dir, output_dir, aggregation_period)