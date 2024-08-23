import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

def create_animation(csv_file, output_file):
    # CSVファイルの読み込み
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    # 日付ごとにデータをグループ化
    daily_data = df.groupby(df.index.date)

    # アニメーションの設定
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title(f"State Vector Visualization for {os.path.basename(csv_file).split('_')[0]}")
    plt.xlabel("State Vector Dimensions")
    plt.ylabel("Value")

    # カスタムカラーマップの作成（青から赤）
    colors = ['blue', 'lightblue', 'white', 'pink', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

    # 初期フレームの作成
    bars = ax.bar(range(len(df.columns)), np.zeros(len(df.columns)))
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # アニメーション更新関数
    def update(frame):
        date, data = list(daily_data)[frame]
        heights = data.mean().values
        for bar, h in zip(bars, heights):
            bar.set_height(h)
            bar.set_color(cmap(h))
        ax.set_ylim(0, 1)
        text.set_text(date.strftime('%Y-%m-%d'))
        return bars.patches + [text]  # ここを修正

    # アニメーションの作成
    anim = animation.FuncAnimation(fig, update, frames=len(daily_data), 
                                   interval=200, blit=True)

    # 動画として保存
    writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(output_file, writer=writer)
    plt.close()

def main():
    input_dir = 'output_data'
    output_dir = 'animations'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith('_state_vector.csv'):
            input_file = os.path.join(input_dir, file)
            output_file = os.path.join(output_dir, f"{file.split('_')[0]}_animation.mp4")
            print(f"Processing {file}...")
            create_animation(input_file, output_file)
            print(f"Animation saved as {output_file}")

if __name__ == "__main__":
    main()