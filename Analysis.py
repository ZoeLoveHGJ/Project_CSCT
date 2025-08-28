import pandas as pd
import os
import glob


def analyze_csv_files_recursive():
    search_path = os.path.join('**', '*.csv')
    csv_files = glob.glob(search_path, recursive=True)

    if not csv_files:
        print("在当前目录及其所有子目录下没有找到任何CSV文件。")
        return

    results = []

    for file_path in csv_files:
        try:

            folder_path = os.path.dirname(file_path)

            if folder_path == '':
                folder_name = os.path.basename(os.getcwd())
            else:
                folder_name = os.path.basename(folder_path)

            df = pd.read_csv(file_path)
            file_name = os.path.basename(file_path)

            baseline_col_name = None
            if 'CSCT' in df.columns:
                baseline_col_name = 'CSCT'
            elif 'ISCT_Update3' in df.columns:
                baseline_col_name = 'ISCT_Update3'
            else:
                print(
                    f"警告: 在文件 '{file_path}' 中未找到 'CSCT' 或 'ISCT_Update3' 列，已跳过。")
                continue

            numeric_cols = df.select_dtypes(
                include=['number']).columns.tolist()

            if baseline_col_name not in numeric_cols:
                print(
                    f"警告: 基准列 '{baseline_col_name}' 在文件 '{file_path}' 中不是数值类型，已跳过。")
                continue

            mean_values = df[numeric_cols].mean()

            baseline_value = mean_values[baseline_col_name]

            comparison_cols = mean_values.drop(
                baseline_col_name, errors='ignore')

            if comparison_cols.empty:
                print(f"警告: 在文件 '{file_path}' 中没有其他可供对比的数值列，已跳过。")
                continue

            suboptimal_value = comparison_cols.max()
            suboptimal_name = comparison_cols.idxmax()

            if suboptimal_value == 0:
                percentage_improvement = float('inf')
            else:

                percentage_improvement = (
                    (baseline_value - suboptimal_value) / suboptimal_value) * 100

            results.append({
                '文件夹名称': folder_name,
                '文件名': file_name,
                'CSCT的数值': baseline_value,
                '对比算法名称': suboptimal_name,
                '对比算法数值': suboptimal_value,
                '提升的百分比': f"{percentage_improvement:.2f}%"
            })

        except Exception as e:
            print(f"处理文件 '{file_path}' 时发生错误: {e}")

    if results:

        output_df = pd.DataFrame(results)

        final_columns = ['文件夹名称', '文件名', 'CSCT的数值',
                         '对比算法名称', '对比算法数值', '提升的百分比']
        output_df = output_df[final_columns]
        print(output_df.to_string(index=False))
    else:
        print("分析完成，但没有生成任何结果。请检查CSV文件内容和格式是否符合要求。")


if __name__ == '__main__':
    analyze_csv_files_recursive()
