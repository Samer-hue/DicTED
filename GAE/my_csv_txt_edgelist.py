import csv

def main(a):
    def extract_and_remove_duplicates(input_csv, output_txt):
        unique_rows = set()  # 用于存储唯一的行

        with open(input_csv, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            
            for row in csv_reader:
                u_value = int(row["u"])
                i_value = int(row["i"])
                # 将提取的整数值组成元组，作为唯一行的标识
                unique_row = (u_value, i_value)
                unique_rows.add(unique_row)

        with open(output_txt, 'w') as txt_file:
            for row in unique_rows:
                txt_file.write(f"{row[0]} {row[1]}\n")

    # 调用函数，替换 'input.csv' 和 'output.txt' 为实际的文件路径
    extract_and_remove_duplicates('../../../data/ml_{}.csv'.format(a), '../../../data/ml_{}.txt'.format(a))

    with open('../../../data/ml_{}.txt'.format(a), 'r') as infile:
        with open('../../../data/ml_{}.edgelist'.format(a), 'w') as f:
            for line in infile:
                parts = line.split()
                f.write(parts[0] + ' ' + parts[1] + '\n')

main('uci')