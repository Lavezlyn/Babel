# 分析脚本集合

本文件夹包含所有语言涌现和演化分析相关的脚本。

## 文件列表

### 核心分析脚本

1. **analyze_emergent_language.py**
   - 主要分析脚本：从 JSON 日志提取消息、token、n-gram 数据
   - 生成 CSV 文件和定性摘要
   - 分析评分、交易质量等指标

2. **analyze_language_evolution.py**
   - 语言演化分析：计算模式生命周期、扩散、演化路径等指标
   - 生成语言演化相关的 CSV 数据

3. **analyze_code_word_significance.py**
   - Code Word 显著性分析：区分 code word 和 resource name
   - 计算 Code Word Score 等综合指标

### 可视化脚本

4. **visualize_emergent_language.py**
   - 生成语言涌现相关的图表（Nature 期刊风格）
   - 包括评分分布、时间趋势、模式-评分相关性等

5. **visualize_language_evolution.py**
   - 生成语言演化相关的图表（EMNLP/Nature 风格）
   - 包括生命周期、词汇演化、结构演化等可视化

### 表格生成脚本

6. **generate_main_table.py**
   - 生成论文主表格（main_table.csv 和 main_table.tex）
   - 使用 Code Word Score ≥ 0.7 标准，排除资源名称

7. **generate_table_only.py**
   - 仅生成语言演化摘要表格（table_evolution_summary.tex）

8. **generate_evolution_table.py**
   - 生成演化相关的表格

### 辅助脚本

9. **show_code_words_details.py**
   - 显示表格中统计的具体 Code Words 和 N-grams 详情
   - 包括词的含义、使用次数、评分等

10. **resource_exchange_to_text.py**
    - 将 JSON 日志转换为可读的 TXT 文件
    - 支持批量处理和翻译

## 使用流程

### 1. 数据预处理
```bash
# 将 JSON 日志转换为 TXT（可选）
python resource_exchange_to_text.py \
    --input_dir examples/logs \
    --output_dir examples/logs_txt
```

### 2. 核心分析
```bash
# 分析语言涌现现象
python analyze_emergent_language.py \
    --input_dir examples/logs \
    --output_dir results/emergent_language

# 分析语言演化
python analyze_language_evolution.py \
    --data_dir results/emergent_language \
    --output_dir results/language_evolution
```

### 3. Code Word 分析（可选）
```bash
# Code Word 显著性分析
python analyze_code_word_significance.py \
    --data_dir results/emergent_language \
    --output_dir results/code_word_analysis
```

### 4. 可视化
```bash
# 生成语言涌现图表
python visualize_emergent_language.py \
    --data_dir results/emergent_language \
    --output_dir results/emergent_language/figures

# 生成语言演化图表
python visualize_language_evolution.py \
    --data_dir results/language_evolution \
    --output_dir results/language_evolution/figures
```

### 5. 生成表格
```bash
# 生成主表格
python generate_main_table.py \
    --data_dir results/emergent_language \
    --output results/main_table.csv

# 查看 Code Words 详情
python show_code_words_details.py \
    --data_dir results/emergent_language \
    --output results/code_words_details.txt
```

## 依赖关系

```
resource_exchange_to_text.py (可选)
    ↓
analyze_emergent_language.py
    ↓
    ├── analyze_language_evolution.py
    ├── analyze_code_word_significance.py
    ├── visualize_emergent_language.py
    ├── visualize_language_evolution.py
    └── generate_main_table.py
            ↓
        show_code_words_details.py
```

## 输出文件说明

### analyze_emergent_language.py 输出
- `messages.csv`: 所有消息的详细数据
- `unknown_tokens.csv`: 未知 token 统计
- `tail_ngrams.csv`: 尾部 n-gram 统计
- `summary_*.txt`: 定性分析摘要

### analyze_language_evolution.py 输出
- `lifecycle_metrics.csv`: 生命周期指标
- `diffusion_metrics.csv`: 扩散指标
- `evolution_paths.csv`: 演化路径
- `lexical_metrics.csv`: 词汇指标
- `structural_metrics.csv`: 结构指标

### generate_main_table.py 输出
- `main_table.csv`: CSV 格式主表格
- `main_table.tex`: LaTeX 格式主表格

## 注意事项

1. **运行顺序**：建议按照上述流程顺序运行
2. **数据路径**：确保输入数据目录存在且包含所需文件
3. **依赖包**：需要安装 `matplotlib`, `seaborn` 等（见 `requirements.txt`）
4. **Code Word Score**：主表格使用 Code Word Score ≥ 0.7 标准，排除资源名称

## 相关文档

- `CODE_WORD_SIGNIFICANCE_GUIDE.md`: Code Word 显著性判断指南
- `MAIN_TABLE_OPTIMIZATION.md`: 主表格优化说明
- `LANGUAGE_EVOLUTION_METRICS.md`: 语言演化指标说明
- `REGENERATE_TABLE.md`: 重新生成表格指南

