# Pipeline 参数检查清单

## ✅ 验证结果总结

已运行 `validate_pipeline.py` 进行完整性检查，结果如下：

**注意**：手语识别 Pipeline 只使用以下三个核心模型：
- Skeleton Model（骨架重构）
- Temporal Model（时序建模）
- Alignment Model（语言对齐）

`hierarchical_keypoint` 和 `hierarchical_image` 是早期实验模型（用于68点面部关键点），与手语识别无关，保留用于参考。

### ✅ 通过的检查

1. **配置文件**：所有模型配置完整
   - Skeleton 模型配置 ✓
   - Temporal 模型配置 ✓
   - Alignment 模型配置 ✓

2. **训练脚本**：所有脚本存在且可访问
   - `training/skeleton/train.py` ✓
   - `training/temporal/train.py` ✓
   - `training/alignment/train.py` ✓

3. **推理脚本**：所有脚本存在且可访问
   - `inference/temporal/inference.py` ✓
   - `inference/alignment/inference.py` ✓

4. **模型文件**：所有模型定义存在
   - `models/skeleton/model.py` ✓
   - `models/temporal/model.py` ✓
   - `models/alignment/model.py` ✓

5. **数据文件**：必要的数据文件存在
   - `sign_language_keypoints.pkl` ✓
   - `video_sequences.pkl` ✓
   - `text_data.json` ✓

6. **路径一致性**：所有路径配置正确
   - Temporal frame encoder checkpoint 路径 ✓
   - Alignment video reprs 路径 ✓

### ⚠️ 需要注意的项

1. **mBART 模型路径**：`../model/mbart-large-cc25`
   - 状态：路径不存在
   - 说明：这是正常的，需要用户提供 mBART 模型
   - 影响：仅影响 Alignment 模型训练
   - 解决方案：确保 mBART 模型已下载到指定路径，或修改 `config/config.py` 中的 `MBART_MODEL_PATH`

2. **Skeleton 模型检查点**：`checkpoints_skeleton_hierarchical/best_model.pth`
   - 状态：不存在（这是正常的，第一次运行时会生成）
   - 说明：Temporal 模型训练需要此检查点
   - 影响：需要先运行 Skeleton 模型训练

## 🔄 Pipeline 流程检查

### 阶段 1: Skeleton 模型训练

**输入要求**：
- ✅ `sign_language_keypoints.pkl` (存在)

**输出**：
- `checkpoints_skeleton_hierarchical/best_model.pth` (待生成)

**状态**：✅ 可以运行

### 阶段 2: Temporal 模型训练

**输入要求**：
- ✅ `video_sequences.pkl` (存在)
- ⚠️ `checkpoints_skeleton_hierarchical/best_model.pth` (需要先运行阶段1)

**输出**：
- `checkpoints_temporal/best_model_stage1.pth` (待生成)

**状态**：⚠️ 需要先完成阶段1

### 阶段 2.5: Temporal 模型推理

**输入要求**：
- `checkpoints_temporal/best_model_stage1.pth` (需要阶段2)

**输出**：
- ✅ `temporal_representations_all/all_representations.npz` (已存在)

**状态**：⚠️ 推理输出已存在，可以跳过或重新生成

### 阶段 3: Alignment 模型训练

**输入要求**：
- ✅ `temporal_representations_all/all_representations.npz` (存在)
- ✅ `text_data.json` (存在)
- ⚠️ mBART 模型路径 (需要配置)

**输出**：
- `checkpoints_alignment/best_model.pth` (待生成)

**状态**：⚠️ 需要配置 mBART 模型路径

### 阶段 4: Alignment 模型推理

**输入要求**：
- `checkpoints_alignment/best_model.pth` (需要阶段3)

**输出**：
- `generation_results.json` (待生成)

**状态**：⚠️ 需要先完成阶段3

## 📋 运行前检查清单

### 必需项

- [x] 配置文件 (`config/config.py`) 存在且有效
- [x] 数据文件存在
- [x] 训练和推理脚本存在
- [x] 模型定义文件存在
- [ ] mBART 模型路径配置正确（如果运行 Alignment 模型）

### 可选但推荐

- [ ] 检查 GPU 可用性（如果使用 GPU）
- [ ] 检查磁盘空间（足够存储检查点和输出）
- [ ] 备份现有数据（如果需要）

## 🚀 快速开始

### 运行完整 Pipeline

```bash
# 1. 验证环境
python validate_pipeline.py

# 2. 运行完整 pipeline
python run_pipeline.py

# 3. 或者运行特定阶段
python run_pipeline.py --stage skeleton
```

### 单独运行各阶段

```bash
# 阶段 1: 训练 Skeleton 模型
python training/skeleton/train.py --config config/config.py

# 阶段 2: 训练 Temporal 模型
python training/temporal/train.py --config config/config.py

# 阶段 2.5: 提取时序表征
python inference/temporal/inference.py --config config/config.py

# 阶段 3: 训练 Alignment 模型
python training/alignment/train.py --config config/config.py

# 阶段 4: 生成文本描述
python inference/alignment/inference.py --config config/config.py
```

## 🔧 已知问题修复

### 问题 1: mBART 模型路径不存在

**解决方案**：
1. 下载 mBART 模型到指定路径
2. 或修改 `config/config.py` 中的 `MBART_MODEL_PATH` 为正确的路径
3. 或使用 HuggingFace 模型名称（需要网络连接）

### 问题 2: 路径拼接使用字符串拼接而非 os.path.join

**状态**：✅ 已修复
- `run_pipeline.py` 中已使用 `os.path.join` 进行路径拼接

### 问题 3: 配置文件导入错误处理

**状态**：✅ 已修复
- `run_pipeline.py` 中已添加 try-except 错误处理

## 📝 参数一致性检查

所有脚本的参数都与配置文件保持一致：

1. **训练脚本**：
   - ✅ 支持 `--config` 参数从配置文件读取
   - ✅ 支持命令行参数覆盖配置值
   - ✅ 参数名称与配置文件一致

2. **推理脚本**：
   - ✅ 支持 `--config` 参数从配置文件读取
   - ✅ 支持命令行参数覆盖配置值
   - ✅ 参数名称与配置文件一致

3. **Pipeline 脚本**：
   - ✅ 自动传递配置文件路径
   - ✅ 检查点路径检查使用 `os.path.join`
   - ✅ 错误处理完善

## ✅ 结论

Pipeline 基本可以正常运行，但需要注意：

1. **mBART 模型路径**：需要配置正确的 mBART 模型路径
2. **检查点依赖**：按顺序运行各阶段（skeleton → temporal → alignment）
3. **数据准备**：确保数据文件已准备好

建议：
- 先运行 `python validate_pipeline.py` 检查环境
- 然后运行 `python run_pipeline.py` 执行完整流程
- 如果遇到问题，查看具体错误信息并参考本文档

