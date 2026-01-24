# PHOENIX-2014-T 数据集快速开始

## 快速命令

### 1. 处理所有划分（推荐用于完整数据集）

```bash
python data/extract_phoenix_keypoints.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints.pkl
```

### 2. 测试模式（只处理少量样本）

```bash
python data/extract_phoenix_keypoints.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints_test.pkl \
    --max_samples_per_split 5
```

### 3. 只处理训练集

```bash
python data/extract_phoenix_keypoints.py \
    --dataset_path /data/phd/wsun/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --output_path phoenix_keypoints_train.pkl \
    --splits train
```

## 输出文件说明

提取完成后，会生成一个 pickle 文件，包含：

- `train`: 训练集关键点数据
- `dev`: 验证集关键点数据
- `test`: 测试集关键点数据
- `stats`: 统计信息
- `keypoint_info`: 关键点信息说明

## 数据使用示例

```python
import pickle

# 加载数据
with open('phoenix_keypoints.pkl', 'rb') as f:
    data = pickle.load(f)

# 访问训练集
train_keypoints = data['train']['keypoints']
train_video_ids = data['train']['video_ids']

print(f"训练集样本数: {len(train_keypoints)}")
print(f"第一个样本的视频ID: {train_video_ids[0]}")
```

## 更多信息

详细说明请参考：[PHOENIX数据集关键点提取指南.md](./PHOENIX数据集关键点提取指南.md)





