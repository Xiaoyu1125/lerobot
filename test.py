from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

# 测试本地数据集 - 注意：StreamingLeRobotDataset 需要 streaming=True
# 即使是本地数据，也是以流式方式读取，避免一次性加载到内存
dataset = StreamingLeRobotDataset(
    repo_id="lerobot/svla_so100_stacking",
    root="/public/xiaoyu/svla_so100_stacking",  # 本地路径
    buffer_size=2,
    streaming=True,  # 必须为 True
    video_backend="pyav"  # 使用 pyav 后端，支持 AV1 视频
)

# 测试迭代
for i, item in enumerate(dataset):
    print(f"Frame {i}: {list(item.keys())}")
    if i >= 2:
        break