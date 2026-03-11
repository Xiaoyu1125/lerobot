#!/usr/bin/env python3
"""
将数据集中的视频从 AV1 编码转换为 H264 编码
H264 解码速度比 AV1 快 5-10 倍
非交互式版本，直接执行转换
"""

import subprocess
import shutil
from pathlib import Path
import time
import sys
import tempfile

# 视频根目录
VIDEO_ROOT = Path("/public/xiaoyu/svla_so100_stacking/videos")
# 备份目录
BACKUP_ROOT = Path("/public/xiaoyu/svla_so100_stacking/videos_backup")

# FFmpeg 参数
FFMPEG_CMD = (
    "ffmpeg -y -i {input} "
    "-c:v libx264 "
    "-preset fast "
    "-crf 23 "
    "-g 2 "
    "-pix_fmt yuv420p "
    "{output}"
)


def get_video_files(video_root: Path) -> list[Path]:
    """获取所有需要转换的视频文件"""
    video_files = []
    for ext in ["*.mp4", "*.mkv", "*.webm"]:
        video_files.extend(video_root.rglob(ext))
    return sorted(video_files)


def backup_videos(video_root: Path, backup_root: Path) -> Path:
    """备份原始视频"""
    if backup_root.exists():
        print(f"⚠ 备份目录已存在，跳过备份: {backup_root}")
        return backup_root

    print(f"正在备份视频到: {backup_root}")
    start_time = time.time()

    shutil.copytree(video_root, backup_root)

    elapsed = time.time() - start_time
    print(f"✓ 备份完成，耗时: {elapsed:.1f}秒")

    return backup_root


def convert_video(input_path: Path) -> dict:
    """转换单个视频文件"""
    print(f"\n转换: {input_path.name}")

    start_time = time.time()
    original_size = input_path.stat().st_size / (1024 ** 2)  # MB

    # 创建临时文件
    temp_path = input_path.with_suffix('.tmp.mp4')

    # 构建命令
    cmd = FFMPEG_CMD.format(input=input_path, output=temp_path)

    # 执行转换
    print(f"  命令: ffmpeg -i {input_path.name} -c:v libx264 ...")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start_time

    # 获取新文件大小
    if temp_path.exists():
        new_size = temp_path.stat().st_size / (1024 ** 2)  # MB

        # 替换原文件
        input_path.unlink()
        temp_path.rename(input_path)

        size_ratio = (new_size / original_size) * 100
    else:
        new_size = 0
        size_ratio = 0

    success = result.returncode == 0

    return {
        "success": success,
        "original_size_mb": original_size,
        "new_size_mb": new_size,
        "size_ratio": size_ratio,
        "time_seconds": elapsed,
        "stderr": result.stderr,
    }


def main():
    print("="*80)
    print("AV1 → H264 视频编码转换工具")
    print("="*80)

    # 查找所有视频文件
    video_files = get_video_files(VIDEO_ROOT)

    if not video_files:
        print(f"✗ 未找到视频文件: {VIDEO_ROOT}")
        sys.exit(1)

    print(f"\n找到 {len(video_files)} 个视频文件:")
    total_size = 0
    for f in video_files:
        size_mb = f.stat().st_size / (1024 ** 2)
        total_size += size_mb
        print(f"  - {f.relative_to(VIDEO_ROOT)} ({size_mb:.1f} MB)")
    print(f"  总计: {total_size:.1f} MB ({total_size/1024:.2f} GB)")

    # 备份原始视频
    print("\n" + "="*80)
    print("步骤 1/2: 备份原始视频")
    print("="*80)
    backup_videos(VIDEO_ROOT, BACKUP_ROOT)

    # 转换视频
    print("\n" + "="*80)
    print("步骤 2/2: 转换视频编码")
    print("="*80)

    results = []
    total_original_size = 0
    total_new_size = 0
    total_time = 0

    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] {video_file.name}")
        print(f"  输入: {video_file}")

        result = convert_video(video_file)
        results.append(result)

        total_original_size += result["original_size_mb"]
        total_new_size += result["new_size_mb"]
        total_time += result["time_seconds"]

        if result["success"]:
            print(f"  ✓ 成功! ({result['time_seconds']:.1f}秒)")
            print(f"    大小: {result['original_size_mb']:.1f}MB → {result['new_size_mb']:.1f}MB ({result['size_ratio']:.1f}%)")
        else:
            print(f"  ✗ 失败!")
            print(f"    stderr: {result['stderr'][:500] if result['stderr'] else 'N/A'}")

    # 汇总报告
    print("\n" + "="*80)
    print("转换完成!")
    print("="*80)
    print(f"总文件数: {len(video_files)}")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"原始大小: {total_original_size:.1f} MB ({total_original_size/1024:.2f} GB)")
    print(f"新大小: {total_new_size:.1f} MB ({total_new_size/1024:.2f} GB)")
    print(f"大小变化: {(total_new_size/total_original_size)*100:.1f}%")

    if all(r["success"] for r in results):
        print("\n✓ 所有视频转换成功!")
        print(f"\n原始视频已备份到: {BACKUP_ROOT}")
        print("\n如需恢复原始视频，运行:")
        print(f"  rm -rf {VIDEO_ROOT}")
        print(f"  mv {BACKUP_ROOT} {VIDEO_ROOT}")

        # 验证新编码
        print("\n验证新视频编码:")
        for video_file in video_files:
            check_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "{video_file}"'
            try:
                codec = subprocess.check_output(check_cmd, shell=True, text=True).strip()
                print(f"  {video_file.name}: {codec}")
            except:
                print(f"  {video_file.name}: 验证失败")

    else:
        failed_count = sum(1 for r in results if not r["success"])
        print(f"\n⚠ {failed_count} 个文件转换失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
