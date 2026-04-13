#!/usr/bin/env python3
"""
将 lerobot__libero 数据集的视频压缩成最利于快速 decode 的形式

优化策略:
1. 使用 H.264 编码 (硬件解码支持最好，解码速度快)
2. 编码预设: ultrafast (编码速度优先) 或 fast (平衡)
3. 关键帧间隔 (GOP): 2 帧一个关键帧 (减少解码依赖)
4. CRF 18-23: 视觉质量优先
5. 线程数优化: 提高并行解码效率
6. 像素格式: yuv420p (兼容性最好)

参考: convert_videos_h264_auto.py
"""

import argparse
import subprocess
import shutil
from pathlib import Path
import time
import sys
import json
from multiprocessing import Pool, cpu_count
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(
        description="压缩 libero 数据集视频以优化解码速度"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/public/xiaoyu/lerobot__libero/videos",
        help="输入视频目录",
    )
    parser.add_argument(
        "--backup_dir",
        type=str,
        default="/public/xiaoyu/lerobot__libero/videos_backup_av1",
        help="备份目录",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="fast",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"],
        help="编码预设 (ultrafast 编码最快但文件较大)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="CRF 值 (0-51, 越小质量越好，18-23 为视觉无损范围)",
    )
    parser.add_argument(
        "--gop_size",
        type=int,
        default=1,
        help="GOP 大小 (关键帧间隔，1 = 每一个关键帧)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="并行转换的 worker 数量 (默认为 CPU 核心数)",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="跳过备份步骤",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只显示将要执行的操作，不实际转换",
    )
    return parser.parse_args()


def get_video_files(video_root: Path) -> list[Path]:
    """获取所有需要转换的视频文件"""
    video_files = []
    for ext in ["*.mp4", "*.mkv", "*.webm"]:
        video_files.extend(video_root.rglob(ext))
    return sorted(video_files)


def get_video_info(video_path: Path) -> dict:
    """获取视频信息"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,duration",
        "-show_entries", "format=size",
        "-of", "json",
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        stream = info.get("streams", [{}])[0]
        format_info = info.get("format", {})

        return {
            "codec": stream.get("codec_name", "unknown"),
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "fps": eval(stream.get("r_frame_rate", "0/1")),
            "duration": float(format_info.get("duration", 0)),
            "size_bytes": int(format_info.get("size", 0)),
        }
    except Exception as e:
        print(f"警告: 无法获取视频信息 {video_path}: {e}")
        return {}


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


def convert_video(input_path: Path, args: argparse.Namespace) -> dict:
    """转换单个视频文件"""
    start_time = time.time()

    try:
        original_size = input_path.stat().st_size
    except:
        return {"success": False, "error": "无法读取文件", "path": str(input_path)}

    # 创建临时文件
    temp_path = input_path.with_suffix('.tmp.mp4')

    # 构建 ffmpeg 命令
    # 关键优化参数:
    # - libx264: 使用 H.264 编码
    # -preset: 编码速度预设
    # -crf: 恒定质量因子
    # -g 1: GOP 大小为 1 (每 1 帧一个关键帧)
    # -threads: 线程数优化
    cmd = [
        "ffmpeg",
        "-y",  # 覆盖输出文件
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", args.preset,
        "-crf", str(args.crf),
        "-g", str(args.gop_size),  # 关键帧间隔
        "-pix_fmt", "yuv420p",
        "-threads", "4",  # 优化多线程解码
        "-movflags", "+faststart",  # 快速启动 (流式传输优化)
        str(temp_path)
    ]

    # 执行转换
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        success = True
        error_msg = ""
    except subprocess.CalledProcessError as e:
        success = False
        error_msg = e.stderr[:500] if e.stderr else str(e)

    elapsed = time.time() - start_time

    # 获取新文件大小
    if success and temp_path.exists():
        new_size = temp_path.stat().st_size

        # 替换原文件
        input_path.unlink()
        temp_path.rename(input_path)

        size_ratio = (new_size / original_size) * 100
    else:
        new_size = 0
        size_ratio = 0
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()

    return {
        "success": success,
        "path": str(input_path),
        "original_size": original_size,
        "new_size": new_size,
        "size_ratio": size_ratio,
        "time_seconds": elapsed,
        "error": error_msg,
    }


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    backup_dir = Path(args.backup_dir)

    if not input_dir.exists():
        print(f"✗ 输入目录不存在: {input_dir}")
        sys.exit(1)

    print("="*80)
    print("lerobot__libero 视频压缩工具 - 快速解码优化")
    print("="*80)
    print(f"输入目录: {input_dir}")
    print(f"备份目录: {backup_dir}")
    print(f"编码预设: {args.preset}")
    print(f"CRF 值: {args.crf}")
    print(f"GOP 大小: {args.gop_size}")
    print(f"并行 workers: {args.num_workers or cpu_count()}")
    print("="*80)

    # 查找所有视频文件
    print("\n正在扫描视频文件...")
    video_files = get_video_files(input_dir)

    if not video_files:
        print(f"✗ 未找到视频文件: {input_dir}")
        sys.exit(1)

    print(f"找到 {len(video_files)} 个视频文件")

    # 检查当前编码
    print("\n检查当前视频编码...")
    sample_info = get_video_info(video_files[0])
    print(f"示例视频: {video_files[0].name}")
    print(f"  编码: {sample_info.get('codec', 'unknown')}")
    print(f"  分辨率: {sample_info.get('width', 0)}x{sample_info.get('height', 0)}")
    print(f"  帧率: {sample_info.get('fps', 0):.2f}")

    if sample_info.get("codec") == "h264":
        print("\n⚠ 视频已经是 H.264 编码，无需转换")
        confirm = input("是否仍要继续重新编码? (y/N): ")
        if confirm.lower() != 'y':
            print("已取消")
            return

    # 计算总大小
    total_size = 0
    for f in video_files:
        try:
            total_size += f.stat().st_size
        except:
            pass

    print(f"  总大小: {total_size / (1024**3):.2f} GB")

    if args.dry_run:
        print("\n[DRY RUN] 将要转换以下文件:")
        for f in video_files[:10]:
            print(f"  - {f.relative_to(input_dir)}")
        if len(video_files) > 10:
            print(f"  ... 还有 {len(video_files) - 10} 个文件")
        print(f"\n总计: {len(video_files)} 个文件")
        return

    # 备份原始视频
    if not args.no_backup:
        print("\n" + "="*80)
        print("步骤 1/2: 备份原始视频")
        print("="*80)
        backup_videos(input_dir, backup_dir)

    # 转换视频
    print("\n" + "="*80)
    print("步骤 2/2: 转换视频编码")
    print("="*80)

    num_workers = args.num_workers or cpu_count()
    print(f"使用 {num_workers} 个并行 workers")

    # 使用多进程并行转换
    with Pool(num_workers) as pool:
        results = pool.map(partial(convert_video, args=args), video_files)

    # 统计结果
    total_original_size = sum(r.get("original_size", 0) for r in results)
    total_new_size = sum(r.get("new_size", 0) for r in results)
    total_time = sum(r.get("time_seconds", 0) for r in results)
    success_count = sum(1 for r in results if r.get("success"))
    failed_count = len(results) - success_count

    # 汇总报告
    print("\n" + "="*80)
    print("转换完成!")
    print("="*80)
    print(f"总文件数: {len(video_files)}")
    print(f"成功: {success_count}, 失败: {failed_count}")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"原始大小: {total_original_size/(1024**3):.2f} GB")
    print(f"新大小: {total_new_size/(1024**3):.2f} GB")
    if total_original_size > 0:
        print(f"大小变化: {(total_new_size/total_original_size)*100:.1f}%")

    if failed_count > 0:
        print(f"\n⚠ {failed_count} 个文件转换失败:")
        for r in results:
            if not r.get("success"):
                print(f"  - {Path(r['path']).name}: {r.get('error', 'Unknown error')[:100]}")

    if success_count == len(video_files):
        print("\n✓ 所有视频转换成功!")
        if not args.no_backup:
            print(f"\n原始视频已备份到: {backup_dir}")
            print("\n如需恢复原始视频，运行:")
            print(f"  rm -rf {input_dir}")
            print(f"  mv {backup_dir} {input_dir}")

        # 验证新编码
        print("\n验证新视频编码 (抽样检查):")
        for video_file in video_files[:5]:
            info = get_video_info(video_file)
            print(f"  {video_file.name}: {info.get('codec', 'unknown')}")


if __name__ == "__main__":
    main()
