#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成所有影片的逐字稿（2024年1月至2025年10月）
从 data 目录下所有文件夹中的 video_details 文件读取影片信息
"""

import csv
import argparse
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from youtube_transcribe import YouTubeTranscriber


# ============================================================
# 辅助函数：字幕处理
# ============================================================

def clean_repeated_lines(lines):
    """
    去除连续重复的逐字稿行
    
    Args:
        lines: 原始逐字稿行列表
        
    Returns:
        去重后的行列表
    """
    cleaned = []
    last_text = None

    for line in lines:
        m = re.match(r"\[\d+:\d+\.\d+ --> .*?\]\s*(.*)", line)
        text = m.group(1).strip() if m else line.strip()
        if text == last_text:
            continue  # 跳过连续重复
        cleaned.append(line)
        last_text = text

    return cleaned


def clean_transcript_file(path: Path):
    """
    读取逐字稿文件，去除连续重复行，并覆写原文件
    
    Args:
        path: 逐字稿文件路径
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = clean_repeated_lines(lines)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)


def get_manual_subtitles(video_url):
    cmd = ["yt-dlp", "-J", video_url]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return {}

    info = json.loads(result.stdout)
    subs = info.get("subtitles", {}) or {}

    # 只保留中文字幕
    zh_keys = [k for k in subs.keys() if k.startswith("zh")]
    return {k: subs[k] for k in zh_keys}


def download_manual_subtitle(video_url, output_stem: Path):
    """
    下载真人字幕（vtt），不下载自动字幕
    
    Args:
        video_url: YouTube 视频 URL
        output_stem: 输出文件名（不含扩展名）
    """
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-subs",
        "--no-write-auto-subs",
        "--sub-lang", "zh,zh-Hant,zh-TW",
        "--sub-format", "vtt",
        "-o", str(output_stem),
        video_url
    ]
    subprocess.run(cmd, check=True)


def vtt_to_txt(vtt_path: Path, txt_path: Path):
    """
    将 VTT 字幕转为纯文本格式（保留时间戳）
    
    Args:
        vtt_path: VTT 字幕文件路径
        txt_path: 输出的 TXT 文件路径
    """
    with open(vtt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    output = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("WEBVTT"):
            continue
        output.append(line + "\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(output)


# ============================================================
# 主类：批量转录器
# ============================================================

class AllVideoTranscriber:
    def __init__(self, data_dir, transcripts_dir="transcripts_all", 
                 start_date="2024-01-01", end_date="2025-10-31"):
        """
        初始化批量逐字稿生成器
        
        Args:
            data_dir: data目录（包含所有频道文件夹）
            transcripts_dir: 逐字稿输出目录
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        self.data_dir = Path(data_dir)
        self.transcripts_dir = Path(transcripts_dir)
        self.transcripts_dir.mkdir(exist_ok=True)
        
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")

        
        # 视频信息
        self.video_info = defaultdict(lambda: {
            'video_id': '',
            'video_title': '',
            'channel': '',
            'channel_title': '',
            'publish_date': '',
            'view_count': 0,
            'like_count': 0,
            'comment_count': 0,
            'duration': '',
            'description': '',
            'transcript_source': ''
        })
        
        # 统计
        self.stats = {
            'total_video_details_files': 0,
            'total_videos_found': 0,
            'videos_in_date_range': 0,
            'videos_with_transcript': 0,
            'videos_need_transcript': 0
        }

    def load_videos_from_sample_json(self, sample_json_path):
        """
        从筛选后的 JSON 样本文件读取影片信息
        JSON 结构需包含 videos[].url 或 videos[].video_id
        """
        print("=" * 80)
        print("步骤 1: 从筛选样本 JSON 加载影片")
        print("=" * 80)

        with open(sample_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        videos = data.get("videos", [])
        print(f"\n读取到 {len(videos):,} 个筛选后影片")

        for video in videos:
            video_id = video.get("video_id")
            url = video.get("url")

            if not video_id and url:
                # fallback：从 url 解析 video_id
                if "v=" in url:
                    video_id = url.split("v=")[-1].split("&")[0]

            if not video_id:
                continue

            # 解析发布日期（可选）
            publish_date = video.get("published_at", "")
            if publish_date:
                try:
                    video_date = datetime.fromisoformat(
                        publish_date.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    if not (self.start_date <= video_date <= self.end_date):
                        continue
                except Exception:
                    pass

            info = self.video_info[video_id]
            info["video_id"] = video_id
            info["video_title"] = video.get("title", "")
            info["channel"] = video.get("channel", "")
            info["channel_title"] = video.get("channel", "")
            info["publish_date"] = publish_date
            info["view_count"] = int(video.get("view_count", 0))
            info["like_count"] = int(video.get("like_count", 0))
            info["comment_count"] = int(video.get("comment_count", 0))
            info["description"] = video.get("description", "")
            info["url"] = video.get("url")

        print(f"\n✅ 成功加载 {len(self.video_info):,} 个影片（来自样本 JSON）")
        return len(self.video_info)
    
    def load_all_videos(self):
        """从 data 目录下所有文件夹中的 video_details 文件加载影片信息"""
        print("="*80)
        print("步骤 1: 从 data 目录加载所有影片信息")
        print("="*80)
        
        print(f"\n扫描目录: {self.data_dir}")
        print(f"日期范围: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}")
        
        # 查找所有 video_details 文件
        detail_files = list(self.data_dir.glob("**/video_details_*.json"))
        self.stats['total_video_details_files'] = len(detail_files)
        
        print(f"\n找到 {len(detail_files)} 个 video_details 文件")
        
        if len(detail_files) == 0:
            print("\n❌ 错误: 未找到任何 video_details 文件")
            print("   请确认 data 目录结构正确")
            return 0
        
        # 加载所有影片信息
        print("\n正在加载影片信息...")
        
        for detail_file in tqdm(detail_files, desc="加载 video_details"):
            try:
                with open(detail_file, 'r', encoding='utf-8') as f:
                    videos = json.load(f)
                    
                    for video in videos:
                        video_id = video.get('id', '')
                        if not video_id:
                            continue
                        
                        self.stats['total_videos_found'] += 1
                        
                        # 解析发布日期
                        snippet = video.get("snippet", {})
                        publish_date = snippet.get('publishedAt', '')
                        if not publish_date:
                            continue

                        try:
                            video_date = datetime.fromisoformat(
                                publish_date.replace("Z", "+00:00")
                            ).replace(tzinfo=None)
                        except ValueError:
                            continue
                        
                        if not (self.start_date <= video_date <= self.end_date):
                            continue
                        
                        self.stats['videos_in_date_range'] += 1
                        
                        # 提取频道信息（从文件路径）
                        channel_id = detail_file.parent.name
                        
                        # 保存影片信息
                        info = self.video_info[video_id]
                        info['video_id'] = video_id
                        info['video_title'] = snippet.get('title', '')
                        info['channel'] = channel_id
                        info['channel_title'] = snippet.get('channelTitle', '')
                        info['publish_date'] = publish_date
                        info['view_count'] = int(video.get('viewCount', 0))
                        info['like_count'] = int(video.get('likeCount', 0))
                        info['comment_count'] = int(video.get('commentCount', 0))
                        info['duration'] = video.get('duration', '')
                        info['description'] = video.get('description', '')
                        
            except Exception as e:
                print(f"\n⚠️  读取文件失败: {detail_file.name} - {e}")
                continue
        
        print(f"\n✅ 数据加载完成:")
        print(f"  扫描的 video_details 文件: {self.stats['total_video_details_files']:,}")
        print(f"  找到的总影片数: {self.stats['total_videos_found']:,}")
        print(f"  日期范围内的影片: {self.stats['videos_in_date_range']:,}")
        print(f"  唯一影片ID: {len(self.video_info):,}")
        
        return len(self.video_info)
    
    def check_existing_transcripts(self):
        """检查哪些影片已经有逐字稿"""
        print("\n" + "="*80)
        print("步骤 2: 检查已有逐字稿")
        print("="*80)
        
        print(f"\n正在扫描目录: {self.transcripts_dir}")
        
        existing_transcripts = set()
        for txt_file in self.transcripts_dir.glob("*.txt"):
            video_id = txt_file.stem
            existing_transcripts.add(video_id)
        
        self.stats['videos_with_transcript'] = len(existing_transcripts)
        
        # 标记哪些影片需要生成逐字稿
        videos_need_transcript = []
        
        for video_id, info in self.video_info.items():
            if video_id in existing_transcripts:
                info['has_transcript'] = True
            else:
                info['has_transcript'] = False
                videos_need_transcript.append(video_id)
        
        self.stats['videos_need_transcript'] = len(videos_need_transcript)
        
        print(f"\n✅ 检查完成:")
        print(f"  已有逐字稿: {self.stats['videos_with_transcript']:,} 个")
        print(f"  需要生成: {self.stats['videos_need_transcript']:,} 个")
        
        return videos_need_transcript
    
    def display_video_summary(self, top_n=20):
        """显示影片统计摘要"""
        print("\n" + "="*80)
        print("影片统计摘要")
        print("="*80)
        
        # 按观看数排序
        sorted_videos = sorted(
            self.video_info.items(),
            key=lambda x: x[1]['view_count'],
            reverse=True
        )
        
        print(f"\n【观看数最多的 {top_n} 个影片】")
        print(f"{'排名':<6} {'影片ID':<15} {'观看数':>10} {'评论数':>8} {'逐字稿':>8} {'频道':<20} {'标题':<40}")
        print("-" * 130)
        
        for i, (video_id, info) in enumerate(sorted_videos[:top_n], 1):
            has_transcript = "✓" if info.get('has_transcript', False) else "✗"
            channel = info['channel_title'][:18] if len(info['channel_title']) > 18 else info['channel_title']
            title = info['video_title'][:38] if len(info['video_title']) > 38 else info['video_title']
            
            print(f"{i:<6} {video_id:<15} {info['view_count']:>10,} "
                  f"{info['comment_count']:>8,} {has_transcript:>8} {channel:<20} {title:<40}")
        
        # 按日期分组统计
        print(f"\n【按月份统计影片数】")
        monthly_counts = defaultdict(int)
        for video_id, info in self.video_info.items():
            publish_date = info['publish_date']
            if publish_date:
                month = publish_date[:7]  # YYYY-MM
                monthly_counts[month] += 1
        
        print(f"{'月份':<10} {'影片数':>10}")
        print("-" * 25)
        for month in sorted(monthly_counts.keys()):
            print(f"{month:<10} {monthly_counts[month]:>10,}")
        
        # 按频道统计
        print(f"\n【按频道统计影片数】")
        channel_counts = defaultdict(int)
        channel_names = {}
        for video_id, info in self.video_info.items():
            channel = info['channel']
            channel_counts[channel] += 1
            channel_names[channel] = info['channel_title']
        
        sorted_channels = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'频道ID':<30} {'频道名称':<30} {'影片数':>10}")
        print("-" * 75)
        for channel_id, count in sorted_channels[:20]:
            channel_name = channel_names.get(channel_id, '')[:28]
            print(f"{channel_id:<30} {channel_name:<30} {count:>10,}")
    
    def batch_transcribe(self, model_size="base", max_videos=None, start_from=0, 
                        sort_by="view_count"):
        """
        批量生成逐字稿
        
        Args:
            model_size: Whisper模型大小 (tiny, base, small, medium, large)
            max_videos: 最多处理的影片数量 (None = 全部)
            start_from: 从第几个影片开始 (用于断点续传)
            sort_by: 排序方式 (view_count, comment_count, publish_date)
        """
        print("\n" + "="*80)
        print("步骤 3: 批量生成逐字稿")
        print("="*80)
        
        # 获取需要处理的影片列表
        videos_to_process = [
            (vid, info) for vid, info in self.video_info.items()
            if not info.get('has_transcript', False)
        ]
        
        # 按指定方式排序
        if sort_by == "view_count":
            videos_to_process.sort(key=lambda x: x[1]['view_count'], reverse=True)
            print(f"\n排序方式: 按观看数（优先处理高观看数影片）")
        elif sort_by == "comment_count":
            videos_to_process.sort(key=lambda x: x[1]['comment_count'], reverse=True)
            print(f"\n排序方式: 按评论数（优先处理高评论数影片）")
        elif sort_by == "publish_date":
            videos_to_process.sort(key=lambda x: x[1]['publish_date'], reverse=False)
            print(f"\n排序方式: 按发布日期（从旧到新）")
        
        # 应用起始位置和数量限制
        if start_from > 0:
            videos_to_process = videos_to_process[start_from:]
        
        if max_videos is not None:
            videos_to_process = videos_to_process[:max_videos]
        
        total = len(videos_to_process)
        
        if total == 0:
            print("\n没有需要处理的影片！")
            return
        
        print(f"\n⚙️  转录设置:")
        print(f"   模型大小: {model_size}")
        print(f"   待处理影片: {total}")
        if start_from > 0:
            print(f"   起始位置: 第 {start_from + 1} 个")
        
        # 创建转录器
        print(f"\n正在初始化 Whisper 模型 ({model_size})...")
        transcriber = YouTubeTranscriber(
            model_size=model_size,
            output_dir=str(self.transcripts_dir)
        )
        
        # 开始批量处理
        print(f"\n开始批量转录 {total} 个影片...")
        print("=" * 80)
        
        success_count = 0
        error_count = 0
        skip_count = 0
        
        for i, (video_id, info) in enumerate(videos_to_process, 1):
            actual_index = start_from + i
            video_title = info['video_title'] or 'Unknown'
            channel = info['channel_title']
            view_count = info['view_count']
            comment_count = info['comment_count']
            publish_date = info['publish_date'][:10] if info['publish_date'] else 'Unknown'
            
            print(f"\n[{i}/{total}] (总第 {actual_index}) 影片ID: {video_id}")
            print(f"标题: {video_title}")
            print(f"频道: {channel}")
            print(f"发布日期: {publish_date}")
            print(f"观看数: {view_count:,} | 评论数: {comment_count:,}")
            print("-" * 80)
            
            # 构建YouTube URL
            video_url = info.get("url") or f"https://www.youtube.com/watch?v={video_id}"
            
            try:
                # 再次检查是否已存在（避免重复处理）
                output_file = self.transcripts_dir / f"{video_id}.txt"
                if output_file.exists():
                    print(f"⊙ 跳过（已存在逐字稿）")
                    skip_count += 1
                    continue
                
                # 1️⃣ 先检查真人字幕
                manual_subs = get_manual_subtitles(video_url)

                if manual_subs:
                    print("✓ 发现真人字幕，使用原始字幕")
                    info['transcript_source'] = 'manual'
                    
                    # 下载字幕
                    download_manual_subtitle(video_url, self.transcripts_dir / video_id)
                    
                    # 查找下载的 VTT 文件
                    vtt_files = sorted(
                        self.transcripts_dir.glob(f"{video_id}.zh*.vtt"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True
                    )
                    if not vtt_files:
                        raise RuntimeError("中文字幕字幕下载失败")
                    
                    # 转为 TXT
                    vtt_to_txt(vtt_files[0], output_file)
                    
                    # 去重
                    clean_transcript_file(output_file)
                    
                    success_count += 1
                    continue

                # 2️⃣ 没有真人字幕 → Whisper fallback
                print("⊙ 无真人字幕，使用 Whisper")
                transcriber.process_video(video_url, language="zh", keep_audio=False)
                if not output_file.exists():
                    raise RuntimeError("Whisper 转录失败，未生成逐字稿")
                info['transcript_source'] = 'whisper'
                clean_transcript_file(output_file)
                success_count += 1
                
            except KeyboardInterrupt:
                print("\n\n⚠️  用户中断批量处理")
                print(f"\n已处理统计:")
                print(f"  成功: {success_count}")
                print(f"  失败: {error_count}")
                print(f"  跳过: {skip_count}")
                print(f"\n提示: 可以使用 start_from={actual_index} 继续处理")
                raise
                
            except Exception as e:
                print(f"✗ 处理失败: {str(e)}")
                error_count += 1
            
            print("=" * 80)
        
        # 最终统计
        print(f"\n" + "="*80)
        print("批量转录完成！")
        print("="*80)
        print(f"\n处理结果:")
        print(f"  总数: {total}")
        print(f"  成功: {success_count}")
        print(f"  失败: {error_count}")
        print(f"  跳过: {skip_count}")
        
        if error_count > 0:
            print(f"\n⚠️  {error_count} 个影片处理失败")
            print(f"  可能原因: 影片已删除、私人影片、地区限制等")
    
    def save_video_list(self, output_file):
        """保存影片列表（含逐字稿状态）"""
        print(f"\n正在保存影片列表到: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
            fieldnames = [
                'video_id',
                'video_title',
                'channel_id',
                'channel_title',
                'publish_date',
                'view_count',
                'like_count',
                'comment_count',
                'duration',
                'has_transcript',
                'youtube_url',
                'transcript_source',
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # 按发布日期排序
            sorted_videos = sorted(
                self.video_info.items(),
                key=lambda x: x[1]['publish_date'],
                reverse=False
            )
            
            for video_id, info in sorted_videos:
                writer.writerow({
                    'video_id': video_id,
                    'video_title': info['video_title'],
                    'channel_id': info['channel'],
                    'channel_title': info['channel_title'],
                    'publish_date': info['publish_date'][:10] if info['publish_date'] else '',
                    'view_count': info['view_count'],
                    'like_count': info['like_count'],
                    'comment_count': info['comment_count'],
                    'duration': info['duration'],
                    'has_transcript': 'Yes' if info.get('has_transcript', False) else 'No',
                    'youtube_url': f"https://www.youtube.com/watch?v={video_id}",
                    'transcript_source': info.get('transcript_source', '')
                })
        
        print(f"✅ 已保存 {len(self.video_info)} 个影片的信息")
        return output_file


def main():
    """主函数"""
    print("\n" + "="*80)
    print("所有影片批量转录工具 (2024-01 至 2025-10)")
    print("="*80)
    
    # 设置参数
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()
    data_dir = './data'
    transcripts_dir = './transcripts_all'
    # sample_json = "./中天新聞_videos.json"
    sample_json = args.filename
    start_date = "2024-01-01"  # 开始日期
    end_date = "2025-10-31"    # 结束日期
    
    # ========================================
    # 🔧 转录参数设置
    # ========================================
    WHISPER_MODEL = "medium"   # 模型选择: tiny, base, small, medium, large
                             # tiny: 最快，准确度较低
                             # base: 快速，准确度中等（推荐）
                             # small/medium/large: 越来越慢，但准确度更高
    
    MAX_VIDEOS = None          # 一次最多处理多少个影片 (None = 全部)
                             # 建议先设为 10 测试，确认无误后设为 None
    
    START_FROM = 0           # 从第几个影片开始 (0 = 从头开始)
                             # 如果中断了，可以设置这个参数继续处理
    
    SORT_BY = "view_count"   # 排序方式: view_count, comment_count, publish_date
                             # view_count: 优先处理高观看数影片
                             # comment_count: 优先处理高评论数影片
                             # publish_date: 按时间顺序处理
    
    print(f"\n⚙️  参数设置:")
    print(f"   数据目录: {data_dir}")
    print(f"   输出目录: {transcripts_dir}")
    print(f"   日期范围: {start_date} 至 {end_date}")
    print(f"   Whisper模型: {WHISPER_MODEL}")
    print(f"   最大处理数: {MAX_VIDEOS if MAX_VIDEOS else '全部'}")
    print(f"   起始位置: 第 {START_FROM + 1} 个")
    print(f"   排序方式: {SORT_BY}")
    
    # 创建转录器
    processor = AllVideoTranscriber(data_dir, transcripts_dir, start_date, end_date)
    
    # 执行流程
    print("\n" + "="*80)
    print("开始处理流程")
    print("="*80)
    
    # 步骤1: 加载影片列表
    num_videos = processor.load_videos_from_sample_json(sample_json)
    
    if num_videos == 0:
        print("\n❌ 未找到任何影片，程序退出")
        return None
    
    # 步骤2: 检查已有逐字稿
    videos_need_transcript = processor.check_existing_transcripts()
    
    # 步骤3: 显示统计摘要
    processor.display_video_summary(top_n=30)
    
    # 步骤4: 保存影片列表
    video_list_file = processor.save_video_list(
        './all_videos_list_2024-2025.csv'
    )
    
    print(f"\n已保存影片列表: {video_list_file}")
    
    # 步骤5: 询问是否开始转录
    print("\n" + "="*80)
    print("准备开始批量转录")
    print("="*80)
    
    print(f"\n将要处理 {processor.stats['videos_need_transcript']} 个影片")
    print(f"使用模型: {WHISPER_MODEL}")
    print(f"排序方式: {SORT_BY}")
    
    if MAX_VIDEOS:
        print(f"本次最多处理: {MAX_VIDEOS} 个")
    
    # response = input("\n是否开始转录? (y/N): ").strip().lower()
    
    # if response == 'y':
    if True:
        try:
            processor.batch_transcribe(
                model_size=WHISPER_MODEL,
                max_videos=MAX_VIDEOS,
                start_from=START_FROM,
                sort_by=SORT_BY
            )
            
            print("\n" + "="*80)
            print("✅ 全部完成！")
            print("="*80)
            
            print(f"\n逐字稿已保存到: {transcripts_dir}")
            print(f"接下来可以使用这些逐字稿进行分析！")
            
        except KeyboardInterrupt:
            print("\n\n程序已中断")
        except Exception as e:
            print(f"\n✗ 发生错误: {str(e)}")
    else:
        print("\n已取消转录")
        print(f"影片列表已保存到: {video_list_file}")
        print("需要时可以重新运行此脚本")
    
    return processor


if __name__ == "__main__":
    try:
        processor = main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作，程序已安全退出")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

