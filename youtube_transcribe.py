#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube 影片转逐字稿工具
使用 OpenAI Whisper 将 YouTube 影片转换为文字稿

安装依赖:
    pip install openai-whisper yt-dlp

使用方式:
    1. 处理单个视频: python youtube_transcribe.py
    2. 批量处理: 修改 main() 函数中的设置
"""

import json
import os
from pathlib import Path
import whisper
from faster_whisper import WhisperModel, BatchedInferencePipeline
import yt_dlp
import torch


class YouTubeTranscriber:
    def __init__(self, model_size="base", output_dir="transcripts"):
        """
        初始化转录器
        
        Args:
            model_size: Whisper 模型大小 (tiny, base, small, medium, large)
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载 Whisper 模型
        print(f"正在加载 Whisper 模型: {model_size}")
        # self.model = whisper.load_model(model_size)
        simple_model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float32")
        self.model = BatchedInferencePipeline(model=simple_model)
        print("模型加载完成!")
    
    def download_audio(self, video_url, output_path="temp_audio.mp3"):
        """
        从 YouTube 下载音频
        
        Args:
            video_url: YouTube 影片 URL
            output_path: 临时音频文件路径
            
        Returns:
            音频文件路径, 视频标题, 视频ID
        """
        print(f"正在下载音频: {video_url}")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_path.replace('.mp3', ''),
            'quiet': True,  # 隐藏大部分输出
            'no_warnings': True,  # 隐藏警告
            'ignoreerrors': False,
            'extractor_args': {'youtube': {'skip': ['dash', 'hls']}},  # 跳过有问题的格式
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
            'cookiesfrombrowser': ('chromium',),
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_title = info.get('title', 'Unknown')
            video_id = info.get('id', 'Unknown')
            
        print(f"音频下载完成: {video_title}")
        return output_path, video_title, video_id
    
    def transcribe_audio(self, audio_path, language="zh", use_traditional=True):
        """
        使用 Whisper 转录音频
        
        Args:
            audio_path: 音频文件路径
            language: 语言代码 (zh=中文, en=英文)
            use_traditional: 是否使用繁体中文
            
        Returns:
            转录结果
        """
        print(f"正在转录音频: {audio_path}")
        
        # 根据需求设置 prompt（引导使用繁体中文）
        initial_prompt = None
        if language == "zh" and use_traditional:
            initial_prompt = (
                "以下是正體中文（繁體中文）的內容。"
                "請使用台灣常用的繁體中文字詞，例如：台灣、國際、政治、經濟、"
                "總統、立法院、行政院、民進黨、國民黨、時事、新聞。"
            )
        
        result, info = self.model.transcribe(
            audio_path, 
            beam_size=5,
            language=language, 
            initial_prompt=initial_prompt,
            batch_size=8,
        )
        print("转录完成!")
        return result
    
    def save_transcript(self, result, video_id, video_title):
        """
        保存转录结果 - 只保存纯文字版本
        
        Args:
            result: Whisper 转录结果
            video_id: 影片 ID
            video_title: 影片标题（不使用）
        """
        import re
        
        # 使用 segments 来保留原始的换行结构
        # 从 segments 构建文本，保留原始的分段
        lines = []
        prev_line = None

        for segment in result:
            text = segment.text.strip()
            if text:
                # 检查是否与前一行重复
                if prev_line is None or not self._is_duplicate_line(text, prev_line):
                    lines.append(text)
                    prev_line = text

        text = '\n'.join(lines)
        
        # 移除 "字幕由 Amara.org 社群提供" 等重复标注
        text = self._clean_subtitle_markers(text)
        
        # 再次去除可能因为清理后产生的连续重复行
        text = self._remove_consecutive_duplicates(text)
        
        # 移除多余的空行（超过2个连续换行的压缩成2个）
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 使用 video_id 作为档名
        txt_path = self.output_dir / f"{video_id}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text.strip())
        print(f"纯文字稿已保存: {txt_path}")
    
    def _remove_duplicates(self, text):
        """
        移除文本中的重复句子和无用字幕标注
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        import re
        
        # 先移除常见的字幕标注和无用文字
        text = self._remove_subtitle_markers(text)
        
        # 按句子分割（中文使用句号、问号、感叹号等）
        sentences = re.split(r'([。！？\n])', text)
        
        # 重组句子（保留分隔符）
        full_sentences = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i+1])
            else:
                full_sentences.append(sentences[i])
        
        # 去除连续重复的句子
        cleaned = []
        prev_sentence = None
        
        for sentence in full_sentences:
            # 清理空白
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查是否与前一句相同或高度相似
            if prev_sentence is None or not self._is_similar(sentence, prev_sentence):
                cleaned.append(sentence)
                prev_sentence = sentence
        
        return ''.join(cleaned)
    
    def _clean_subtitle_markers(self, text):
        """
        移除常见的字幕标注和无用文字（用于保存时清理）
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        import re
        
        # 定义要移除的模式
        patterns = [
            r'字幕由\s*Amara\.org\s*社群提供',
            r'字幕由.*?社群提供',
            r'翻译.*?校对.*',
            r'字幕制作.*',
            r'Subtitles by.*',
            r'Translated by.*',
            r'請訂閱.*頻道',
            r'喜歡.*請按讚',
            r'記得開啟小鈴鐺',
            r'\[音樂\]',
            r'\[掌聲\]',
            r'\[笑聲\]',
            r'\(音樂\)',
            r'\(掌聲\)',
            r'\(笑聲\)',
        ]
        
        # 逐个移除匹配的模式
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _is_duplicate_line(self, line1, line2, threshold=0.9):
        """
        检查两行是否重复（用于实时去重）
        
        Args:
            line1, line2: 要比较的行
            threshold: 相似度阈值（0-1）
            
        Returns:
            是否重复
        """
        # 完全相同
        if line1 == line2:
            return True
        
        # 移除标点和空白后比较
        clean1 = ''.join(c for c in line1 if c.isalnum())
        clean2 = ''.join(c for c in line2 if c.isalnum())
        
        if not clean1 or not clean2:
            return False
        
        # 完全相同
        if clean1 == clean2:
            return True
        
        # 检查是否一个是另一个的子串（可能是部分重复）
        if len(clean1) < len(clean2):
            shorter, longer = clean1, clean2
        else:
            shorter, longer = clean2, clean1
        
        # 如果短的在长的里面，且占比超过阈值
        if shorter in longer and len(shorter) / len(longer) > threshold:
            return True
        
        return False
    
    def _remove_consecutive_duplicates(self, text):
        """
        移除文本中连续重复的行
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        lines = text.split('\n')
        cleaned = []
        prev_line = None
        
        for line in lines:
            line = line.strip()
            if not line:
                # 保留空行
                if prev_line != '':
                    cleaned.append('')
                    prev_line = ''
                continue
            
            # 检查是否与前一行重复
            if prev_line is None or not self._is_duplicate_line(line, prev_line):
                cleaned.append(line)
                prev_line = line
        
        return '\n'.join(cleaned)
    
    def _remove_subtitle_markers(self, text):
        """
        移除常见的字幕标注和无用文字
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        import re
        
        # 定义要移除的模式
        patterns = [
            r'字幕由\s*Amara\.org\s*社群提供',
            r'字幕由.*?社群提供',
            r'翻译.*?校对.*',
            r'字幕制作.*',
            r'Subtitles by.*',
            r'Translated by.*',
            r'請訂閱.*頻道',
            r'喜歡.*請按讚',
            r'記得開啟小鈴鐺',
            r'\[音樂\]',
            r'\[掌聲\]',
            r'\[笑聲\]',
            r'\(音樂\)',
            r'\(掌聲\)',
            r'\(笑聲\)',
        ]
        
        # 逐个移除匹配的模式
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 移除多余的空行
        text = re.sub(r'\n\s*\n+', '\n', text)
        
        return text.strip()
    
    def _is_similar(self, s1, s2, threshold=0.85):
        """
        检查两个句子是否相似（用于去重）
        
        Args:
            s1, s2: 要比较的句子
            threshold: 相似度阈值（0-1）
            
        Returns:
            是否相似
        """
        # 移除标点和空白
        s1_clean = ''.join(c for c in s1 if c.isalnum())
        s2_clean = ''.join(c for c in s2 if c.isalnum())
        
        if not s1_clean or not s2_clean:
            return False
        
        # 计算相似度（使用简单的字符匹配）
        if len(s1_clean) != len(s2_clean):
            return False
        
        matches = sum(c1 == c2 for c1, c2 in zip(s1_clean, s2_clean))
        similarity = matches / len(s1_clean)
        
        return similarity >= threshold
    
    def process_video(self, video_url, language="zh", keep_audio=False):
        """
        处理单个视频的完整流程
        
        Args:
            video_url: YouTube 影片 URL
            language: 语言代码
            keep_audio: 是否保留音频文件
            
        Returns:
            转录结果
        """
        audio_path = None
        try:
            # 下载音频
            audio_path, video_title, video_id = self.download_audio(video_url)
            
            # 转录
            result = self.transcribe_audio(audio_path, language)
            
            # 保存结果
            self.save_transcript(result, video_id, video_title)
            
            # 清理临时文件
            if not keep_audio and os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"已删除临时音频文件: {audio_path}")
            
            return result
            
        except KeyboardInterrupt:
            print("\n\n检测到用户中断，正在清理临时文件...")
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"已删除临时音频文件: {audio_path}")
            raise
        except Exception as e:
            print(f"处理视频时出错: {str(e)}")
            # 出错时也清理临时文件
            if audio_path and not keep_audio and os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"已删除临时音频文件: {audio_path}")
            raise
    
    def process_json_file(self, json_path, max_videos=None, language="zh"):
        """
        从 JSON 文件批量处理视频
        
        Args:
            json_path: JSON 文件路径
            max_videos: 最多处理的视频数量（None = 全部）
            language: 语言代码
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        videos = data.get('videos', [])
        total = len(videos) if max_videos is None else min(max_videos, len(videos))
        
        print(f"\n将处理 {total} 个视频")
        print("=" * 50)
        
        try:
            for i, video in enumerate(videos[:total], 1):
                video_url = video.get('url')
                video_title = video.get('title', 'Unknown')
                video_id = video.get('video_id')  # 尝试从 JSON 获取
                
                # 如果 JSON 没有 video_id，从 URL 提取
                if not video_id and video_url:
                    import re
                    match = re.search(r'(?:v=|\/)([a-zA-Z0-9_-]{11})', video_url)
                    video_id = match.group(1) if match else None
                
                # 检查是否已处理（跳过已存在的文件）
                if video_id:
                    output_file = self.output_dir / f"{video_id}.txt"
                    if output_file.exists():
                        print(f"\n[{i}/{total}] 跳过（已处理）: {video_title}")
                        continue
                
                print(f"\n[{i}/{total}] 处理: {video_title}")
                print(f"URL: {video_url}")
                print("-" * 50)
                
                try:
                    self.process_video(video_url, language)
                    print(f"✓ 完成: {video_title}")
                except KeyboardInterrupt:
                    print("\n\n批量处理被用户中断")
                    raise
                except Exception as e:
                    print(f"✗ 失败: {video_title}")
                    print(f"错误: {str(e)}")
                
                print("=" * 50)
        except KeyboardInterrupt:
            print("\n批量处理已停止")
            raise


def interactive_mode():
    """互动模式 - 引导用户一步步操作"""
    print("=" * 60)
    print("YouTube 影片转逐字稿工具 - 互动模式")
    print("=" * 60)
    
    # 选择模型大小
    print("\n请选择 Whisper 模型大小:")
    print("1. tiny   - 最快，准确度较低（测试推荐）")
    print("2. base   - 快速，准确度中等（日常推荐）")
    print("3. small  - 中等速度，准确度好")
    print("4. medium - 较慢，准确度很好")
    print("5. large  - 最慢，准确度最高")
    
    model_choice = input("\n请输入选择 (1-5) [默认: 2]: ").strip() or "2"
    model_map = {
        "1": "tiny",
        "2": "base",
        "3": "small",
        "4": "medium",
        "5": "large"
    }
    model_size = model_map.get(model_choice, "base")
    
    # 选择处理模式
    print("\n请选择处理模式:")
    print("1. 处理单个视频")
    print("2. 从 JSON 文件批量处理")
    
    mode = input("\n请输入选择 (1-2) [默认: 1]: ").strip() or "1"
    
    # 创建转录器
    print(f"\n初始化转录器（模型: {model_size}）...")
    transcriber = YouTubeTranscriber(model_size=model_size, output_dir="transcripts")
    
    if mode == "1":
        # 单个视频模式
        video_url = input("\n请输入 YouTube 视频 URL: ").strip()
        
        if not video_url:
            print("错误: 未输入视频 URL")
            return
        
        # 选择语言
        print("\n请选择语言:")
        print("1. 中文 (zh)")
        print("2. 英文 (en)")
        print("3. 自动检测")
        
        lang_choice = input("\n请输入选择 (1-3) [默认: 1]: ").strip() or "1"
        lang_map = {
            "1": "zh",
            "2": "en",
            "3": None
        }
        language = lang_map.get(lang_choice, "zh")
        
        # 是否保留音频
        keep_audio = input("\n是否保留下载的音频文件? (y/N): ").strip().lower() == 'y'
        
        print("\n" + "=" * 60)
        print("开始处理...")
        print("=" * 60)
        
        try:
            transcriber.process_video(video_url, language=language, keep_audio=keep_audio)
            print("\n✓ 处理完成!")
            print(f"输出文件已保存到 transcripts/ 目录")
        except Exception as e:
            print(f"\n✗ 处理失败: {str(e)}")
    
    elif mode == "2":
        # 批量处理模式
        json_file = "political_videos_keywords.json"
        
        if not os.path.exists(json_file):
            print(f"\n错误: 找不到文件 {json_file}")
            return
        
        # 选择处理数量
        max_videos = input("\n请输入要处理的视频数量 [默认: 3，输入 'all' 处理全部]: ").strip()
        
        if max_videos.lower() == 'all':
            max_videos = None
        else:
            try:
                max_videos = int(max_videos) if max_videos else 3
            except ValueError:
                max_videos = 3
        
        # 选择语言
        print("\n请选择语言:")
        print("1. 中文 (zh)")
        print("2. 英文 (en)")
        print("3. 自动检测")
        
        lang_choice = input("\n请输入选择 (1-3) [默认: 1]: ").strip() or "1"
        lang_map = {
            "1": "zh",
            "2": "en",
            "3": None
        }
        language = lang_map.get(lang_choice, "zh")
        
        print("\n" + "=" * 60)
        print("开始批量处理...")
        print("=" * 60)
        
        try:
            transcriber.process_json_file(json_file, max_videos=max_videos, language=language)
            print("\n✓ 所有视频处理完成!")
            print(f"输出文件已保存到 transcripts/ 目录")
        except Exception as e:
            print(f"\n✗ 处理失败: {str(e)}")
    
    else:
        print("\n错误: 无效的选择")


def main():
    """主函数 - 可以选择直接模式或互动模式"""
    
    # ========================================
    # 方式 1: 互动模式（推荐新手）
    # ========================================
    # interactive_mode()
    
    # ========================================
    # 方式 2: 直接模式（批量处理 JSON）
    # ========================================
    # 创建转录器（可选模型: tiny, base, small, medium, large）
    transcriber = YouTubeTranscriber(model_size="large", output_dir="transcripts")
    
    # 批量处理 JSON 文件
    json_file = "political_videos_keywords.json"
    if os.path.exists(json_file):
        transcriber.process_json_file(
            json_file,
            max_videos=None,  # 处理前10个，设为 None 处理全部 592 个
            language="zh"
        )
    else:
        print(f"找不到文件: {json_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作，程序已安全退出")
        # 清理任何残留的临时音频文件
        temp_files = ["temp_audio.mp3", "temp_audio"]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"已清理临时文件: {temp_file}")
                except Exception:
                    pass
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
