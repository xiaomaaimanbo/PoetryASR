import gradio as gr
import os
import sys
import tempfile
import base64
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import TranscriptionPipeline
from src.audio_processor import AudioProcessor

MODEL_MAP = {
    "tiny": "tiny (最快, 低准确率)",
    "base": "base (平衡)",
    "small": "small",
    "medium": "medium (推荐)",
    "large-v3": "large-v3 (最准, 需GPU)",
}
LANG_MAP = {"zh": "中文", "en": "English", "auto": "自动检测"}

pipeline = None
processor = None


def _get_pipeline(model_name, language):
    global pipeline
    real_lang = None if language == "auto" else "zh"
    if pipeline is None:
        pipeline = TranscriptionPipeline(model_name=model_name, language=real_lang or "zh")
    elif pipeline.model_name != model_name or pipeline.language != real_lang:
        pipeline.reset_model(model_name=model_name, language=real_lang or "zh")
    return pipeline


def _get_processor():
    global processor
    if processor is None:
        processor = AudioProcessor()
    return processor


def _load_banner():
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    if not os.path.isdir(assets_dir):
        return None
    for fname in os.listdir(assets_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            p = os.path.join(assets_dir, fname)
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None


BANNER_B64 = _load_banner()
CREDIT_TEXT = "此项目由 nchku-1524-小马爱曼波 独立用 AI 开发"


def transcribe_file(audio_file, model_name, language, enable_poetry):
    if audio_file is None:
        return "", "", "", "请先上传音频文件"

    try:
        pl = _get_pipeline(model_name, language)
        pl.enable_poetry = enable_poetry
        result = pl.run(audio_file)

        segment_text = _format_segments(result.segments)
        poetry_html = _render_poetry_html(result) if enable_poetry else ""

        return result.text, segment_text, poetry_html, _build_summary(result)
    except Exception as e:
        error_msg = f"转录失败: {e}"
        return "", "", "", error_msg


def transcribe_microphone(audio, model_name, language, enable_poetry):
    if audio is None:
        return "", "", "", "请先录制音频"

    sample_rate, audio_data = audio
    proc = _get_processor()
    audio_data = proc.normalize_audio(audio_data.astype("float32"))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    proc.save_audio(tmp_path, audio_data, sample_rate)

    try:
        pl = _get_pipeline(model_name, language)
        pl.enable_poetry = enable_poetry
        result = pl.run(tmp_path)

        segment_text = _format_segments(result.segments)
        poetry_html = _render_poetry_html(result) if enable_poetry else ""

        return result.text, segment_text, poetry_html, _build_summary(result)
    except Exception as e:
        error_msg = f"转录失败: {e}"
        return "", "", "", error_msg
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _render_poetry_html(result):
    if not result.matched_poetry:
        raw_display = result.text.strip() if result.text.strip() else "（识别结果为空）"
        return f"""<div style="padding:12px; background:#fff7ed; border:1px solid #f97316; border-radius:8px;">
<h3 style="color:#ea580c; margin:0 0 8px 0;">未匹配到已知古诗词</h3>
<p style="color:#92400e; margin:0 0 8px 0;">该音频可能不是古诗词，或诗词不在当前数据库中。</p>
<p style="color:#92400e; margin:0 0 8px 0;">可在「诗词库」标签页中添加新诗词后重新转录。</p>
<div style="margin-top:8px; padding:8px; background:#ffffff; border-radius:4px;">
  <strong>Whisper 原始识别文本:</strong><br/>
  <span style="font-size:16px;color:#1e40af;">{raw_display}</span>
</div></div>"""

    aligned = result.aligned_segments
    html = f"""<div style="padding: 12px; background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px;">
<h3 style="color:#166534; margin:0 0 8px 0;"> 《{result.matched_poetry}》</h3>
<p style="color:#4ade80; margin:0 0 12px 0;">作者: {result.author} | 相似度: {result.similarity:.1%}</p>
"""

    if aligned:
        html += '<table style="width:100%; border-collapse:collapse;">'
        html += '<tr style="background:#dcfce7;"><th style="padding:6px;text-align:left;">时间</th><th style="padding:6px;text-align:left;">诗句</th></tr>'
        for seg in aligned:
            start = seg["start"]
            end = seg["end"]
            txt = seg["text"]
            matched = seg.get("is_matched", False)
            score = seg.get("match_score", 0)
            color = "#166534" if matched else "#9ca3af"
            icon = "[match]" if matched else "[?]"
            html += f'<tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:6px;color:#6b7280;">{start:.1f}s - {end:.1f}s</td><td style="padding:6px;color:{color};">{icon} {txt} ({score:.0%})</td></tr>'
        html += "</table>"

    html += f"""
<div style="margin-top:12px; padding:8px; background:#ffffff; border-radius:4px;">
  <strong>纠正后文本:</strong><br/>
  <span style="font-size:16px;color:#1e40af;">{result.corrected}</span>
</div></div>"""
    return html


def _format_segments(segments):
    rows = []
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        txt = seg.get("text", "")
        rows.append(f"[{start:6.1f}s - {end:6.1f}s]  {txt}")
    return "\n".join(rows)


def _build_summary(result):
    word_count = len(result.text.replace(" ", ""))
    seg_count = len(result.segments)
    return f"模型: {result.model_used} | 字数: {word_count} | 分段: {seg_count} | 音频时长: {result.duration:.1f}s"


def list_poetry():
    pl = _get_pipeline("medium", "zh")
    titles = pl.get_poetry_titles()
    if not titles:
        return "诗词库为空"
    rows = []
    for t in titles:
        info = pl.corrector.get_poetry_info(t)
        author = info.get("author", "")
        dynasty = info.get("dynasty", "")
        content = info.get("content", "")
        preview = content[:50] + "..." if len(content) > 50 else content
        rows.append(f"**《{t}》** — {dynasty} · {author}\n> {preview}")
    return "\n\n".join(rows)


def add_poem_handler(title, author, dynasty, content):
    if not title.strip() or not content.strip():
        return "[!] 标题和内容不能为空", list_poetry()
    pl = _get_pipeline("medium", "zh")
    ok = pl.add_poem(title.strip(), content.strip(), author.strip(), dynasty.strip())
    if ok:
        return f"[OK] 成功添加《{title}》", list_poetry()
    return f"[!] 添加失败", list_poetry()


def _check_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


with gr.Blocks(title="古诗词语音识别系统") as demo:
    banner_html = ""
    if BANNER_B64:
        banner_html = f'<div style="text-align:center; margin-bottom:16px;"><img src="data:image/jpeg;base64,{BANNER_B64}" style="max-width:100%; max-height:320px; border-radius:12px; box-shadow:0 4px 16px rgba(0,0,0,0.15);" alt="Banner"></div>'

    cuda_available = _check_cuda()
    header_md = """
    # 古诗词语音识别与时间对齐系统
    ### 基于 OpenAI Whisper · 同音字纠错 · 时间戳对齐

    支持上传音频文件或直接录音，自动识别古诗词朗读内容并纠正同音字错误。
    """
    if not cuda_available:
        header_md += "\n> [!] 未检测到 CUDA GPU，将使用 CPU 推理（速度较慢）"
    gr.Markdown(header_md)

    gr.HTML(f"{banner_html}")

    with gr.Tabs():
        with gr.Tab("语音识别"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 输入")
                    audio_input = gr.Audio(
                        label="上传音频文件",
                        type="filepath",
                        sources=["upload"],
                    )
                    mic_input = gr.Audio(
                        label="或直接录音",
                        type="numpy",
                        sources=["microphone"],
                    )
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=list(MODEL_MAP.keys()),
                            value="medium",
                            label="Whisper 模型",
                            info="越大越准, 越慢",
                        )
                        lang_dropdown = gr.Dropdown(
                            choices=list(LANG_MAP.keys()),
                            value="zh",
                            label="语言",
                        )
                    poetry_toggle = gr.Checkbox(
                        value=True,
                        label="启用古诗词纠错与时间对齐",
                    )
                    with gr.Row():
                        submit_btn = gr.Button("转录音频文件", variant="primary")
                        mic_btn = gr.Button("转录录音", variant="secondary")

                with gr.Column(scale=2):
                    gr.Markdown("### 输出")
                    text_output = gr.Textbox(label="识别文本", lines=3, interactive=False)
                    segments_output = gr.Textbox(label="时间戳分段", lines=8, interactive=False)
                    poetry_output = gr.HTML(label="诗词匹配结果")
                    summary_output = gr.Textbox(label="统计摘要", interactive=False)

            submit_btn.click(
                fn=transcribe_file,
                inputs=[audio_input, model_dropdown, lang_dropdown, poetry_toggle],
                outputs=[text_output, segments_output, poetry_output, summary_output],
            )
            mic_btn.click(
                fn=transcribe_microphone,
                inputs=[mic_input, model_dropdown, lang_dropdown, poetry_toggle],
                outputs=[text_output, segments_output, poetry_output, summary_output],
            )

        with gr.Tab("诗词库"):
            gr.Markdown("### 当前诗词数据库")
            poetry_list = gr.Markdown(value=list_poetry())
            with gr.Row():
                refresh_btn = gr.Button("刷新列表")
            gr.Markdown("---")
            gr.Markdown("### 添加新诗词")
            gr.Markdown("如果识别的诗词不在库中，输入后点击添加，再重新转录即可正确识别和纠错。")
            with gr.Row():
                poem_title = gr.Textbox(label="诗词标题", placeholder="如：将进酒")
                poem_author = gr.Textbox(label="作者", placeholder="如：李白")
                poem_dynasty = gr.Textbox(label="朝代", placeholder="如：唐")
            poem_content = gr.Textbox(label="诗词全文", lines=5, placeholder="粘贴诗词完整内容，带标点")
            with gr.Row():
                add_btn = gr.Button("添加诗词", variant="primary")
                add_msg = gr.Textbox(label="操作结果", interactive=False)
            refresh_btn.click(fn=list_poetry, outputs=[poetry_list])
            add_btn.click(
                fn=add_poem_handler,
                inputs=[poem_title, poem_author, poem_dynasty, poem_content],
                outputs=[add_msg, poetry_list],
            )

        with gr.Tab("使用说明"):
            gr.Markdown("""
            ## 使用方法

            1. **上传音频** — 点击「上传音频文件」选择 `.wav/.mp3/.m4a` 文件，或点击录音
            2. **选择模型** — `medium` 推荐，`large-v3` 最准但需要 GPU
            3. **勾选诗词纠错** — 系统自动匹配诗词库并进行同音字纠错
            4. **点击转录** — 查看识别文本、时间戳对齐和诗词匹配

            ## 核心特性

            - **Whisper 多模型支持** — tiny/base/small/medium/large-v3
            - **同音字纠错** — 基于 pypinyin 拼音匹配，自动纠正 ASR 同音字错误
            - **时间戳对齐** — 将识别的诗句逐句对齐到原始音频时间轴

            ## 诗词库

            当前内置多首经典唐诗，覆盖李白、杜甫、白居易等诗人。
            可通过 `data/poetry/` 目录下的 JSON 文件扩展诗词库。

            ## 技术栈

            - Python 3.13 · PyTorch · Whisper
            - Gradio · pypinyin · librosa
            """)

    gr.HTML(f"""
    <div style="text-align:center; margin-top:32px; padding:16px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius:12px; color:white;">
      <p style="margin:0; font-size:18px; font-weight:bold;">{CREDIT_TEXT}</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True, share=False)
