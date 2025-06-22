#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   infer_play_diffusion.py
@Time    :   2025/06/21 17:24:05
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   PlayDiffusion inference script
'''

# here put the import lib
import os
import sys
import argparse
from typing import List, Dict, Any, Optional

sys.path.append("src")
import nemo.collections.asr as nemo_asr
from playdiffusion import PlayDiffusion, InpaintInput, TTSInput
import soundfile as sf


class PlayDiffusionInference:
    def __init__(self, asr_model_path: str = "modeldata/audio/asr/parakeet/parakeet-tdt-0.6b-v2.nemo"):
        """
        初始化推理器
        
        Args:
            asr_model_path: ASR模型路径
        """
        self.inpainter = PlayDiffusion()
        self.asr_model = nemo_asr.models.ASRModel.restore_from(asr_model_path)
        print(f"✅ 模型加载完成: {asr_model_path}")

    def run_asr(self, audio_path: str) -> tuple[str, str, List[Dict[str, Any]]]:
        """
        运行ASR识别
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            tuple: (input_text, output_text, word_times)
        """
        print(f"🎤 运行ASR识别: {audio_path}")
        output = self.asr_model.transcribe([audio_path], timestamps=True)
        
        word_times = [{
            "word": word['word'].strip(",.?!"),
            "start": word['start'],
            "end": word['end']
        } for word in output[0].timestamp['word']]
        
        text = output[0].text
        print(f"📝 识别结果: {text}")
        return text, text, word_times

    def inpaint_audio(
        self,
        audio_path: str,
        output_text: str,
        num_steps: int = 30,
        init_temp: float = 1.0,
        init_diversity: float = 1.0,
        guidance: float = 0.5,
        rescale: float = 0.7,
        topk: int = 25,
        output_path: Optional[str] = None
    ) -> str:
        """
        音频修复/编辑
        
        Args:
            audio_path: 输入音频路径
            output_text: 目标文本
            num_steps: 采样步数
            init_temp: 初始温度
            init_diversity: 初始多样性
            guidance: 引导强度
            rescale: 重缩放因子
            topk: top-k采样
            output_path: 输出路径，如果为None则自动生成
            
        Returns:
            str: 输出音频路径
        """
        print(f"🔧 开始音频修复...")
        print(f"📁 输入音频: {audio_path}")
        print(f"📝 目标文本: {output_text}")
        
        # 先运行ASR获取原始文本和时间戳
        input_text, _, word_times = self.run_asr(audio_path)
        
        # 运行inpainter
        output_audio = self.inpainter.inpaint(InpaintInput(
            input_text=input_text,
            output_text=output_text,
            input_word_times=word_times,
            audio=audio_path,
            num_steps=num_steps,
            init_temp=init_temp,
            init_diversity=init_diversity,
            guidance=guidance,
            rescale=rescale,
            topk=topk
        ))
        print("执行成功")
        
        # 保存输出
        if output_path is None:
            output_path = f"output_inpaint_{os.path.basename(audio_path)}"
        
        # 这里需要根据 output_audio 的类型来保存
        # 假设 output_audio 是音频数据，需要保存为文件
        print(f"💾 保存输出音频: {output_path}")
        sf.write(output_path, output_audio[1], output_audio[0], subtype='PCM_16')
        
        return output_path

    def text_to_speech(
        self,
        text: str,
        voice_audio_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        文本转语音
        
        Args:
            text: 要转换的文本
            voice_audio_path: 参考语音路径
            output_path: 输出路径，如果为None则自动生成
            
        Returns:
            str: 输出音频路径
        """
        print(f"🗣️ 开始文本转语音...")
        print(f"📝 输入文本: {text}")
        print(f"🎵 参考语音: {voice_audio_path}")
        
        # 运行TTS
        output_audio = self.inpainter.tts(TTSInput(
            output_text=text,
            voice=voice_audio_path
        ))
        
        # 保存输出
        if output_path is None:
            output_path = f"output_tts_{os.path.basename(voice_audio_path)}"
        
        print(f"💾 保存输出音频: {output_path}")
        sf.write(output_path, output_audio[1], output_audio[0], subtype='PCM_16')

        return output_path

    def run_complete_pipeline(
        self,
        audio_path: str,
        target_text: str,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        运行完整的音频编辑流程
        
        Args:
            audio_path: 输入音频路径
            target_text: 目标文本
            output_path: 输出路径
            **kwargs: 其他参数
            
        Returns:
            str: 输出音频路径
        """
        print("🚀 开始完整音频编辑流程")
        print("="*50)
        
        result = self.inpaint_audio(
            audio_path=audio_path,
            output_text=target_text,
            output_path=output_path,
            **kwargs
        )
        
        print("="*50)
        print("✅ 音频编辑完成!")
        return result


def main():
    parser = argparse.ArgumentParser(description="PlayDiffusion 推理脚本")
    parser.add_argument("--mode", choices=["inpaint", "tts", "asr"], default="inpaint", 
                       help="运行模式")
    parser.add_argument("--audio", required=True, help="输入音频路径")
    parser.add_argument("--text", help="目标文本 (inpaint/tts模式需要)")
    parser.add_argument("--voice", help="参考语音路径 (tts模式需要)")
    parser.add_argument("--output", help="输出路径")
    parser.add_argument("--asr_model", default="modeldata/audio/asr/parakeet/parakeet-tdt-0.6b-v2.nemo",
                       help="ASR模型路径")
    
    # 高级参数
    parser.add_argument("--num_steps", type=int, default=30, help="采样步数")
    parser.add_argument("--init_temp", type=float, default=1.0, help="初始温度")
    parser.add_argument("--init_diversity", type=float, default=1.0, help="初始多样性")
    parser.add_argument("--guidance", type=float, default=0.5, help="引导强度")
    parser.add_argument("--rescale", type=float, default=0.7, help="重缩放因子")
    parser.add_argument("--topk", type=int, default=25, help="top-k采样")
    
    args = parser.parse_args()
    
    # 初始化推理器
    inference = PlayDiffusionInference(args.asr_model)
    
    if args.mode == "asr":
        # 只运行ASR
        text, _, word_times = inference.run_asr(args.audio)
        print(f"识别文本: {text}")
        print(f"词时间戳: {word_times}")
        
    elif args.mode == "inpaint":
        # 音频修复模式
        if not args.text:
            raise ValueError("inpaint模式需要指定 --text 参数")
        
        result = inference.run_complete_pipeline(
            audio_path=args.audio,
            target_text=args.text,
            output_path=args.output,
            num_steps=args.num_steps,
            init_temp=args.init_temp,
            init_diversity=args.init_diversity,
            guidance=args.guidance,
            rescale=args.rescale,
            topk=args.topk
        )
        print(f"输出文件: {result}")
        
    elif args.mode == "tts":
        # TTS模式
        if not args.text:
            raise ValueError("tts模式需要指定 --text 参数")
        if not args.voice:
            raise ValueError("tts模式需要指定 --voice 参数")
        
        result = inference.text_to_speech(
            text=args.text,
            voice_audio_path=args.voice,
            output_path=args.output
        )
        print(f"输出文件: {result}")


if __name__ == "__main__":
    # 如果直接运行，使用简单的测试
    if len(sys.argv) == 1:
        # 测试模式
        inference = PlayDiffusionInference()
        target_text = "Hey, Erika0920, your instant avatar is ready. Fucking you body. Also, click the feedback button to share what you think. Hope you enjoy. Hey, Erika0920"
        # inference.run_complete_pipeline(
        #     audio_path="notebook/data_source/audio/doc_audio/干净_en.mp3",
        #     target_text=target_text,
        #     output_path="output_inpaint_test.wav",
        # )
        inference.text_to_speech(
            text=target_text,
            voice_audio_path="notebook/data_source/audio/doc_audio/干净_en.mp3",
            output_path="output_tts_test.wav",
        )
    else:
        # 命令行模式
        main()