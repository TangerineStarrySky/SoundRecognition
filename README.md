# 基于 GMM-HMM 的孤立词语音识别系统

本项目实现了一个面向孤立词语音识别任务的系统，目标是对预定义的孤立词进行准确识别。孤立词识别是指识别在固定时间段内单独发音的词语，不涉及连续语音中的上下文信息。


本系统支持以下20个孤立词的语音识别：
<div align=center>  
<img src="./imgs/words.png" width=70%> 
</div>

## 功能特性

- 数据预处理：包括数据整理与筛选、端点检测、MFCC 特征提取。
- 语音识别模型设计：采用 GMM-HMM 声学模型架构，包括隐马尔可夫模型（HMM）和高斯混合模型（GMM）。
- 实验结果与分析：对不同说话人的识别准确率进行评估，并分析低准确率样本的识别错误。

## 系统框架图

<div align=center>  
<img src="./imgs/framework.png" width=70%> 
</div>

## 快速开始

1. **环境准备**
- Python 3.x
- numpy
- scipy
- librosa
- hmmlearn
- flask
- flask-cors
- pyaudio
- scikit-learn

2. **安装依赖**
```bash
pip install numpy scipy librosa hmmlearn flask flask-cors pyaudio scikit-learn
```

3. **运行前端界面**
```bash
python app.py
```


**运行示例图**

<div align=center>  
<img src="./imgs/show.png" width=70%> 
</div>

**前端页面组成**
- **波形可视化区域**：用于绘制音频波形。用户开始录音后，该区域会实时显示麦克风采集的语音信号，录音结束后，自动切换显示当前录音文件的波形图。
- **控制按钮**：
  - “开始录音”按钮：点击后启动语音录制功能，同时激活波形绘制。
  - “识别”按钮：用于将录制音频提交至后端模型进行语音识别。
  - “重置”按钮：用于清空当前录音结果，准备下一次识别。
- **音频回放区**：显示最近一次录音的音频控件，用户可手动播放回听录音内容。
- **识别结果显示区**：在“识别结果”标签后实时显示系统预测的词语。

