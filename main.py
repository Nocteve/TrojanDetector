import sys
import os
import json
import datetime
import math
import traceback
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLineEdit, QLabel, QTextEdit, QProgressBar, QFileDialog,
                             QMessageBox, QFrame, QSizePolicy, QTabWidget, QComboBox, QScrollArea,
                             QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QFont, QImage, qRed, qGreen, qBlue, qAlpha, qRgba, QPalette, QColor, QIcon

# 原始检测函数导入（分类模型）
from models_for_generating_triggers.glide import generate_trigger_with_classifier
from imc.load_model import load_classification_model, get_classifier_layer
from imc.similarity_analysis import find_far
from imc.evaluate import evaluate_trigger
from imc.detect import detect_backdoor_in_classification_model

# 目标检测相关导入
from obj.detr import DETRBackdoorDetector
from obj.fasterRcnn import FasterRCNNBackdoorDetector

# 忽略警告
warnings.filterwarnings('ignore')

# ==================== 辅助函数（用于目标检测） ====================
def load_clean_images_and_paths(model_dir: str, num_images: int = 4):
    """从模型目录加载干净图像（目标检测专用）"""
    clean_data_dir = os.path.join(model_dir, "clean-example-data")
    if not os.path.exists(clean_data_dir):
        clean_data_dir = os.path.join(model_dir, "example_data")
    if not os.path.exists(clean_data_dir):
        return [], []

    image_files = []
    for f in sorted(os.listdir(clean_data_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(clean_data_dir, f))

    if not image_files:
        return [], []

    images = []
    valid_paths = []
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    for i, img_path in enumerate(image_files[:num_images]):
        try:
            img = Image.open(img_path).convert('RGB')
            tensor_img = transform(img).unsqueeze(0).to(device)
            images.append(tensor_img)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")
    return images, valid_paths

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 输出重定向 ====================
class StreamRedirector(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        if text.strip():
            self.text_written.emit(text)

    def flush(self):
        pass


# ==================== 分类模型检测工作线程 ====================
class ClassificationWorker(QObject):
    output_ready = pyqtSignal(str)
    detection_finished = pyqtSignal(object, object, tuple, str, str)  # score, trigger, best_pair, clean_img_path, trigger_img_path
    detection_error = pyqtSignal(str)

    def __init__(self, model_dir, clean_images_dir,
                 guidance_scale=100.0, num_iterations=2, timestep=50, search_strategy="greedy"):
        super().__init__()
        self.model_dir = model_dir
        self.clean_images_dir = clean_images_dir
        self.guidance_scale = guidance_scale
        self.num_iterations = num_iterations
        self.timestep = timestep
        self.search_strategy = search_strategy
        self._is_running = True

    def run(self):
        redirector = StreamRedirector()
        redirector.text_written.connect(self.output_ready)
        old_stdout = sys.stdout
        sys.stdout = redirector

        try:
            score, trigger, best_pair = detect_backdoor_in_classification_model(
                model_dir=self.model_dir,
                clean_images_dir=self.clean_images_dir,
                guidance_scale=self.guidance_scale,
                num_iterations=self.num_iterations,
                timestep=self.timestep,
                search_strategy=self.search_strategy
            )

            if not self._is_running:
                return

            trigger_path = None
            clean_img_path = None

            if trigger is not None:
                trigger_path = os.path.join(self.model_dir, "detected_trigger_ui.png")
                trigger_img = trigger.squeeze(0).cpu()
                trigger_img = torch.clamp(trigger_img, 0, 1)
                transform = transforms.ToPILImage()
                pil_img = transform(trigger_img)
                pil_img.save(trigger_path)

                if os.path.exists(self.clean_images_dir):
                    files = [f for f in os.listdir(self.clean_images_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                    if files:
                        clean_img_path = os.path.join(self.clean_images_dir, files[0])

            self.detection_finished.emit(score, trigger, best_pair, clean_img_path, trigger_path)

        except Exception as e:
            self.detection_error.emit(str(e))
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout

    def stop(self):
        self._is_running = False


# ==================== 目标检测模型检测工作线程 ====================
class ObjectDetectionWorker(QObject):
    output_ready = pyqtSignal(str)
    detection_finished = pyqtSignal(object, object, str, str)  # score, trigger, clean_img_path, trigger_img_path
    detection_error = pyqtSignal(str)

    def __init__(self, model_dir, detector_type, guidance_scale=80.0, num_iterations=2, timestep=50, noise_scale=0.2):
        super().__init__()
        self.model_dir = model_dir
        self.detector_type = detector_type  # 'fasterrcnn' 或 'detr'
        self.guidance_scale = guidance_scale
        self.num_iterations = num_iterations
        self.timestep = timestep
        self.noise_scale = noise_scale
        self._is_running = True

    def run(self):
        redirector = StreamRedirector()
        redirector.text_written.connect(self.output_ready)
        old_stdout = sys.stdout
        sys.stdout = redirector

        try:
            # 根据类型选择检测器
            if self.detector_type == 'fasterrcnn':
                detector = FasterRCNNBackdoorDetector(device=device)
                config_path = os.path.join(self.model_dir, "reduced-config.json")
                model_path = os.path.join(self.model_dir, "model.pt")
            else:  # detr
                detector = DETRBackdoorDetector(device=device)
                config_path = os.path.join(self.model_dir, "reduced-config.json")
                model_path = os.path.join(self.model_dir, "model-state-dict.pt")

            # 加载模型
            self.output_ready.emit(f"加载目标检测模型 ({self.detector_type})...")
            model = detector.load_model(model_path, config_path)
            if model is None:
                raise Exception("模型加载失败")

            # 加载干净图像
            self.output_ready.emit("加载干净示例图像...")
            clean_images, image_paths = load_clean_images_and_paths(self.model_dir, num_images=4)
            if not clean_images:
                raise Exception("无法加载干净图像")

            # 执行检测
            self.output_ready.emit("运行后门检测算法（使用GLIDE扩散模型）...")
            score, trigger, results = detector.detect_backdoor(
                model, clean_images, image_paths,
                guidance_scale=self.guidance_scale,
                num_iterations=self.num_iterations,
                timestep=self.timestep,
                noise_scale=self.noise_scale
            )

            if not self._is_running:
                return

            # 保存触发器图像
            trigger_path = None
            clean_img_path = image_paths[0] if image_paths else None

            if trigger is not None:
                # 保存触发器可视化图像
                output_dir = os.path.join(self.model_dir, "backdoor_detection")
                os.makedirs(output_dir, exist_ok=True)
                trigger_path = os.path.join(output_dir, "generated_trigger.png")

                # 处理触发器张量（可能多通道）
                t = trigger.squeeze(0).cpu()
                if t.size(0) > 3:
                    t = t.mean(0, keepdim=True).repeat(3, 1, 1)
                elif t.size(0) == 1:
                    t = t.repeat(3, 1, 1)
                # 归一化到0-255
                t = (t - t.min()) / (t.max() - t.min() + 1e-8) * 255
                t = t.byte()
                trigger_pic = transforms.ToPILImage()(t)
                trigger_pic.save(trigger_path)

            self.detection_finished.emit(score, trigger, clean_img_path, trigger_path)

        except Exception as e:
            self.detection_error.emit(str(e))
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout

    def stop(self):
        self._is_running = False


# ==================== 图像分类界面 ====================
class ClassificationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.thread = None
        self.worker = None

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # 目录选择
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("📁 模型目录:"))
        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setPlaceholderText("选择包含模型的文件夹...")
        top_layout.addWidget(self.model_dir_edit)
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_model_dir)
        # 浏览按钮固定宽度，不随字体变化
        self.browse_btn.setFixedWidth(110)
        top_layout.addWidget(self.browse_btn)
        self.start_btn = QPushButton("▶ 开始检测")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setEnabled(False)
        # 开始检测按钮固定宽度，确保大字体下完整显示
        self.start_btn.setFixedWidth(150)
        top_layout.addWidget(self.start_btn)
        layout.addLayout(top_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(22)
        layout.addWidget(self.progress_bar)

        # 图像并排显示
        images_layout = QHBoxLayout()
        images_layout.setSpacing(25)

        # 中毒示例卡片
        clean_card = QFrame()
        clean_card.setFrameShape(QFrame.StyledPanel)
        clean_vbox = QVBoxLayout(clean_card)
        clean_vbox.setAlignment(Qt.AlignCenter)
        clean_vbox.addWidget(QLabel("== 中毒示例 =="), alignment=Qt.AlignCenter)
        self.clean_img_label = QLabel()
        self.clean_img_label.setAlignment(Qt.AlignCenter)
        self.clean_img_label.setMinimumSize(320, 320)
        self.clean_img_label.setMaximumSize(400, 400)
        self.clean_img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.clean_img_label.setStyleSheet("border: none; background-color: #1a1a2a; border-radius: 5px;")
        self.clean_img_label.setScaledContents(False)
        clean_vbox.addWidget(self.clean_img_label)
        images_layout.addWidget(clean_card)

        # 触发器图像卡片
        trigger_card = QFrame()
        trigger_card.setFrameShape(QFrame.StyledPanel)
        trigger_vbox = QVBoxLayout(trigger_card)
        trigger_vbox.setAlignment(Qt.AlignCenter)
        trigger_vbox.addWidget(QLabel("== 检测到的触发器 =="), alignment=Qt.AlignCenter)
        self.trigger_img_label = QLabel()
        self.trigger_img_label.setAlignment(Qt.AlignCenter)
        self.trigger_img_label.setMinimumSize(320, 320)
        self.trigger_img_label.setMaximumSize(400, 400)
        self.trigger_img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.trigger_img_label.setStyleSheet("border: none; background-color: #1a1a2a; border-radius: 5px;")
        self.trigger_img_label.setScaledContents(False)
        trigger_vbox.addWidget(self.trigger_img_label)
        images_layout.addWidget(trigger_card)
        layout.addLayout(images_layout)

        # 检测结果卡片
        result_card = QFrame()
        result_card.setObjectName("resultCard")
        result_card.setFrameShape(QFrame.StyledPanel)
        result_card.setFixedHeight(80)
        result_layout = QHBoxLayout(result_card)
        result_layout.setSpacing(20)
        result_layout.setContentsMargins(15, 5, 15, 5)

        # 得分区域
        score_widget = QWidget()
        score_layout = QVBoxLayout(score_widget)
        score_layout.setAlignment(Qt.AlignCenter)
        score_layout.addWidget(QLabel("后门得分"), alignment=Qt.AlignCenter)
        self.score_label = QLabel("--")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setStyleSheet("font-weight: bold; color: #3498db;")
        score_layout.addWidget(self.score_label)
        result_layout.addWidget(score_widget)

        # 类别对
        pair_widget = QWidget()
        pair_layout = QVBoxLayout(pair_widget)
        pair_layout.setAlignment(Qt.AlignCenter)
        pair_layout.addWidget(QLabel("目标 / 源类别"), alignment=Qt.AlignCenter)
        self.class_pair_label = QLabel("-- / --")
        self.class_pair_label.setAlignment(Qt.AlignCenter)
        self.class_pair_label.setStyleSheet("")
        pair_layout.addWidget(self.class_pair_label)
        result_layout.addWidget(pair_widget)

        # 结论
        verdict_widget = QWidget()
        verdict_layout = QVBoxLayout(verdict_widget)
        verdict_layout.setAlignment(Qt.AlignCenter)
        verdict_layout.addWidget(QLabel("检测结果"), alignment=Qt.AlignCenter)
        self.verdict_label = QLabel("等待检测")
        self.verdict_label.setAlignment(Qt.AlignCenter)
        self.verdict_label.setStyleSheet("font-weight: bold; color: #95a5a6;")
        verdict_layout.addWidget(self.verdict_label)
        result_layout.addWidget(verdict_widget)
        layout.addWidget(result_card)

        # 控制台输出
        console_label = QLabel("📟 检测日志")
        console_label.setStyleSheet("font-weight: bold; color: #3498db;")
        layout.addWidget(console_label)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 10))
        self.output_text.setMinimumHeight(180)
        layout.addWidget(self.output_text)

        # 状态标签
        self.status_label = QLabel("⚡ 就绪")
        self.status_label.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(self.status_label, alignment=Qt.AlignRight)

        # 连接信号
        self.model_dir_edit.textChanged.connect(self.check_directory)

    def browse_model_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if dir_path:
            self.model_dir_edit.setText(dir_path)

    def check_directory(self, text):
        if os.path.isdir(text):
            clean_dir = os.path.join(text, "clean-example-data")
            if not os.path.isdir(clean_dir):
                clean_dir = os.path.join(text, "example_data")
            if os.path.isdir(clean_dir):
                self.start_btn.setEnabled(True)
                self.status_label.setText("✅ 目录有效，可以开始检测")
            else:
                self.start_btn.setEnabled(False)
                self.status_label.setText("❌ 缺少干净图像子目录")
        else:
            self.start_btn.setEnabled(False)
            self.status_label.setText("⛔ 请选择有效目录")

    def start_detection(self):
        self.start_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.output_text.clear()
        self.clean_img_label.clear()
        self.trigger_img_label.clear()
        self.score_label.setText("--")
        self.class_pair_label.setText("-- / --")
        self.verdict_label.setText("检测中...")
        self.verdict_label.setStyleSheet("font-weight: bold; color: #3498db;")
        self.progress_bar.setVisible(True)
        self.status_label.setText("🔮 检测中，请稍候...")

        model_dir = self.model_dir_edit.text()
        clean_dir = os.path.join(model_dir, "clean-example-data")
        if not os.path.isdir(clean_dir):
            clean_dir = os.path.join(model_dir, "example_data")

        self.thread = QThread()
        self.worker = ClassificationWorker(model_dir, clean_dir)
        self.worker.moveToThread(self.thread)

        self.worker.output_ready.connect(self.append_output)
        self.worker.detection_finished.connect(self.on_detection_finished)
        self.worker.detection_error.connect(self.on_detection_error)
        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self.cleanup_thread)

        self.thread.start()

    def append_output(self, text):
        self.output_text.append(text)
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.End)
        self.output_text.setTextCursor(cursor)

    def on_detection_finished(self, score, trigger, best_pair, clean_img_path, trigger_img_path):
        self.progress_bar.setVisible(False)

        # 显示叠加图像
        if clean_img_path and os.path.exists(clean_img_path) and trigger_img_path and os.path.exists(trigger_img_path):
            clean_qimage = QImage(clean_img_path)
            trigger_qimage = QImage(trigger_img_path)

            if clean_qimage.size() != trigger_qimage.size():
                trigger_qimage = trigger_qimage.scaled(clean_qimage.size(), Qt.KeepAspectRatioByExpanding)

            result_qimage = QImage(clean_qimage.size(), QImage.Format_ARGB32)

            for x in range(clean_qimage.width()):
                for y in range(clean_qimage.height()):
                    c_pixel = clean_qimage.pixel(x, y)
                    t_pixel = trigger_qimage.pixel(x, y)

                    c_r = qRed(c_pixel)
                    c_g = qGreen(c_pixel)
                    c_b = qBlue(c_pixel)
                    c_a = qAlpha(c_pixel)

                    t_r = qRed(t_pixel)
                    t_g = qGreen(t_pixel)
                    t_b = qBlue(t_pixel)
                    t_a = qAlpha(t_pixel)

                    r = min(255, c_r + t_r)
                    g = min(255, c_g + t_g)
                    b = min(255, c_b + t_b)
                    a = min(255, c_a + t_a)

                    result_qimage.setPixel(x, y, qRgba(r, g, b, a))

            pixmap = QPixmap.fromImage(result_qimage)
            label_size = self.clean_img_label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.clean_img_label.setPixmap(scaled_pixmap)
        else:
            self.clean_img_label.setText("无法生成中毒示例")

        if trigger_img_path and os.path.exists(trigger_img_path):
            pixmap = QPixmap(trigger_img_path)
            label_size = self.trigger_img_label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.trigger_img_label.setPixmap(scaled_pixmap)
        else:
            self.trigger_img_label.setText("未生成触发器")

        self.score_label.setText(f"{score:.4f}")
        self.class_pair_label.setText(f"{best_pair[0]} / {best_pair[1]}")

        if score > 0.6:
            verdict = "⚠️ 可能存在后门"
            color = "#e67e22"
        else:
            verdict = "✅ 可能安全"
            color = "#2ecc71"
        self.verdict_label.setText(verdict)
        self.verdict_label.setStyleSheet(f"font-weight: bold; color: {color};")

        self.append_output("\n" + "="*60)
        self.append_output(f"📊 后门得分: {score:.4f}")
        self.append_output(f"🎯 最佳类别对: 目标={best_pair[0]}, 源={best_pair[1]}")
        self.append_output(f"🔍 结论: {verdict}")
        self.append_output("="*60)

        self.status_label.setText("✅ 检测完成")
        self.start_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)

        self.thread.quit()
        self.thread.wait()

    def on_detection_error(self, error_msg):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "错误", f"检测过程中发生异常:\n{error_msg}")
        self.append_output(f"\n🔥 错误: {error_msg}")
        self.status_label.setText("❌ 检测出错")
        self.start_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.verdict_label.setText("检测失败")
        self.verdict_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

    def cleanup_thread(self):
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if self.thread:
            self.thread.deleteLater()
            self.thread = None


# ==================== 目标检测界面 ====================
class ObjectDetectionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.thread = None
        self.worker = None

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # 目录选择 + 检测器类型
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("📁 模型目录:"))
        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setPlaceholderText("选择包含模型的文件夹...")
        top_layout.addWidget(self.model_dir_edit)
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_model_dir)
        # 浏览按钮固定宽度
        self.browse_btn.setFixedWidth(110)
        top_layout.addWidget(self.browse_btn)

        top_layout.addWidget(QLabel("检测器:"))
        self.detector_combo = QComboBox()
        self.detector_combo.addItems(["Faster R-CNN", "DETR"])
        self.detector_combo.setFixedWidth(120)
        top_layout.addWidget(self.detector_combo)

        self.start_btn = QPushButton("▶ 开始检测")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setEnabled(False)
        # 开始检测按钮固定宽度
        self.start_btn.setFixedWidth(150)
        top_layout.addWidget(self.start_btn)
        layout.addLayout(top_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(22)
        layout.addWidget(self.progress_bar)

        # 图像并排显示
        images_layout = QHBoxLayout()
        images_layout.setSpacing(25)

        # 中毒示例卡片
        clean_card = QFrame()
        clean_card.setFrameShape(QFrame.StyledPanel)
        clean_vbox = QVBoxLayout(clean_card)
        clean_vbox.setAlignment(Qt.AlignCenter)
        clean_vbox.addWidget(QLabel("== 中毒示例 =="), alignment=Qt.AlignCenter)
        self.clean_img_label = QLabel()
        self.clean_img_label.setAlignment(Qt.AlignCenter)
        self.clean_img_label.setMinimumSize(320, 320)
        self.clean_img_label.setMaximumSize(400, 400)
        self.clean_img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.clean_img_label.setStyleSheet("border: none; background-color: #1a1a2a; border-radius: 5px;")
        self.clean_img_label.setScaledContents(False)
        clean_vbox.addWidget(self.clean_img_label)
        images_layout.addWidget(clean_card)

        # 触发器图像卡片
        trigger_card = QFrame()
        trigger_card.setFrameShape(QFrame.StyledPanel)
        trigger_vbox = QVBoxLayout(trigger_card)
        trigger_vbox.setAlignment(Qt.AlignCenter)
        trigger_vbox.addWidget(QLabel("== 检测到的触发器 =="), alignment=Qt.AlignCenter)
        self.trigger_img_label = QLabel()
        self.trigger_img_label.setAlignment(Qt.AlignCenter)
        self.trigger_img_label.setMinimumSize(320, 320)
        self.trigger_img_label.setMaximumSize(400, 400)
        self.trigger_img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.trigger_img_label.setStyleSheet("border: none; background-color: #1a1a2a; border-radius: 5px;")
        self.trigger_img_label.setScaledContents(False)
        trigger_vbox.addWidget(self.trigger_img_label)
        images_layout.addWidget(trigger_card)
        layout.addLayout(images_layout)

        # 检测结果卡片
        result_card = QFrame()
        result_card.setObjectName("resultCard")
        result_card.setFrameShape(QFrame.StyledPanel)
        result_card.setFixedHeight(80)
        result_layout = QHBoxLayout(result_card)
        result_layout.setSpacing(20)
        result_layout.setContentsMargins(15, 5, 15, 5)

        # 得分区域
        score_widget = QWidget()
        score_layout = QVBoxLayout(score_widget)
        score_layout.setAlignment(Qt.AlignCenter)
        score_layout.addWidget(QLabel("后门得分"), alignment=Qt.AlignCenter)
        self.score_label = QLabel("--")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setStyleSheet("font-weight: bold; color: #3498db;")
        score_layout.addWidget(self.score_label)
        result_layout.addWidget(score_widget)

        # 结论
        verdict_widget = QWidget()
        verdict_layout = QVBoxLayout(verdict_widget)
        verdict_layout.setAlignment(Qt.AlignCenter)
        verdict_layout.addWidget(QLabel("检测结果"), alignment=Qt.AlignCenter)
        self.verdict_label = QLabel("等待检测")
        self.verdict_label.setAlignment(Qt.AlignCenter)
        self.verdict_label.setStyleSheet("font-weight: bold; color: #95a5a6;")
        verdict_layout.addWidget(self.verdict_label)
        result_layout.addWidget(verdict_widget)
        layout.addWidget(result_card)

        # 控制台输出
        console_label = QLabel("📟 检测日志")
        console_label.setStyleSheet("font-weight: bold; color: #3498db;")
        layout.addWidget(console_label)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 10))
        self.output_text.setMinimumHeight(180)
        layout.addWidget(self.output_text)

        # 状态标签
        self.status_label = QLabel("⚡ 就绪")
        self.status_label.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(self.status_label, alignment=Qt.AlignRight)

        # 连接信号
        self.model_dir_edit.textChanged.connect(self.check_directory)

    def browse_model_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if dir_path:
            self.model_dir_edit.setText(dir_path)

    def check_directory(self, text):
        if os.path.isdir(text):
            config_exists = (os.path.exists(os.path.join(text, "reduced-config.json")) or
                             os.path.exists(os.path.join(text, "config.json")))
            if config_exists:
                self.start_btn.setEnabled(True)
                self.status_label.setText("✅ 目录有效，可以开始检测")
            else:
                self.start_btn.setEnabled(False)
                self.status_label.setText("❌ 缺少配置文件 (reduced-config.json)")
        else:
            self.start_btn.setEnabled(False)
            self.status_label.setText("⛔ 请选择有效目录")

    def start_detection(self):
        self.start_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.output_text.clear()
        self.clean_img_label.clear()
        self.trigger_img_label.clear()
        self.score_label.setText("--")
        self.verdict_label.setText("检测中...")
        self.verdict_label.setStyleSheet("font-weight: bold; color: #3498db;")
        self.progress_bar.setVisible(True)
        self.status_label.setText("🔮 检测中，请稍候...")

        model_dir = self.model_dir_edit.text()
        detector_type = 'fasterrcnn' if self.detector_combo.currentText() == "Faster R-CNN" else 'detr'

        self.thread = QThread()
        self.worker = ObjectDetectionWorker(model_dir, detector_type)
        self.worker.moveToThread(self.thread)

        self.worker.output_ready.connect(self.append_output)
        self.worker.detection_finished.connect(self.on_detection_finished)
        self.worker.detection_error.connect(self.on_detection_error)
        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self.cleanup_thread)

        self.thread.start()

    def append_output(self, text):
        self.output_text.append(text)
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.End)
        self.output_text.setTextCursor(cursor)

    def on_detection_finished(self, score, trigger, clean_img_path, trigger_img_path):
        self.progress_bar.setVisible(False)

        if clean_img_path and os.path.exists(clean_img_path) and trigger_img_path and os.path.exists(trigger_img_path):
            clean_qimage = QImage(clean_img_path)
            trigger_qimage = QImage(trigger_img_path)

            if clean_qimage.size() != trigger_qimage.size():
                trigger_qimage = trigger_qimage.scaled(clean_qimage.size(), Qt.KeepAspectRatioByExpanding)

            result_qimage = QImage(clean_qimage.size(), QImage.Format_ARGB32)

            for x in range(clean_qimage.width()):
                for y in range(clean_qimage.height()):
                    c_pixel = clean_qimage.pixel(x, y)
                    t_pixel = trigger_qimage.pixel(x, y)

                    c_r = qRed(c_pixel)
                    c_g = qGreen(c_pixel)
                    c_b = qBlue(c_pixel)
                    c_a = qAlpha(c_pixel)

                    t_r = qRed(t_pixel)
                    t_g = qGreen(t_pixel)
                    t_b = qBlue(t_pixel)
                    t_a = qAlpha(t_pixel)

                    r = min(255, c_r + t_r)
                    g = min(255, c_g + t_g)
                    b = min(255, c_b + t_b)
                    a = min(255, c_a + t_a)

                    result_qimage.setPixel(x, y, qRgba(r, g, b, a))

            pixmap = QPixmap.fromImage(result_qimage)
            label_size = self.clean_img_label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.clean_img_label.setPixmap(scaled_pixmap)
        else:
            self.clean_img_label.setText("无法生成中毒示例")

        if trigger_img_path and os.path.exists(trigger_img_path):
            pixmap = QPixmap(trigger_img_path)
            label_size = self.trigger_img_label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.trigger_img_label.setPixmap(scaled_pixmap)
        else:
            self.trigger_img_label.setText("未生成触发器")

        display_score = score
        self.score_label.setText(f"{display_score:.4f}")

        if display_score > 0.45:
            verdict = "⚠️ 可能存在后门"
            color = "#e67e22"
        else:
            verdict = "✅ 可能安全"
            color = "#2ecc71"
        self.verdict_label.setText(verdict)
        self.verdict_label.setStyleSheet(f"font-weight: bold; color: {color};")

        self.append_output("\n" + "="*60)
        self.append_output(f"📊 后门得分: {display_score:.4f}")
        self.append_output(f"🔍 结论: {verdict}")
        self.append_output("="*60)

        self.status_label.setText("✅ 检测完成")
        self.start_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)

        self.thread.quit()
        self.thread.wait()

    def on_detection_error(self, error_msg):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "错误", f"检测过程中发生异常:\n{error_msg}")
        self.append_output(f"\n🔥 错误: {error_msg}")
        self.status_label.setText("❌ 检测出错")
        self.start_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.verdict_label.setText("检测失败")
        self.verdict_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

    def cleanup_thread(self):
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if self.thread:
            self.thread.deleteLater()
            self.thread = None


# ==================== 设置界面 ====================
class SettingsTab(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title_label = QLabel("TrojanDetector——触发器反演式深度学习模型后门检测平台")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; color: #2c7eb5; margin: 10px;")
        layout.addWidget(title_label)

        # 主题选择
        theme_label = QLabel("界面主题")
        theme_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(theme_label)

        theme_layout = QHBoxLayout()
        self.dark_radio = QRadioButton("深色主题")
        self.light_radio = QRadioButton("浅色主题")
        theme_layout.addWidget(self.dark_radio)
        theme_layout.addWidget(self.light_radio)
        theme_layout.addStretch()
        layout.addLayout(theme_layout)

        # 默认选中深色
        self.dark_radio.setChecked(True)

        # 连接信号
        self.dark_radio.toggled.connect(self.on_theme_changed)
        self.light_radio.toggled.connect(self.on_theme_changed)

        # 字体大小设置
        font_label = QLabel("字体大小")
        font_label.setStyleSheet("font-weight: bold; margin-top: 15px;")
        layout.addWidget(font_label)

        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("大小:"))
        self.font_combo = QComboBox()
        self.font_combo.addItems(["小 (9pt)", "中 (10pt)", "大 (12pt)"])
        self.font_combo.setCurrentIndex(1)  # 默认中
        self.font_combo.currentIndexChanged.connect(self.on_font_changed)
        font_layout.addWidget(self.font_combo)
        font_layout.addStretch()
        layout.addLayout(font_layout)

        layout.addStretch()

    def on_theme_changed(self):
        if self.dark_radio.isChecked():
            self.main_window.apply_theme('dark')
        elif self.light_radio.isChecked():
            self.main_window.apply_theme('light')

    def on_font_changed(self, index):
        sizes = [9, 10, 12]
        font = QFont("SansSerif", sizes[index])
        QApplication.setFont(font)
        # 强制刷新界面：重新应用当前主题（即使主题未变），使字体变化立即生效
        current_theme = self.main_window.current_theme
        self.main_window.apply_theme(current_theme)


# ==================== 关于界面 ====================
class AboutTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        title_label = QLabel("系统架构图")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; margin: 10px; color: #2c7eb5;")
        layout.addWidget(title_label)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, "assets/UML.png")

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(False)
        scroll_area.setAlignment(Qt.AlignCenter)
        scroll_area.setStyleSheet("border: none; background-color: #1e1e2f;")
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setMinimumSize(200, 150)
        else:
            self.image_label.setText(f"图片未找到: {img_path}")
            self.image_label.setStyleSheet("color: red;")

        scroll_area.setWidget(self.image_label)
        layout.addWidget(scroll_area)

        info_label = QLabel("TrojanDetector 后门检测系统 - ver 1.3.6\n© 2026")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #aaa; margin: 10px;")
        layout.addWidget(info_label)


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrojanDetector 后门检测系统 - ver 1.3.6")
        self.setGeometry(100, 100, 1400, 950)

        # 当前主题，默认为深色
        self.current_theme = 'dark'

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.tabs = QTabWidget()
        self.tabs.addTab(ClassificationTab(), "图像分类模型")
        self.tabs.addTab(ObjectDetectionTab(), "目标检测模型")
        self.tabs.addTab(SettingsTab(self), "设置")
        self.tabs.addTab(AboutTab(), "关于")

        main_layout.addWidget(self.tabs)

        # 应用默认深色主题
        self.apply_theme('dark')

    def apply_theme(self, theme_name):
        """根据主题名称应用样式表，并记录当前主题"""
        self.current_theme = theme_name

        if theme_name == 'dark':
            # 深色主题样式（无font-size）
            style = """
            QMainWindow {
                background-color: #1e1e2f;
            }
            QTabWidget::pane {
                border: 1px solid #3a4a5a;
                background-color: #252536;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #2c3e50;
                color: #ecf0f1;
                padding: 8px 20px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #34495e;
            }
            QLabel {
                color: #eaeaea;
            }
            QPushButton {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #34495e;
                border-color: #3498db;
            }
            QPushButton:disabled {
                background-color: #3a3a4a;
                color: #7f8c8d;
                border-color: #4a4a5a;
            }
            QLineEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                padding: 6px;
                border-radius: 4px;
            }
            QTextEdit {
                background-color: #1e1e2f;
                color: #a0d6ff;
                font-family: 'Consolas', 'Courier New', monospace;
                border: 1px solid #34495e;
                border-radius: 5px;
            }
            QProgressBar {
                border: 1px solid #34495e;
                border-radius: 5px;
                text-align: center;
                color: #ecf0f1;
                background-color: #2c3e50;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 5px;
            }
            QFrame {
                background-color: #252536;
                border: 1px solid #3a4a5a;
                border-radius: 8px;
            }
            #resultCard {
                background-color: #2a2a3c;
                border: 2px solid #3498db;
                border-radius: 10px;
            }
            QComboBox {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                padding: 5px;
                border-radius: 4px;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #34495e;
                background-color: #2c3e50;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ecf0f1;
                width: 0;
                height: 0;
            }
            QComboBox QAbstractItemView {
                background-color: #2c3e50;
                color: #ecf0f1;
                selection-background-color: #3498db;
                selection-color: white;
                border: none;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                min-height: 25px;
                padding-left: 10px;
                background-color: #2c3e50;
                color: #ecf0f1;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #3498db;
                color: white;
            }
            QComboBox QAbstractItemView::item:hover:!selected {
                background-color: #34495e;
            }
            QComboBox::popup {
                border: none;
                background-color: #2c3e50;
                border-radius: 4px;
                padding: 0;
                margin: 0;
                box-shadow: none;
            }
            QRadioButton {
                color: #ecf0f1;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
            }
            QRadioButton::indicator::unchecked {
                border: 1px solid #7f8c8d;
                background: #2c3e50;
                border-radius: 8px;
            }
            QRadioButton::indicator::checked {
                border: 1px solid #3498db;
                background: #3498db;
                border-radius: 8px;
            }
            """
            # 设置分类界面类别标签颜色
            class_tab = self.tabs.widget(0)
            if class_tab and hasattr(class_tab, 'class_pair_label'):
                class_tab.class_pair_label.setStyleSheet("color: #ecf0f1;")
        else:
            # 浅色主题样式（无font-size）
            style = """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: #ffffff;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                color: #333333;
                padding: 8px 20px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #d0d0d0;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background-color: #e0e0e0;
                color: #333333;
                border: 1px solid #c0c0c0;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
                border-color: #3498db;
            }
            QPushButton:disabled {
                background-color: #f0f0f0;
                color: #a0a0a0;
                border-color: #d0d0d0;
            }
            QLineEdit {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #c0c0c0;
                padding: 6px;
                border-radius: 4px;
            }
            QTextEdit {
                background-color: #ffffff;
                color: #2c3e50;
                font-family: 'Consolas', 'Courier New', monospace;
                border: 1px solid #c0c0c0;
                border-radius: 5px;
            }
            QProgressBar {
                border: 1px solid #c0c0c0;
                border-radius: 5px;
                text-align: center;
                color: #333333;
                background-color: #e0e0e0;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 5px;
            }
            QFrame {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                border-radius: 8px;
            }
            #resultCard {
                background-color: #f8f8f8;
                border: 2px solid #3498db;
                border-radius: 10px;
            }
            QComboBox {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #c0c0c0;
                padding: 5px;
                border-radius: 4px;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #c0c0c0;
                background-color: #ffffff;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #333333;
                width: 0;
                height: 0;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #333333;
                selection-background-color: #3498db;
                selection-color: white;
                border: 1px solid #c0c0c0;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                min-height: 25px;
                padding-left: 10px;
                background-color: #ffffff;
                color: #333333;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #3498db;
                color: white;
            }
            QComboBox QAbstractItemView::item:hover:!selected {
                background-color: #f0f0f0;
            }
            QComboBox::popup {
                border: 1px solid #c0c0c0;
                background-color: #ffffff;
                border-radius: 4px;
                padding: 0;
                margin: 0;
                box-shadow: none;
            }
            QRadioButton {
                color: #333333;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
            }
            QRadioButton::indicator::unchecked {
                border: 1px solid #b0b0b0;
                background: #ffffff;
                border-radius: 8px;
            }
            QRadioButton::indicator::checked {
                border: 1px solid #3498db;
                background: #3498db;
                border-radius: 8px;
            }
            """
            class_tab = self.tabs.widget(0)
            if class_tab and hasattr(class_tab, 'class_pair_label'):
                class_tab.class_pair_label.setStyleSheet("color: #2c3e50;")

        self.setStyleSheet(style)


# ==================== 程序入口 ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    icon_paths = [
        os.path.join(current_dir, "assets/trojan_detector.ico"),
        os.path.join(current_dir, "assets/trojan_detector.png")
    ]
    for icon_path in icon_paths:
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
            break

    palette = app.palette()
    palette.setColor(palette.Base, QColor("#2c3e50"))
    palette.setColor(palette.Window, QColor("#1e1e2f"))
    palette.setColor(palette.Text, QColor("#ecf0f1"))
    palette.setColor(palette.Highlight, QColor("#3498db"))
    palette.setColor(palette.HighlightedText, QColor("white"))
    app.setPalette(palette)

    window = MainWindow()
    for icon_path in icon_paths:
        if os.path.exists(icon_path):
            window.setWindowIcon(QIcon(icon_path))
            break
    window.show()
    sys.exit(app.exec_())