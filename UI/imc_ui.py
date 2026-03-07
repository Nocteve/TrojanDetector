import sys
import os
import datetime
import traceback

import torch
import torchvision.transforms as transforms
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLineEdit, QLabel, QTextEdit, QProgressBar, QFileDialog,
                             QMessageBox, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette,QImage,qRed, qGreen, qBlue, qAlpha, qRgba

# 原始检测函数导入（需确保路径正确）
from models_for_generating_triggers.glide import generate_trigger_with_classifier
from imc.load_model import load_classification_model, get_classifier_layer
from imc.similarity_analysis import find_far
from imc.evaluate import evaluate_trigger
from imc.detect import detect_backdoor_in_classification_model


# ==================== 输出重定向 ====================
class StreamRedirector(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        if text.strip():
            self.text_written.emit(text)

    def flush(self):
        pass


# ==================== 检测工作线程 ====================
class DetectionWorker(QObject):
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


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DISTIL 后门检测 - 安全中心版")
        self.setGeometry(100, 100, 1300, 900)

        # ---------- 优化后的色彩方案 ----------
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2f;  /* 深紫灰背景 */
            }
            QLabel {
                color: #eaeaea;
                font-size: 12px;
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
                color: #a0d6ff;  /* 浅蓝绿 */
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
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
                background-color: #3498db;  /* 柔和蓝色 */
                border-radius: 5px;
            }
            QFrame {
                background-color: #252536;
                border: 1px solid #3a4a5a;
                border-radius: 8px;
            }
            /* 结果卡片特殊样式 */
            #resultCard {
                background-color: #2a2a3c;
                border: 2px solid #3498db;
                border-radius: 10px;
            }
        """)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ---------- 1. 顶部：目录选择 + 开始按钮 ----------
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("📁 模型目录:"))
        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setPlaceholderText("选择包含模型的文件夹...")
        top_layout.addWidget(self.model_dir_edit)

        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_model_dir)
        top_layout.addWidget(self.browse_btn)

        self.start_btn = QPushButton("▶ 开始检测")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setEnabled(False)
        self.start_btn.setFixedWidth(120)
        top_layout.addWidget(self.start_btn)

        main_layout.addLayout(top_layout)

        # ---------- 2. 进度条 ----------
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(22)
        main_layout.addWidget(self.progress_bar)

        # ---------- 3. 图像并排显示区域 ----------
        images_layout = QHBoxLayout()
        images_layout.setSpacing(25)

        # 干净图像卡片
        clean_card = QFrame()
        clean_card.setFrameShape(QFrame.StyledPanel)
        clean_vbox = QVBoxLayout(clean_card)
        clean_vbox.setAlignment(Qt.AlignCenter)
        clean_vbox.addWidget(QLabel("== 干净样本 =="), alignment=Qt.AlignCenter)

        self.clean_img_label = QLabel()
        self.clean_img_label.setAlignment(Qt.AlignCenter)
        self.clean_img_label.setMinimumSize(320, 320)
        self.clean_img_label.setMaximumSize(400, 400)
        self.clean_img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.clean_img_label.setStyleSheet("border: none; background-color: #1a1a2a; border-radius: 5px;")
        self.clean_img_label.setScaledContents(False)  # 手动缩放保持比例
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

        main_layout.addLayout(images_layout)

        # ---------- 4. 检测结果卡片 ----------
        result_card = QFrame()
        result_card.setObjectName("resultCard")
        result_card.setFrameShape(QFrame.StyledPanel)
        result_layout = QHBoxLayout(result_card)
        result_layout.setSpacing(30)

        # 得分区域（带标注）
        score_widget = QWidget()
        score_layout = QVBoxLayout(score_widget)
        score_layout.setAlignment(Qt.AlignCenter)
        score_layout.addWidget(QLabel("后门得分"), alignment=Qt.AlignCenter)
        self.score_label = QLabel("--")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setStyleSheet("font-size: 25px; font-weight: bold; color: #3498db;")
        score_layout.addWidget(self.score_label)
        result_layout.addWidget(score_widget)

        # 类别对
        pair_widget = QWidget()
        pair_layout = QVBoxLayout(pair_widget)
        pair_layout.setAlignment(Qt.AlignCenter)
        pair_layout.addWidget(QLabel("目标 / 源类别"), alignment=Qt.AlignCenter)
        self.class_pair_label = QLabel("-- / --")
        self.class_pair_label.setAlignment(Qt.AlignCenter)
        self.class_pair_label.setStyleSheet("font-size: 25px; color: #ecf0f1;")
        pair_layout.addWidget(self.class_pair_label)
        result_layout.addWidget(pair_widget)

        # 结论
        verdict_widget = QWidget()
        verdict_layout = QVBoxLayout(verdict_widget)
        verdict_layout.setAlignment(Qt.AlignCenter)
        verdict_layout.addWidget(QLabel("检测结果"), alignment=Qt.AlignCenter)
        self.verdict_label = QLabel("等待检测")
        self.verdict_label.setAlignment(Qt.AlignCenter)
        self.verdict_label.setStyleSheet("font-size: 25px; font-weight: bold; color: #95a5a6;")
        verdict_layout.addWidget(self.verdict_label)
        result_layout.addWidget(verdict_widget)

        main_layout.addWidget(result_card)

        # ---------- 5. 控制台输出 ----------
        console_label = QLabel("📟 检测日志")
        console_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #3498db;")
        main_layout.addWidget(console_label)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 10))
        self.output_text.setMinimumHeight(200)
        main_layout.addWidget(self.output_text)

        # ---------- 状态栏 ----------
        self.status_label = QLabel("⚡ 就绪")
        self.status_label.setStyleSheet("color: #7f8c8d;")
        main_layout.addWidget(self.status_label, alignment=Qt.AlignRight)

        # 线程相关
        self.thread = None
        self.worker = None

        # 监听目录输入
        self.model_dir_edit.textChanged.connect(self.check_directory)

    # ---------- 界面功能 ----------
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
        # 禁用控件
        self.start_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.output_text.clear()
        self.clean_img_label.clear()
        self.trigger_img_label.clear()
        self.score_label.setText("--")
        self.class_pair_label.setText("-- / --")
        self.verdict_label.setText("检测中...")
        self.verdict_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #3498db;")
        self.progress_bar.setVisible(True)
        self.status_label.setText("🔮 检测中，请稍候...")

        model_dir = self.model_dir_edit.text()
        clean_dir = os.path.join(model_dir, "clean-example-data")
        if not os.path.isdir(clean_dir):
            clean_dir = os.path.join(model_dir, "example_data")

        self.thread = QThread()
        self.worker = DetectionWorker(model_dir, clean_dir)
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

        # 显示干净图像（保持比例并居中）
        if clean_img_path and os.path.exists(clean_img_path) and trigger_img_path and os.path.exists(trigger_img_path):
            # 1. 加载两张图片为 QImage（便于像素操作）
            clean_img = QImage(clean_img_path)
            trigger_img = QImage(trigger_img_path)

            # 2. 确保两张图片尺寸一致（将 trigger 缩放到 clean 的尺寸）
            if clean_img.size() != trigger_img.size():
                trigger_img = trigger_img.scaled(clean_img.size(), Qt.KeepAspectRatioByExpanding)

            # 3. 创建结果图像（与 clean_img 同尺寸，格式为 ARGB32）
            result_img = QImage(clean_img.size(), QImage.Format_ARGB32)

            # 4. 逐像素相加（饱和处理）
            for x in range(clean_img.width()):
                for y in range(clean_img.height()):
                    # 获取两个图像的像素值（ARGB）
                    c_pixel = clean_img.pixel(x, y)
                    t_pixel = trigger_img.pixel(x, y)

                    # 提取各通道值
                    c_r = qRed(c_pixel)
                    c_g = qGreen(c_pixel)
                    c_b = qBlue(c_pixel)
                    c_a = qAlpha(c_pixel)

                    t_r = qRed(t_pixel)
                    t_g = qGreen(t_pixel)
                    t_b = qBlue(t_pixel)
                    t_a = qAlpha(t_pixel)

                    # 相加并饱和（确保不超过255）
                    r = min(255, c_r + t_r)
                    g = min(255, c_g + t_g)
                    b = min(255, c_b + t_b)
                    a = min(255, c_a + t_a)  # 透明度也可相加，或改为 max(c_a, t_a) 等其他策略

                    # 设置结果像素
                    result_img.setPixel(x, y, qRgba(r, g, b, a))

            # 5. 将 QImage 转为 QPixmap
            pixmap = QPixmap.fromImage(result_img)

            # 6. 缩放以适应标签并显示
            label_size = self.clean_img_label.size()
            scaled_pixmap = pixmap.scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.clean_img_label.setPixmap(scaled_pixmap)

        else:
            self.clean_img_label.setText("无可用图像或触发器图像")

        # 显示触发器图像
        if trigger_img_path and os.path.exists(trigger_img_path):
            pixmap = QPixmap(trigger_img_path)
            label_size = self.trigger_img_label.size()
            scaled_pixmap = pixmap.scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.trigger_img_label.setPixmap(scaled_pixmap)
        else:
            self.trigger_img_label.setText("未生成触发器")

        # 更新结果卡片
        self.score_label.setText(f"{score:.4f}")
        self.class_pair_label.setText(f"{best_pair[0]} / {best_pair[1]}")

        if score > 0.6:
            verdict = "⚠️ 可能存在后门"
            color = "#e67e22"  # 橙色警告
        else:
            verdict = "✅ 可能安全"
            color = "#2ecc71"  # 翠绿安全
        self.verdict_label.setText(verdict)
        self.verdict_label.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {color};")

        # 日志输出摘要
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
        self.verdict_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #e74c3c;")

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

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.worker.stop()
            self.thread.quit()
            self.thread.wait(2000)
        event.accept()


# ==================== 启动 ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())