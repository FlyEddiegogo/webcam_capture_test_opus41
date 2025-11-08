#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光學檢測系統 - Webcam擷取測試程式
專案位置: D:\Dev\OpticalInspect
建立日期: 2025-11-07
版本: 1.0.0
作者: Eddie
描述: 用於測試webcam連接、影像擷取和基本物件檢測功能
"""

import cv2  # OpenCV影像處理庫
import numpy as np  # 數值運算庫
import datetime  # 日期時間處理
import os  # 作業系統介面
import sys  # 系統相關功能
import time  # 時間控制
import threading  # 多執行緒支援
from pathlib import Path  # 路徑處理
import logging  # 日誌記錄

# ========================== 全域設定 ==========================
# 設定日誌系統
logging.basicConfig(
    level=logging.INFO,  # 日誌等級設為INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日誌格式
    handlers=[
        logging.FileHandler('webcam_test.log', encoding='utf-8'),  # 寫入檔案
        logging.StreamHandler()  # 輸出到控制台
    ]
)
logger = logging.getLogger(__name__)  # 取得日誌記錄器

# 建立輸出目錄結構
OUTPUT_DIR = Path("output")  # 主輸出目錄
CAPTURE_DIR = OUTPUT_DIR / "captures"  # 擷取影像目錄
VIDEO_DIR = OUTPUT_DIR / "videos"  # 錄影檔案目錄

# 建立必要的目錄（如果不存在）
for dir_path in [OUTPUT_DIR, CAPTURE_DIR, VIDEO_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)  # 遞迴建立目錄
    logger.info(f"確認目錄存在: {dir_path}")  # 記錄目錄建立

# ========================== WebcamCapture類別 ==========================
class WebcamCapture:
    """
    Webcam擷取類別 - 處理所有webcam相關操作
    包含初始化、擷取、顯示、儲存等功能
    """
    
    def __init__(self, camera_index=0, resolution=(1280, 720)):
        """
        初始化Webcam擷取器
        
        參數:
            camera_index (int): 攝影機索引，預設為0（第一台攝影機）
            resolution (tuple): 解析度設定，預設為720p (1280x720)
        """
        self.camera_index = camera_index  # 儲存攝影機索引
        self.resolution = resolution  # 儲存解析度設定
        self.cap = None  # VideoCapture物件初始化為None
        self.is_running = False  # 執行狀態旗標
        self.frame_count = 0  # 影格計數器
        self.fps = 0  # 每秒影格數
        self.last_frame = None  # 儲存最後一個影格
        self.recording = False  # 錄影狀態旗標
        self.video_writer = None  # 影片寫入器
        
        # 效能監控變數
        self.fps_start_time = time.time()  # FPS計算起始時間
        self.fps_frame_count = 0  # FPS計算用的影格計數
        
        logger.info(f"初始化WebcamCapture - 攝影機:{camera_index}, 解析度:{resolution}")
    
    def initialize_camera(self):
        """
        初始化並設定攝影機
        
        返回:
            bool: 成功返回True，失敗返回False
        """
        try:
            # 嘗試開啟攝影機（使用DirectShow在Windows上）
            if sys.platform == "win32":
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                logger.info("使用DirectShow API開啟攝影機")
            else:
                self.cap = cv2.VideoCapture(self.camera_index)
                logger.info("使用預設API開啟攝影機")
            
            # 檢查攝影機是否成功開啟
            if not self.cap.isOpened():
                logger.error(f"無法開啟攝影機 {self.camera_index}")
                return False
            
            # 設定攝影機參數
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])  # 設定寬度
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])  # 設定高度
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # 設定目標FPS為30
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 設定緩衝區大小為1，減少延遲
            
            # 讀取並驗證實際設定值
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 實際寬度
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 實際高度
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # 實際FPS
            
            logger.info(f"攝影機設定成功:")
            logger.info(f"  解析度: {actual_width}x{actual_height}")
            logger.info(f"  FPS: {actual_fps}")
            
            # 測試讀取一個影格
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("無法讀取測試影格")
                return False
            
            logger.info("攝影機初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"攝影機初始化失敗: {str(e)}")
            return False
    
    def calculate_fps(self):
        """
        計算實際FPS（每秒影格數）
        使用滑動視窗方法計算平均FPS
        """
        self.fps_frame_count += 1  # 增加影格計數
        
        # 每30個影格計算一次FPS
        if self.fps_frame_count >= 30:
            current_time = time.time()  # 取得當前時間
            elapsed_time = current_time - self.fps_start_time  # 計算經過時間
            
            if elapsed_time > 0:
                self.fps = self.fps_frame_count / elapsed_time  # 計算FPS
            
            # 重置計數器
            self.fps_frame_count = 0
            self.fps_start_time = current_time
    
    def preprocess_frame(self, frame):
        """
        預處理影格 - 用於後續的物件檢測
        
        參數:
            frame: 原始影格
            
        返回:
            dict: 包含各種處理後的影像
        """
        processed = {}  # 建立處理結果字典
        
        try:
            # 1. 轉換為灰階影像（物件檢測常用）
            processed['gray'] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 2. 高斯模糊（減少雜訊）
            processed['blur'] = cv2.GaussianBlur(frame, (5, 5), 0)
            
            # 3. 邊緣檢測（Canny演算法）
            gray_blur = cv2.GaussianBlur(processed['gray'], (5, 5), 0)
            processed['edges'] = cv2.Canny(gray_blur, 50, 150)
            
            # 4. 二值化（用於物件分割）
            _, processed['binary'] = cv2.threshold(
                processed['gray'],  # 輸入影像
                127,  # 閾值
                255,  # 最大值
                cv2.THRESH_BINARY  # 二值化類型
            )
            
            # 5. 自適應閾值（對光照變化更穩定）
            processed['adaptive'] = cv2.adaptiveThreshold(
                processed['gray'],  # 輸入影像
                255,  # 最大值
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 自適應方法
                cv2.THRESH_BINARY,  # 閾值類型
                11,  # 區塊大小
                2  # 常數C
            )
            
            # 6. 形態學操作（開運算-去除小雜點）
            kernel = np.ones((3, 3), np.uint8)  # 建立3x3核心
            processed['morphology'] = cv2.morphologyEx(
                processed['binary'],  # 輸入影像
                cv2.MORPH_OPEN,  # 開運算
                kernel,  # 結構元素
                iterations=2  # 迭代次數
            )
            
        except Exception as e:
            logger.error(f"影格預處理失敗: {str(e)}")
            
        return processed
    
    def detect_objects(self, frame):
        """
        簡單的物件檢測（使用輪廓檢測）
        這是基礎版本，可以擴展為YOLO等深度學習方法
        
        參數:
            frame: 輸入影格
            
        返回:
            frame_with_objects: 標記了物件的影格
            objects: 檢測到的物件列表
        """
        objects = []  # 物件列表
        frame_with_objects = frame.copy()  # 複製原始影格
        
        try:
            # 預處理影格
            processed = self.preprocess_frame(frame)
            
            # 使用Canny邊緣檢測結果找輪廓
            contours, hierarchy = cv2.findContours(
                processed['edges'],  # 輸入影像
                cv2.RETR_EXTERNAL,  # 只檢測外部輪廓
                cv2.CHAIN_APPROX_SIMPLE  # 壓縮輪廓點
            )
            
            # 處理每個輪廓
            for i, contour in enumerate(contours):
                # 計算輪廓面積
                area = cv2.contourArea(contour)
                
                # 過濾太小的輪廓（雜訊）
                if area < 500:  # 面積閾值：500像素
                    continue
                
                # 計算邊界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 計算輪廓中心
                M = cv2.moments(contour)  # 計算矩
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])  # 中心x座標
                    cy = int(M["m01"] / M["m00"])  # 中心y座標
                else:
                    cx, cy = x + w // 2, y + h // 2  # 使用邊界框中心
                
                # 儲存物件資訊
                obj_info = {
                    'id': i,  # 物件ID
                    'bbox': (x, y, w, h),  # 邊界框
                    'center': (cx, cy),  # 中心點
                    'area': area,  # 面積
                    'contour': contour  # 輪廓點
                }
                objects.append(obj_info)
                
                # 在影像上繪製檢測結果
                # 繪製邊界框（綠色）
                cv2.rectangle(
                    frame_with_objects,  # 目標影像
                    (x, y),  # 左上角
                    (x + w, y + h),  # 右下角
                    (0, 255, 0),  # 顏色（綠色）
                    2  # 線寬
                )
                
                # 繪製中心點（紅色圓圈）
                cv2.circle(
                    frame_with_objects,  # 目標影像
                    (cx, cy),  # 中心點
                    5,  # 半徑
                    (0, 0, 255),  # 顏色（紅色）
                    -1  # 填充
                )
                
                # 添加標籤
                label = f"Obj_{i}: {int(area)}"  # 物件標籤
                cv2.putText(
                    frame_with_objects,  # 目標影像
                    label,  # 文字內容
                    (x, y - 10),  # 位置（在邊界框上方）
                    cv2.FONT_HERSHEY_SIMPLEX,  # 字型
                    0.5,  # 字體大小
                    (0, 255, 0),  # 顏色（綠色）
                    1,  # 線寬
                    cv2.LINE_AA  # 抗鋸齒
                )
            
        except Exception as e:
            logger.error(f"物件檢測失敗: {str(e)}")
        
        return frame_with_objects, objects
    
    def add_overlay_info(self, frame):
        """
        在影格上添加資訊覆蓋層
        
        參數:
            frame: 輸入影格
            
        返回:
            frame: 添加了資訊的影格
        """
        try:
            # 取得影格尺寸
            height, width = frame.shape[:2]
            
            # 建立半透明覆蓋層
            overlay = frame.copy()
            
            # 添加頂部資訊條（黑色半透明背景）
            cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
            
            # 添加時間戳記
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                overlay,  # 目標影像
                f"Time: {timestamp}",  # 時間文字
                (10, 25),  # 位置
                cv2.FONT_HERSHEY_SIMPLEX,  # 字型
                0.6,  # 字體大小
                (0, 255, 0),  # 顏色（綠色）
                1,  # 線寬
                cv2.LINE_AA  # 抗鋸齒
            )
            
            # 添加FPS資訊
            cv2.putText(
                overlay,  # 目標影像
                f"FPS: {self.fps:.1f}",  # FPS文字
                (width - 150, 25),  # 位置（右側）
                cv2.FONT_HERSHEY_SIMPLEX,  # 字型
                0.6,  # 字體大小
                (0, 255, 255),  # 顏色（黃色）
                1,  # 線寬
                cv2.LINE_AA  # 抗鋸齒
            )
            
            # 添加影格計數
            cv2.putText(
                overlay,  # 目標影像
                f"Frame: {self.frame_count}",  # 影格計數文字
                (width // 2 - 50, 25),  # 位置（中間）
                cv2.FONT_HERSHEY_SIMPLEX,  # 字型
                0.6,  # 字體大小
                (255, 255, 255),  # 顏色（白色）
                1,  # 線寬
                cv2.LINE_AA  # 抗鋸齒
            )
            
            # 如果正在錄影，添加錄影指示
            if self.recording:
                cv2.circle(overlay, (width - 30, 20), 10, (0, 0, 255), -1)  # 紅色圓圈
                cv2.putText(
                    overlay,  # 目標影像
                    "REC",  # 錄影文字
                    (width - 70, 25),  # 位置
                    cv2.FONT_HERSHEY_SIMPLEX,  # 字型
                    0.6,  # 字體大小
                    (0, 0, 255),  # 顏色（紅色）
                    2,  # 線寬
                    cv2.LINE_AA  # 抗鋸齒
                )
            
            # 混合原始影像和覆蓋層（透明度0.8）
            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
            
        except Exception as e:
            logger.error(f"添加覆蓋資訊失敗: {str(e)}")
        
        return frame
    
    def capture_image(self):
        """
        擷取並儲存當前影格
        
        返回:
            str: 儲存的檔案路徑，失敗返回None
        """
        if self.last_frame is None:
            logger.warning("沒有可用的影格進行擷取")
            return None
        
        try:
            # 產生檔案名稱（包含時間戳記）
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            filepath = CAPTURE_DIR / filename
            
            # 儲存影像
            success = cv2.imwrite(str(filepath), self.last_frame)
            
            if success:
                logger.info(f"影像擷取成功: {filepath}")
                return str(filepath)
            else:
                logger.error("影像儲存失敗")
                return None
                
        except Exception as e:
            logger.error(f"影像擷取錯誤: {str(e)}")
            return None
    
    def start_recording(self):
        """
        開始錄影
        
        返回:
            bool: 成功返回True，失敗返回False
        """
        if self.recording:
            logger.warning("已經在錄影中")
            return False
        
        try:
            # 產生檔案名稱
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_{timestamp}.avi"
            filepath = VIDEO_DIR / filename
            
            # 取得影格尺寸
            if self.last_frame is not None:
                height, width = self.last_frame.shape[:2]
            else:
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 設定編碼器（使用XVID）
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            
            # 建立VideoWriter物件
            self.video_writer = cv2.VideoWriter(
                str(filepath),  # 檔案路徑
                fourcc,  # 編碼器
                20.0,  # FPS
                (width, height)  # 影格尺寸
            )
            
            if self.video_writer.isOpened():
                self.recording = True
                logger.info(f"開始錄影: {filepath}")
                return True
            else:
                logger.error("無法開啟VideoWriter")
                return False
                
        except Exception as e:
            logger.error(f"開始錄影失敗: {str(e)}")
            return False
    
    def stop_recording(self):
        """
        停止錄影
        """
        if not self.recording:
            return
        
        try:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()  # 釋放VideoWriter
                self.video_writer = None
            logger.info("錄影已停止")
            
        except Exception as e:
            logger.error(f"停止錄影失敗: {str(e)}")
    
    def run(self):
        """
        主執行迴圈 - 持續擷取和顯示影像
        """
        if not self.initialize_camera():
            logger.error("攝影機初始化失敗，無法執行")
            return
        
        self.is_running = True
        logger.info("開始執行webcam擷取迴圈")
        
        # 建立所有需要的視窗
        cv2.namedWindow('Webcam原始影像', cv2.WINDOW_NORMAL)  # 原始影像視窗
        cv2.namedWindow('物件檢測結果', cv2.WINDOW_NORMAL)  # 檢測結果視窗
        cv2.namedWindow('影像處理', cv2.WINDOW_NORMAL)  # 處理結果視窗
        
        # 設定視窗大小
        cv2.resizeWindow('Webcam原始影像', 640, 480)
        cv2.resizeWindow('物件檢測結果', 640, 480)
        cv2.resizeWindow('影像處理', 640, 480)
        
        logger.info("按鍵說明:")
        logger.info("  q/ESC - 結束程式")
        logger.info("  s - 擷取影像")
        logger.info("  r - 開始/停止錄影")
        logger.info("  p - 暫停/繼續")
        logger.info("  d - 顯示檢測結果")
        
        paused = False  # 暫停旗標
        show_detection = True  # 顯示檢測旗標
        
        while self.is_running:
            try:
                # 如果暫停，只處理按鍵事件
                if paused:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('p'):
                        paused = False
                        logger.info("繼續執行")
                    elif key in [ord('q'), 27]:  # q或ESC
                        self.is_running = False
                    continue
                
                # 讀取影格
                ret, frame = self.cap.read()
                
                # 檢查影格是否有效
                if not ret or frame is None:
                    logger.warning("無法讀取影格，嘗試重新連接...")
                    time.sleep(0.1)
                    continue
                
                # 更新影格計數和FPS
                self.frame_count += 1
                self.calculate_fps()
                
                # 儲存當前影格
                self.last_frame = frame.copy()
                
                # 添加覆蓋資訊
                display_frame = self.add_overlay_info(frame.copy())
                
                # 物件檢測
                if show_detection:
                    detected_frame, objects = self.detect_objects(frame)
                    cv2.imshow('物件檢測結果', detected_frame)
                    
                    # 如果檢測到物件，記錄資訊
                    if objects:
                        logger.debug(f"檢測到 {len(objects)} 個物件")
                
                # 顯示處理結果
                processed = self.preprocess_frame(frame)
                if 'edges' in processed:
                    # 將邊緣檢測轉為3通道以便顯示
                    edges_color = cv2.cvtColor(processed['edges'], cv2.COLOR_GRAY2BGR)
                    cv2.imshow('影像處理', edges_color)
                
                # 顯示原始影像
                cv2.imshow('Webcam原始影像', display_frame)
                
                # 如果正在錄影，寫入影格
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)
                
                # 處理按鍵事件
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # q或ESC鍵退出
                    logger.info("收到退出指令")
                    self.is_running = False
                    
                elif key == ord('s'):  # s鍵擷取影像
                    filepath = self.capture_image()
                    if filepath:
                        logger.info(f"影像已儲存: {filepath}")
                        
                elif key == ord('r'):  # r鍵開始/停止錄影
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                        
                elif key == ord('p'):  # p鍵暫停
                    paused = True
                    logger.info("暫停執行")
                    
                elif key == ord('d'):  # d鍵切換檢測顯示
                    show_detection = not show_detection
                    logger.info(f"物件檢測顯示: {'開啟' if show_detection else '關閉'}")
                
            except Exception as e:
                logger.error(f"主迴圈錯誤: {str(e)}")
                continue
        
        # 清理資源
        self.cleanup()
    
    def cleanup(self):
        """
        清理資源，釋放攝影機和關閉視窗
        """
        logger.info("開始清理資源...")
        
        # 停止錄影
        if self.recording:
            self.stop_recording()
        
        # 釋放攝影機
        if self.cap:
            self.cap.release()
            logger.info("攝影機已釋放")
        
        # 關閉所有OpenCV視窗
        cv2.destroyAllWindows()
        logger.info("所有視窗已關閉")
        
        logger.info("清理完成")

# ========================== 輔助函數 ==========================
def test_camera_availability():
    """
    測試系統中可用的攝影機
    
    返回:
        list: 可用的攝影機索引列表
    """
    available_cameras = []  # 可用攝影機列表
    
    logger.info("開始掃描可用攝影機...")
    
    # 測試前10個攝影機索引
    for i in range(10):
        try:
            # 嘗試開啟攝影機
            cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                # 嘗試讀取一個影格
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # 取得攝影機資訊
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    logger.info(f"找到攝影機 {i}: {width}x{height} @ {fps}fps")
                    available_cameras.append(i)
                
                cap.release()  # 釋放攝影機
                
        except Exception as e:
            logger.debug(f"測試攝影機 {i} 失敗: {str(e)}")
    
    logger.info(f"掃描完成，找到 {len(available_cameras)} 個可用攝影機")
    return available_cameras

def verify_opencv_installation():
    """
    驗證OpenCV安裝和功能
    
    返回:
        bool: 驗證成功返回True，失敗返回False
    """
    logger.info("開始驗證OpenCV安裝...")
    
    try:
        # 檢查OpenCV版本
        version = cv2.__version__
        logger.info(f"OpenCV版本: {version}")
        
        # 檢查是否有GPU支援（CUDA）
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        if gpu_count > 0:
            logger.info(f"偵測到 {gpu_count} 個CUDA裝置")
            for i in range(gpu_count):
                cv2.cuda.setDevice(i)
                device_info = cv2.cuda.getDevice()
                logger.info(f"  GPU {i}: 裝置 {device_info}")
        else:
            logger.info("未偵測到CUDA裝置，使用CPU模式")
        
        # 測試基本功能
        # 建立測試影像
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:50, :50] = [255, 0, 0]  # 藍色
        test_img[:50, 50:] = [0, 255, 0]  # 綠色
        test_img[50:, :50] = [0, 0, 255]  # 紅色
        test_img[50:, 50:] = [255, 255, 255]  # 白色
        
        # 測試影像處理功能
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  # 轉灰階
        edges = cv2.Canny(gray, 50, 150)  # 邊緣檢測
        
        logger.info("OpenCV基本功能測試通過")
        
        # 測試編碼器
        codecs = ['XVID', 'MJPG', 'X264', 'MP4V']
        available_codecs = []
        
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            if fourcc != -1:
                available_codecs.append(codec)
        
        logger.info(f"可用的視訊編碼器: {', '.join(available_codecs)}")
        
        return True
        
    except Exception as e:
        logger.error(f"OpenCV驗證失敗: {str(e)}")
        return False

# ========================== 主程式 ==========================
def main():
    """
    主程式進入點
    """
    logger.info("="*60)
    logger.info("光學檢測系統 - Webcam測試程式")
    logger.info("="*60)
    
    # 驗證OpenCV安裝
    if not verify_opencv_installation():
        logger.error("OpenCV驗證失敗，請檢查安裝")
        sys.exit(1)
    
    # 測試可用攝影機
    available_cameras = test_camera_availability()
    
    if not available_cameras:
        logger.error("未找到可用的攝影機")
        sys.exit(1)
    
    # 選擇攝影機
    if len(available_cameras) == 1:
        camera_index = available_cameras[0]
        logger.info(f"使用唯一可用的攝影機: {camera_index}")
    else:
        logger.info(f"可用攝影機: {available_cameras}")
        # 預設使用第一個可用的攝影機
        camera_index = available_cameras[0]
        logger.info(f"使用預設攝影機: {camera_index}")
    
    # 建立WebcamCapture實例
    webcam = WebcamCapture(camera_index=camera_index, resolution=(1280, 720))
    
    try:
        # 執行主程式
        webcam.run()
        
    except KeyboardInterrupt:
        logger.info("收到鍵盤中斷訊號")
        
    except Exception as e:
        logger.error(f"程式執行錯誤: {str(e)}")
        
    finally:
        # 確保資源被正確釋放
        webcam.cleanup()
        logger.info("程式結束")

if __name__ == "__main__":
    main()
