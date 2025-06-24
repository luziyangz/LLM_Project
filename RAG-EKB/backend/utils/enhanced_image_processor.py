import pytesseract
import easyocr
from PIL import Image
import io
import base64
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from openai import OpenAI
import os

@dataclass
class ImageContent:
    """图像内容数据类"""
    ocr_text: str
    description: str
    confidence: float
    language: str
    bbox: tuple
    image_format: str
    size: tuple

class EnhancedImageProcessor:
    """增强的图像处理器，支持OCR和AI视觉理解"""
    
    def __init__(self, openai_config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        
        # 初始化EasyOCR
        try:
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'])  # 支持中英文
            self.logger.info("EasyOCR初始化成功")
        except Exception as e:
            self.logger.warning(f"EasyOCR初始化失败: {e}")
            self.ocr_reader = None
        
        # 初始化OpenAI客户端（用于图像理解）
        if openai_config:
            try:
                self.openai_client = OpenAI(
                    api_key=openai_config.get('api_key'),
                    base_url=openai_config.get('base_url')
                )
                self.vision_model = openai_config.get('vision_model', 'gpt-4-vision-preview')
                self.logger.info("OpenAI视觉模型初始化成功")
            except Exception as e:
                self.logger.warning(f"OpenAI视觉模型初始化失败: {e}")
                self.openai_client = None
        else:
            self.openai_client = None
    
    def process_image_comprehensive(self, image_data: bytes, 
                                  image_format: str = 'PNG') -> ImageContent:
        """综合处理图像：OCR + AI视觉理解"""
        try:
            # 转换为PIL图像
            image = Image.open(io.BytesIO(image_data))
            
            # OCR文字识别
            ocr_text = self._extract_text_ocr(image)
            
            # AI视觉理解
            description = self._generate_image_description(image_data, image_format)
            
            # 计算置信度（基于OCR结果和描述质量）
            confidence = self._calculate_confidence(ocr_text, description)
            
            return ImageContent(
                ocr_text=ocr_text,
                description=description,
                confidence=confidence,
                language='zh-cn' if self._contains_chinese(ocr_text) else 'en',
                bbox=(0, 0, image.width, image.height),
                image_format=image_format,
                size=(image.width, image.height)
            )
            
        except Exception as e:
            self.logger.error(f"图像处理失败: {e}")
            return ImageContent(
                ocr_text="",
                description="图像处理失败",
                confidence=0.0,
                language='unknown',
                bbox=(0, 0, 0, 0),
                image_format=image_format,
                size=(0, 0)
            )
    
    def _extract_text_ocr(self, image: Image.Image) -> str:
        """使用OCR提取图像中的文字"""
        ocr_results = []
        
        # 尝试使用EasyOCR
        if self.ocr_reader:
            try:
                # 转换PIL图像为numpy数组
                import numpy as np
                img_array = np.array(image)
                
                results = self.ocr_reader.readtext(img_array)
                easyocr_text = ' '.join([result[1] for result in results if result[2] > 0.5])
                if easyocr_text.strip():
                    ocr_results.append(easyocr_text)
                    
            except Exception as e:
                self.logger.warning(f"EasyOCR处理失败: {e}")
        
        # 尝试使用Tesseract
        try:
            # 配置Tesseract支持中英文
            tesseract_text = pytesseract.image_to_string(
                image, 
                lang='chi_sim+eng',
                config='--psm 6'
            )
            if tesseract_text.strip():
                ocr_results.append(tesseract_text.strip())
                
        except Exception as e:
            self.logger.warning(f"Tesseract处理失败: {e}")
        
        # 合并OCR结果
        return '\n'.join(ocr_results) if ocr_results else ""
    
    def _generate_image_description(self, image_data: bytes, 
                                  image_format: str) -> str:
        """使用AI模型生成图像描述"""
        if not self.openai_client:
            return "AI视觉理解功能未启用"
        
        try:
            # 将图像转换为base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "请详细描述这张图片的内容，包括：1）主要对象和场景 2）文字内容（如果有） 3）图表或数据（如果有） 4）整体布局和设计特点。请用中文回答。"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format.lower()};base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"AI图像描述生成失败: {e}")
            return "AI图像描述生成失败"
    
    def _calculate_confidence(self, ocr_text: str, description: str) -> float:
        """计算处理结果的置信度"""
        confidence = 0.0
        
        # OCR文字长度贡献
        if ocr_text.strip():
            confidence += min(len(ocr_text) / 100, 0.4)  # 最多贡献0.4
        
        # 描述质量贡献
        if description and "失败" not in description and "未启用" not in description:
            confidence += min(len(description) / 200, 0.6)  # 最多贡献0.6
        
        return min(confidence, 1.0)
    
    def _contains_chinese(self, text: str) -> bool:
        """检测文本是否包含中文"""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False