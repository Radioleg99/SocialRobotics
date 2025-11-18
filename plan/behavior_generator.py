"""行为生成器：将动作描述转换为 Furhat API 调用"""
from typing import Optional, Tuple, Dict, Any
from furhat_realtime_api import AsyncFurhatClient


class BehaviorGenerator:
    """将信心等级和动作描述转换为 Furhat API 调用"""

    # 信心等级对应的前缀话术与动作描述（增强版：包含多模态）
    # 格式：(语言前缀, 基础手势, 手势表情, LED颜色)
    # 手势表情参考 Furhat Gestures 面板：BigSmile, Thoughtful, Oh, Smile 等
    CONFIDENCE_BEHAVIORS: Dict[str, Tuple[str, str, str, str]] = {
        "low": ("I'm not entirely sure, but", "slight head shake", "Oh", "yellow"),      # Oh 手势：不确定
        "medium": ("Let me think", "look straight", "Thoughtful", "blue"),               # Thoughtful 手势：思考中
        "high": ("I'm confident that", "nod head", "BigSmile", "green"),                 # BigSmile 手势：自信微笑
    }

    # 原始格式兼容（用于向后兼容）
    @staticmethod
    def _get_legacy_behavior(confidence: str) -> Tuple[str, str]:
        """获取旧版本的两元组格式（仅语言+手势）"""
        full = BehaviorGenerator.CONFIDENCE_BEHAVIORS.get(confidence,
            ("Let me think", "look straight", "Oh", "blue"))
        return (full[0], full[1])  # 只返回前两个元素

    def __init__(self, furhat_client: Optional[AsyncFurhatClient] = None):
        self.furhat = furhat_client
        self._thinking_mode = False

    def get_confidence_behavior(self, confidence: str) -> Tuple[str, str]:
        """获取信心等级对应的语言前缀和动作描述（向后兼容方法）"""
        if confidence not in self.CONFIDENCE_BEHAVIORS:
            confidence = "medium"  # 默认值
        return self._get_legacy_behavior(confidence)

    def get_full_confidence_behavior(self, confidence: str) -> Tuple[str, str, str, str]:
        """获取信心等级对应的完整多模态行为（语言、基础手势、手势表情、LED）"""
        if confidence not in self.CONFIDENCE_BEHAVIORS:
            confidence = "medium"  # 默认值
        return self.CONFIDENCE_BEHAVIORS[confidence]

    def set_thinking_mode(self, active: bool):
        """标记当前是否处于思考语音阶段"""
        self._thinking_mode = active

    def is_in_thinking_mode(self) -> bool:
        return self._thinking_mode

    async def perform_thinking_behavior(self, sequence_index: int = 0):
        """思考阶段的多模态行为（手势 + 手势表情 + LED）"""
        if not self.furhat:
            return

        # 思考阶段的行为循环
        gesture_cycle = ["look straight", "slight head shake"]
        expression_cycle = ["Thoughtful", "Oh"]  # 使用 Furhat Gestures 面板中的手势表情
        led_color = "#FFA500"  # 橙色表示正在思考

        gesture = gesture_cycle[sequence_index % len(gesture_cycle)]
        expression = expression_cycle[sequence_index % len(expression_cycle)]

        # 并发执行多个模态
        import asyncio
        await asyncio.gather(
            self.execute_gesture(gesture),
            self.execute_gesture_expression(expression),  # 使用手势表情方法
            self.execute_led_color_hex(led_color),  # 直接使用十六进制
            return_exceptions=True
        )

    async def execute_multimodal_behavior(self, confidence: str):
        """执行完整的多模态行为（基础手势+手势表情+LED）"""
        if not self.furhat:
            return

        prefix, gesture, expression, led_color = self.get_full_confidence_behavior(confidence)

        # 并发执行多个模态（提高响应速度）
        import asyncio
        tasks = []

        # 1. 基础手势（如 Nod, Shake, look straight）
        tasks.append(self.execute_gesture(gesture))

        # 2. 手势表情（如 BigSmile, Thoughtful, Oh）
        tasks.append(self.execute_gesture_expression(expression))

        # 3. LED 颜色
        tasks.append(self.execute_led_color(led_color))

        # 等待所有模态执行完成
        await asyncio.gather(*tasks, return_exceptions=True)

    async def execute_gesture(self, gesture_description: str):
        """根据动作描述执行 Furhat 动作"""
        if not self.furhat:
            return

        # 将动作描述映射到 Furhat API 调用
        gesture_map = {
            "slight head shake": self._shake_head_slightly,
            "轻微摇头": self._shake_head_slightly,  # 兼容旧描述
            "look straight": self._look_straight,
            "平视凝神": self._look_straight,  # 兼容旧描述
            "nod head": self._nod_head,
            "点头示意": self._nod_head,  # 兼容旧描述
        }

        gesture_func = gesture_map.get(gesture_description)
        if gesture_func:
            try:
                await gesture_func()
            except Exception as e:
                print(f"执行动作 {gesture_description} 时出错: {e}")

    async def execute_gesture_expression(self, expression: str):
        """执行手势表情（如 BigSmile, Thoughtful, Oh）"""
        if not self.furhat:
            return
        try:
            # 使用 request_gesture_start 执行手势表情
            # 这些是 Furhat Gestures 面板中显示的预设动作
            await self.furhat.request_gesture_start(
                name=expression,
                intensity=0.7,
                duration=1.0
            )
            print(f"[Multimodal] 执行手势表情: {expression}")
        except Exception as e:
            print(f"执行手势表情 {expression} 时出错: {e}")

    async def execute_led_color(self, color: str):
        """执行 LED 颜色变化"""
        if not self.furhat:
            return
        try:
            # LED 颜色映射到十六进制格式（根据 furhat-realtime-api 0.1.3）
            color_map = {
                "red": "#FF0000",
                "green": "#00FF00",
                "blue": "#0066FF",
                "yellow": "#FFC800",
                "purple": "#9600FF",
                "white": "#FFFFFF",
            }

            hex_color = color_map.get(color.lower(), "#0066FF")  # 默认蓝色

            # 使用正确的 API：request_led_set (根据 furhat-realtime-api 0.1.3 文档)
            await self.furhat.request_led_set(color=hex_color)
            print(f"[Multimodal] 设置 LED 颜色: {color} ({hex_color})")
        except Exception as e:
            print(f"设置 LED 颜色 {color} 时出错: {e}")

    async def execute_led_color_hex(self, hex_color: str):
        """直接使用十六进制颜色设置 LED"""
        if not self.furhat:
            return
        try:
            await self.furhat.request_led_set(color=hex_color)
            print(f"[Multimodal] 设置 LED 颜色: {hex_color}")
        except Exception as e:
            print(f"设置 LED 颜色 {hex_color} 时出错: {e}")

    async def _shake_head_slightly(self):
        """轻微摇头 - 使用 Furhat Gestures 面板中的 Shake"""
        if not self.furhat:
            return
        try:
            await self.furhat.request_gesture_start(
                name="Shake",  
                intensity=0.5,
                duration=0.8
            )
            print(f"[Multimodal] 执行手势: Shake")
        except Exception as e:
            print(f"执行摇头动作时出错: {e}")

    async def _look_straight(self):
        """平视 - 注视用户"""
        if not self.furhat:
            return
        try:
            await self.furhat.request_attend_user()
            print(f"[Multimodal] 执行手势: 注视用户")
        except Exception as e:
            print(f"执行注视动作时出错: {e}")

    async def _nod_head(self):
        """点头 - 使用 Furhat Gestures 面板中的 Nod"""
        if not self.furhat:
            return
        try:
            await self.furhat.request_gesture_start(
                name="Nod",  
                intensity=0.7,
                duration=0.6
            )
            print(f"[Multimodal] 执行手势: Nod")
        except Exception as e:
            print(f"执行点头动作时出错: {e}")

    def resolve_confidence(self, hint: Optional[str], word_count: int) -> str:
        """根据提示或词数解析信心等级"""
        if hint and hint.strip().lower() in self.CONFIDENCE_BEHAVIORS:
            return hint.strip().lower()
        return self._estimate_confidence_from_words(word_count)

    def infer_confidence_from_text(self, text: str) -> str:
        """根据文本内容推断信心等级"""
        text_lower = text.lower()
        # 根据前缀话术推断信心等级
        if "i'm not entirely sure" in text_lower or "i'm not sure" in text_lower:
            return "low"
        elif "i'm confident" in text_lower or "i'm certain" in text_lower:
            return "high"
        elif "let me think" in text_lower or "i think" in text_lower:
            return "medium"
        # 默认中等信心
        return "medium"

    @staticmethod
    def _estimate_confidence_from_words(word_count: int) -> str:
        """根据累计词数粗略估计信心等级"""
        if word_count < 25:
            return "low"
        if word_count < 60:
            return "medium"
        return "high"