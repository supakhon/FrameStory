import json
import os
import time

from dotenv import load_dotenv
from PIL import Image
from google import genai
from google.genai import types

class FrameStoryEngine:
    def __init__(self, api_key: str = None, model_name: str = "gemini-3.8-flash"):
        """
        ระบบ FrameStory Engine สำหรับแปลงรูปภาพเป็นนิยายสไตล์ต่อเนื่อง
        """
        # 1. ตั้งค่า Gemini API Client
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("กรุณากำหนด GEMINI_API_KEY ก่อนเริ่มใช้งานครับ")
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.history = []  # โครงสร้างเก็บประวัติ [{ "step": 1, "image_path": "...", "story": "..." }]

        # 2. System Instruction คุมสำนวนภาษานิยาย (กระชับ + ใส่ใจบรรยากาศ)
        self.system_instruction = (
            "คุณคือนักเขียนนิยายสไตล์ Slice of Life / Visual Novel ที่เชี่ยวชาญการเล่าเรื่องจากภาพ "
            "หลักการเขียนที่ต้องยึดถืออย่างเคร่งครัด:\n"
            "1. ใช้ภาษากระชับ อ่านง่าย ไม่เยิ่นเย้อ แต่ต้องใส่ใจบรรยายบรรยากาศ แสง เงา สภาพแวดล้อม และอารมณ์ในภาพ (Show, Don't Tell)\n"
            "2. แต่งเนื้อเรื่องให้มีความยาวประมาณ 3–5 ประโยคต่อรูปภาพ\n"
            "3. หากมีประวัติเนื้อเรื่องย้อนหลัง ให้แต่งเนื้อเรื่องต่อจากเดิมอย่างสอดคล้อง ลื่นไหล และรักษาพล็อตเรื่องให้เชื่อมโยงกันเป็นเรื่องเดียวกัน"
        )

    def generate_next_chapter(self, image_path: str, max_retries: int = 5) -> str:
        """
        เจนคำบรรยายภาพถัดไป พร้อมระบบ Exponential Backoff รองรับ 503
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"ไม่พบไฟล์รูปภาพที่ตำแหน่ง: {image_path}")

        context_prompt = ""
        if self.history:
            context_prompt = "=== ประวัติเนื้อเรื่องที่ผ่านมา ===\n"
            for item in self.history:
                context_prompt += f"ฉากที่ {item['step']}: {item['story']}\n"
            context_prompt += "\n=== โจทย์ปัจจุบัน ===\nจงแต่งเรื่องราวสำหรับรูปภาพใหม่นี้ โดยเล่าเรื่องสืบเนื่องจากฉากที่ผ่านมา:"
        else:
            context_prompt = "จงเริ่มต้นเปิดเรื่องราวสำหรับรูปภาพแรกนี้:"

        with Image.open(image_path) as img:
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[img, context_prompt],
                        config=types.GenerateContentConfig(
                            system_instruction=self.system_instruction,
                            temperature=0.7
                        )
                    )

                    generated_story = response.text.strip()

                    step_number = len(self.history) + 1
                    record = {
                        "step": step_number,
                        "image_path": image_path,
                        "story": generated_story
                    }
                    self.history.append(record)
                    return generated_story

                except Exception as e:
                    error_msg = str(e)
                    if "503" in error_msg or "UNAVAILABLE" in error_msg:
                        if attempt < max_retries:
                            # คำนวณเวลารอแบบ Exponential Backoff: 3, 6, 12, 24 วินาที
                            wait_time = 3 * (2 ** (attempt - 1))
                            print(f"⚠️️ เซิร์ฟเวอร์ AI หนาแน่น (503) กำลังพยายามส่งใหม่ใน {wait_time} วินาที... (พยายามครั้งที่ {attempt}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            raise Exception(f"ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ AI ได้หลังจากลอง {max_retries} ครั้ง: {e}")
                    else:
                        raise e

    def edit_chapter_story(self, step: int, new_story: str) -> bool:
        """
        ฟีเจอร์ให้ผู้ใช้กดแก้ไขเนื้อเรื่องฉากนั้นๆ เองได้ ก่อนที่จะเจนฉากถัดไป
        """
        for item in self.history:
            if item["step"] == step:
                item["story"] = new_story.strip()
                return True
        return False

    def get_full_story(self) -> str:
        """
        ดึงเนื้อเรื่องทั้งหมดมาเรียงต่อกันเป็นเล่มนิยาย
        """
        full_text = []
        for item in self.history:
            full_text.append(f"[ฉากที่ {item['step']}]\n{item['story']}\n")
        return "\n".join(full_text)

    def save_to_json(self, output_path: str = "framestory_data.json"):
        """
        บันทึกโครงสร้างข้อมูลเป็นไฟล์ JSON ไว้ส่งต่อให้ระบบอื่น/UI
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        print(f"💾 บันทึกข้อมูลเรียบร้อยแล้วที่: {output_path}")


# ==========================================
# 🧪 ตัวอย่างการทดสอบใช้งาน (Test Run)
# ==========================================
if __name__ == "__main__":
    load_dotenv()

    try:
        engine = FrameStoryEngine()
        print("🚀 เริ่มต้นทดสอบ FrameStory Engine Prototype...\n")

        test_images = ["scene1.jpg", "scene2.jpg", "scene3.jpg"]

        for img_path in test_images:
            if os.path.exists(img_path):
                print(f"📸 กำลังประมวลผลรูปภาพ: {img_path} ...")
                story = engine.generate_next_chapter(img_path)
                print(f"📖 คำบรรยายที่ได้:\n{story}\n")
                print("-" * 50)
                
                # พัก 2 วินาทีก่อนประมวลผลรูปถัดไป เพื่อลดภาระเซิร์ฟเวอร์
                time.sleep(2)
            else:
                print(f"⚠️ ไม่พบไฟล์ {img_path}")

        if engine.history:
            print("\n📚 === เรื่องราวทั้งหมด (Full Story) ===")
            print(engine.get_full_story())
            engine.save_to_json("story_output.json")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")