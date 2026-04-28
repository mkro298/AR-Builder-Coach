import base64
import io
import json
import math
import os
import time
import uuid

from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


# =========================
# Configuration
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEFAULT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
DEFAULT_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
DEFAULT_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")


def maybe_get_openai_client() -> Optional[Any]:
    if not OPENAI_API_KEY or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None


# =========================
# Pydantic models
# =========================

class InventoryRequest(BaseModel):
    materials: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)


class PlanSummary(BaseModel):
    id: str
    name: str
    badge: str
    time_estimate: str
    description: str
    available: bool = True
    missing_parts: List[str] = Field(default_factory=list)


class StartSessionRequest(BaseModel):
    plan_id: str
    plan_name: Optional[str] = None
    inventory: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)


class RepeatStepRequest(BaseModel):
    replay_seconds: int = 5


class TogglePauseRequest(BaseModel):
    paused: Optional[bool] = None


class ZoomRequest(BaseModel):
    delta: float = 0.2


class FrameAnalyzeRequest(BaseModel):
    session_id: str
    image_base64: str
    image_width: int
    image_height: int


class RefWiringRequest(BaseModel):
    session_id: str


# =========================
# Domain models
# =========================

@dataclass
class StepDefinition:
    id: str
    title: str
    subtitle: str
    icon: str
    goal: str
    completion_check: Literal["led_region", "resistor_region", "nano_region", "scene_match", "generic"]
    expected_objects: List[str]
    overlay_hint: str
    target_region: Dict[str, float]
    instruction_prompt: str
    ai_reference_prompt: str


@dataclass
class PlanDefinition:
    id: str
    name: str
    badge: str
    time_estimate: str
    description: str
    required_materials: List[str]
    steps: List[StepDefinition]


@dataclass
class SessionState:
    id: str
    plan_id: str
    plan_name: str
    inventory: List[str]
    preferences: Dict[str, Any]
    steps: List[StepDefinition]
    current_step_index: int = 0
    paused: bool = False
    zoom: float = 1.0
    reference_cache: Dict[str, str] = field(default_factory=dict)
    last_analysis: Dict[str, Any] = field(default_factory=dict)
    vision_stability: Dict[str, Any] = field(default_factory=dict)
    gpt_eval_state: Dict[str, Any] = field(default_factory=dict)
    last_next_ts: float = 0.0


# =========================
# Seed plans
# =========================


def seed_plans() -> Dict[str, PlanDefinition]:
    return {
        "blinking_led": PlanDefinition(
            id="blinking_led",
            name="Blinking LED",
            badge="Beginner",
            time_estimate="10-12m",
            description="Build a basic blinking LED circuit with an Arduino, breadboard, LED, resistor, and jumper wires.",
            required_materials=["arduino nano", "breadboard", "led", "220Ω resistor", "jumper wire"],
            steps=[
                StepDefinition(
                    id="blinking_led_step_led",
                    title="Place the LED on the breadboard",
                    subtitle="Insert the LED so its legs go into two different rows.",
                    icon="💡",
                    goal="The LED should be clearly inserted on the breadboard.",
                    completion_check="generic",
                    expected_objects=["breadboard", "led"],
                    overlay_hint="Place the LED in the center area so the camera can see it clearly.",
                    target_region={"x": 0.50, "y": 0.38, "w": 0.24, "h": 0.18},
                    instruction_prompt="Explain to a beginner how to place an LED onto a breadboard using two different rows.",
                    ai_reference_prompt="A clean top-down beginner-friendly image showing one LED inserted into a breadboard.",
                ),
                StepDefinition(
                    id="blinking_led_step_resistor",
                    title="Add the resistor",
                    subtitle="Place the resistor so it connects in series with the LED.",
                    icon="🟫",
                    goal="The resistor should be placed beside the LED to form a simple LED circuit.",
                    completion_check="generic",
                    expected_objects=["breadboard", "led", "resistor"],
                    overlay_hint="Put the resistor beside the LED so the camera can capture both parts.",
                    target_region={"x": 0.43, "y": 0.55, "w": 0.32, "h": 0.16},
                    instruction_prompt="Explain to a beginner how to add a resistor in series with an LED on a breadboard.",
                    ai_reference_prompt="A top-down tutorial image showing an LED and resistor placed on a breadboard.",
                ),
                StepDefinition(
                    id="blinking_led_step_wires",
                    title="Connect the jumper wires",
                    subtitle="Use jumper wires to connect the breadboard circuit to the Arduino.",
                    icon="🔌",
                    goal="The jumper wires should connect the LED circuit to the Arduino Nano.",
                    completion_check="generic",
                    expected_objects=["arduino nano", "breadboard", "jumper wire"],
                    overlay_hint="Keep the Arduino and jumper wires in the center of the camera view.",
                    target_region={"x": 0.68, "y": 0.46, "w": 0.36, "h": 0.30},
                    instruction_prompt="Explain to a beginner how to connect an LED breadboard circuit to an Arduino using jumper wires.",
                    ai_reference_prompt="A top-down beginner electronics image showing jumper wires connecting an Arduino Nano to a breadboard LED circuit.",
                ),
                StepDefinition(
                    id="blinking_led_step_verify",
                    title="Verify the blinking LED setup",
                    subtitle="Check that the LED circuit is fully connected before testing.",
                    icon="✅",
                    goal="The Arduino, LED, resistor, and jumper wires should look like one complete blinking LED circuit.",
                    completion_check="scene_match",
                    expected_objects=["arduino nano", "breadboard", "led", "resistor"],
                    overlay_hint="Place the full circuit setup in the middle of the screen for a final check.",
                    target_region={"x": 0.50, "y": 0.50, "w": 0.80, "h": 0.72},
                    instruction_prompt="Explain how a beginner should visually verify a full blinking LED circuit before testing.",
                    ai_reference_prompt="A complete top-down reference showing an Arduino Nano, breadboard, LED, resistor, and jumper wires for a simple blinking LED circuit.",
                ),
            ],
        ),
        "button_led": PlanDefinition(
            id="button_led",
            name="Button-Controlled LED",
            badge="Beginner",
            time_estimate="12-15m",
            description="Build a circuit where a push button controls an LED.",
            required_materials=["arduino nano", "breadboard", "led", "220Ω resistor", "push button", "jumper wire"],
            steps=[
                StepDefinition(
                    id="button_led_step_button",
                    title="Place the push button on the breadboard",
                    subtitle="Insert the push button across the center gap of the breadboard.",
                    icon="🔘",
                    goal="The push button should sit securely across the breadboard gap.",
                    completion_check="generic",
                    expected_objects=["breadboard", "push button"],
                    overlay_hint="Place the button in the center region so the camera can track it.",
                    target_region={"x": 0.50, "y": 0.40, "w": 0.24, "h": 0.18},
                    instruction_prompt="Explain to a beginner how to place a push button across the center gap of a breadboard.",
                    ai_reference_prompt="A top-down beginner-friendly tutorial image showing a push button placed across the center gap of a breadboard.",
                ),
                StepDefinition(
                    id="button_led_step_led",
                    title="Add the LED and resistor",
                    subtitle="Place the LED and resistor so they can connect to the button circuit.",
                    icon="💡",
                    goal="The LED and resistor should be placed clearly beside the push button.",
                    completion_check="generic",
                    expected_objects=["breadboard", "push button", "led", "resistor"],
                    overlay_hint="Keep the push button, LED, and resistor close together in the center.",
                    target_region={"x": 0.50, "y": 0.52, "w": 0.34, "h": 0.20},
                    instruction_prompt="Explain to a beginner how to add an LED and resistor beside a push button on a breadboard.",
                    ai_reference_prompt="A top-down tutorial image showing a push button, LED, and resistor arranged on a breadboard.",
                ),
                StepDefinition(
                    id="button_led_step_wiring",
                    title="Connect the button circuit to the Arduino",
                    subtitle="Use jumper wires to connect the button and LED circuit to the Arduino.",
                    icon="🔌",
                    goal="The breadboard circuit should be wired to the Arduino Nano.",
                    completion_check="generic",
                    expected_objects=["arduino nano", "breadboard", "jumper wire"],
                    overlay_hint="Place the Arduino and jumper wires in the center so the setup is easy to inspect.",
                    target_region={"x": 0.66, "y": 0.46, "w": 0.36, "h": 0.30},
                    instruction_prompt="Explain to a beginner how to wire a push button LED circuit to an Arduino using jumper wires.",
                    ai_reference_prompt="A top-down beginner electronics image showing an Arduino Nano wired to a push button and LED circuit on a breadboard.",
                ),
                StepDefinition(
                    id="button_led_step_verify",
                    title="Verify the button-controlled LED setup",
                    subtitle="Check the button, LED, resistor, and wires before testing.",
                    icon="✅",
                    goal="The Arduino, push button, LED, resistor, and jumper wires should look like one complete button-controlled LED circuit.",
                    completion_check="scene_match",
                    expected_objects=["arduino nano", "breadboard", "led"],
                    overlay_hint="Place the full button LED setup in the middle for a final check.",
                    target_region={"x": 0.50, "y": 0.50, "w": 0.82, "h": 0.72},
                    instruction_prompt="Explain how a beginner should visually verify a full button-controlled LED circuit before testing.",
                    ai_reference_prompt="A complete top-down reference showing an Arduino Nano, push button, LED, resistor, and jumper wires in a beginner button-controlled LED circuit.",
                ),
            ],
        ),
        "mini_alarm": PlanDefinition(
            id="mini_alarm",
            name="Mini Alarm",
            badge="Intermediate",
            time_estimate="15-18m",
            description="Build a simple motion alarm using a PIR sensor, buzzer, and LED.",
            required_materials=["arduino nano", "pir motion sensor", "buzzer", "led", "220Ω resistor", "jumper wire"],
            steps=[
                StepDefinition(
                    id="mini_alarm_step_sensor",
                    title="Place the PIR motion sensor",
                    subtitle="Position the PIR sensor so it can connect clearly to the Arduino.",
                    icon="👁️",
                    goal="The PIR motion sensor should be clearly visible and ready to wire.",
                    completion_check="generic",
                    expected_objects=["pir motion sensor"],
                    overlay_hint="Place the PIR sensor in the center area so the camera can identify it.",
                    target_region={"x": 0.50, "y": 0.34, "w": 0.26, "h": 0.18},
                    instruction_prompt="Explain to a beginner how to place a PIR motion sensor so it can be wired to an Arduino.",
                    ai_reference_prompt="A top-down beginner-friendly electronics image showing a PIR motion sensor placed for wiring.",
                ),
                StepDefinition(
                    id="mini_alarm_step_outputs",
                    title="Add the buzzer and LED",
                    subtitle="Place the buzzer and LED that will act as the alarm outputs.",
                    icon="🚨",
                    goal="The buzzer and LED should be placed clearly beside the sensor and wiring area.",
                    completion_check="generic",
                    expected_objects=["buzzer", "led", "resistor"],
                    overlay_hint="Keep the buzzer and LED close together near the center.",
                    target_region={"x": 0.50, "y": 0.52, "w": 0.34, "h": 0.20},
                    instruction_prompt="Explain to a beginner how to add a buzzer and LED to a simple alarm circuit.",
                    ai_reference_prompt="A top-down tutorial image showing a buzzer, LED, and resistor placed for a simple alarm circuit.",
                ),
                StepDefinition(
                    id="mini_alarm_step_wiring",
                    title="Connect the alarm parts to the Arduino",
                    subtitle="Use jumper wires to connect the PIR sensor, buzzer, and LED to the Arduino.",
                    icon="🔌",
                    goal="The sensor and alarm outputs should all be wired to the Arduino Nano.",
                    completion_check="generic",
                    expected_objects=["arduino nano", "jumper wire"],
                    overlay_hint="Place the Arduino and the full wiring path in the middle of the camera view.",
                    target_region={"x": 0.66, "y": 0.46, "w": 0.36, "h": 0.30},
                    instruction_prompt="Explain to a beginner how to wire a PIR sensor, buzzer, and LED to an Arduino.",
                    ai_reference_prompt="A top-down beginner electronics image showing an Arduino Nano wired to a PIR sensor, buzzer, and LED.",
                ),
                StepDefinition(
                    id="mini_alarm_step_verify",
                    title="Verify the mini alarm setup",
                    subtitle="Check that the sensor, buzzer, LED, and wires are all connected correctly.",
                    icon="✅",
                    goal="The full mini alarm should look ready to test.",
                    completion_check="scene_match",
                    expected_objects=["arduino nano", "led"],
                    overlay_hint="Place the entire mini alarm setup in the center for a final check.",
                    target_region={"x": 0.50, "y": 0.50, "w": 0.82, "h": 0.72},
                    instruction_prompt="Explain how a beginner should visually verify a mini alarm circuit before testing.",
                    ai_reference_prompt="A complete top-down reference showing a beginner mini alarm circuit with Arduino Nano, PIR sensor, buzzer, LED, resistor, and jumper wires.",
                ),
            ],
        ),
        "temperature_display": PlanDefinition(
            id="temperature_display",
            name="Temperature Display",
            badge="Intermediate",
            time_estimate="18-22m",
            description="Build a simple temperature display using a DHT11 sensor and an OLED display.",
            required_materials=["arduino nano", "dht11 sensor", "oled display", "jumper wire"],
            steps=[
                StepDefinition(
                    id="temperature_display_step_sensor",
                    title="Place the DHT11 sensor",
                    subtitle="Position the temperature sensor where it can be connected clearly.",
                    icon="🌡️",
                    goal="The DHT11 sensor should be clearly visible and ready to wire.",
                    completion_check="generic",
                    expected_objects=["dht11 sensor"],
                    overlay_hint="Place the temperature sensor in the middle so the camera can inspect it.",
                    target_region={"x": 0.44, "y": 0.38, "w": 0.22, "h": 0.16},
                    instruction_prompt="Explain to a beginner how to place a DHT11 temperature sensor for a small Arduino project.",
                    ai_reference_prompt="A top-down beginner-friendly electronics image showing a DHT11 sensor placed for wiring.",
                ),
                StepDefinition(
                    id="temperature_display_step_display",
                    title="Place the display module",
                    subtitle="Position the OLED display so it can be connected to the Arduino.",
                    icon="🖥️",
                    goal="The display module should be clearly visible next to the sensor.",
                    completion_check="generic",
                    expected_objects=["oled display"],
                    overlay_hint="Keep the display module centered and visible to the camera.",
                    target_region={"x": 0.58, "y": 0.38, "w": 0.24, "h": 0.18},
                    instruction_prompt="Explain to a beginner how to place an OLED display module in a small Arduino setup.",
                    ai_reference_prompt="A top-down beginner-friendly image showing an OLED display module placed beside a sensor.",
                ),
                StepDefinition(
                    id="temperature_display_step_wiring",
                    title="Connect the sensor and display",
                    subtitle="Use jumper wires to connect both modules to the Arduino.",
                    icon="🔌",
                    goal="The temperature sensor and display should both be wired to the Arduino Nano.",
                    completion_check="generic",
                    expected_objects=["arduino nano", "jumper wire"],
                    overlay_hint="Show the Arduino and the full wiring path clearly in the center.",
                    target_region={"x": 0.64, "y": 0.48, "w": 0.38, "h": 0.30},
                    instruction_prompt="Explain to a beginner how to wire a DHT11 sensor and OLED display to an Arduino Nano.",
                    ai_reference_prompt="A top-down beginner electronics image showing an Arduino Nano wired to a DHT11 sensor and OLED display.",
                ),
                StepDefinition(
                    id="temperature_display_step_verify",
                    title="Verify the temperature display setup",
                    subtitle="Check the sensor, display, and wires before testing.",
                    icon="✅",
                    goal="The Arduino, temperature sensor, display, and jumper wires should look like a complete temperature display setup.",
                    completion_check="scene_match",
                    expected_objects=["arduino nano", "jumper wire"],
                    overlay_hint="Place the full temperature display setup in the middle for a final check.",
                    target_region={"x": 0.50, "y": 0.50, "w": 0.82, "h": 0.72},
                    instruction_prompt="Explain how a beginner should visually verify a small temperature display setup before testing.",
                    ai_reference_prompt="A complete top-down reference showing a beginner Arduino Nano temperature display project with a DHT11 sensor, OLED display, and jumper wires.",
                ),
            ],
        ),
        "robot_car": PlanDefinition(
            id="robot_car",
            name="Small Robot Car",
            badge="Advanced",
            time_estimate="25-30m",
            description="Assemble a small robot car with motors, a motor driver, battery pack, and ultrasonic sensor.",
            required_materials=["arduino nano", "dc motor", "motor driver", "battery pack", "wheel", "chassis", "ultrasonic sensor"],
            steps=[
                StepDefinition(
                    id="robot_car_step_base",
                    title="Mount the chassis and wheels",
                    subtitle="Place the chassis, wheels, and motors into the basic car layout.",
                    icon="🚗",
                    goal="The chassis and wheels should be assembled into a clear robot car base.",
                    completion_check="generic",
                    expected_objects=["chassis", "wheel", "dc motor"],
                    overlay_hint="Place the car base in the center so the camera can inspect the layout.",
                    target_region={"x": 0.50, "y": 0.42, "w": 0.40, "h": 0.24},
                    instruction_prompt="Explain to a beginner how to assemble a small robot car chassis with wheels and motors.",
                    ai_reference_prompt="A top-down beginner-friendly image showing a small robot car chassis with wheels and DC motors.",
                ),
                StepDefinition(
                    id="robot_car_step_control",
                    title="Place the Arduino and motor driver",
                    subtitle="Add the Arduino and motor driver on top of the robot car base.",
                    icon="🧠",
                    goal="The Arduino and motor driver should be positioned clearly on the chassis.",
                    completion_check="generic",
                    expected_objects=["arduino nano", "motor driver", "chassis"],
                    overlay_hint="Keep the Arduino and motor driver visible near the center of the car.",
                    target_region={"x": 0.50, "y": 0.50, "w": 0.34, "h": 0.20},
                    instruction_prompt="Explain to a beginner how to place an Arduino Nano and motor driver on a robot car chassis.",
                    ai_reference_prompt="A top-down beginner tutorial image showing an Arduino Nano and motor driver placed on a small robot car chassis.",
                ),
                StepDefinition(
                    id="robot_car_step_sensor",
                    title="Add the ultrasonic sensor",
                    subtitle="Mount the ultrasonic sensor at the front of the car.",
                    icon="📡",
                    goal="The ultrasonic sensor should be clearly attached at the front of the robot car.",
                    completion_check="generic",
                    expected_objects=["ultrasonic sensor"],
                    overlay_hint="Show the front sensor clearly in the center area for tracking.",
                    target_region={"x": 0.50, "y": 0.28, "w": 0.22, "h": 0.16},
                    instruction_prompt="Explain to a beginner how to place an ultrasonic sensor on the front of a small robot car.",
                    ai_reference_prompt="A top-down beginner-friendly image showing an ultrasonic sensor mounted at the front of a small robot car.",
                ),
                StepDefinition(
                    id="robot_car_step_power",
                    title="Connect the battery pack and wiring",
                    subtitle="Connect the battery pack, motors, and driver wiring to complete the build.",
                    icon="🔋",
                    goal="The wiring and battery pack should complete the robot car assembly.",
                    completion_check="scene_match",
                    expected_objects=["battery pack", "motor driver", "arduino nano"],
                    overlay_hint="Place the whole robot car setup in the middle for a final wiring check.",
                    target_region={"x": 0.50, "y": 0.50, "w": 0.86, "h": 0.74},
                    instruction_prompt="Explain how a beginner should visually verify a small robot car setup with battery, motor driver, and Arduino before testing.",
                    ai_reference_prompt="A complete top-down reference showing a beginner small robot car build with Arduino Nano, motor driver, battery pack, motors, wheels, chassis, and ultrasonic sensor.",
                ),
            ],
        ),
    }


PLANS = seed_plans()
SESSIONS: Dict[str, SessionState] = {}
DYNAMIC_PLANS: Dict[str, PlanDefinition] = {}


# =========================
# AI services
# =========================

class PlannerService:
    def __init__(self) -> None:
        self.client = maybe_get_openai_client()

    def build_dynamic_steps(self, plan_name: str, inventory: List[str], preferences: Dict[str, Any]) -> Optional[List[StepDefinition]]:
        if self.client is None:
            return None

        prompt = (
            "You are an AR assembly coach for beginner electronics makers. "
            "Generate 4 to 6 beginner-friendly physical build steps for the selected plan. "
            "Return JSON only with a top-level key 'steps'. Each step must include: "
            "id, title, subtitle, icon, goal, completion_check, expected_objects, overlay_hint, target_region, instruction_prompt, ai_reference_prompt. "
            "Allowed completion_check values: led_region, resistor_region, nano_region, scene_match, generic. "
            f"Plan name: {plan_name}. Inventory: {inventory}. Preferences: {preferences}."
        )

        try:
            response = self.client.responses.create(
                model=DEFAULT_MODEL,
                input=prompt,
                temperature=0.2,
            )
            text = getattr(response, "output_text", "") or ""
            if not text:
                return None
            data = json.loads(text)
            result: List[StepDefinition] = []
            for raw in data.get("steps", []):
                result.append(StepDefinition(**raw))
            return result or None
        except Exception:
            return None

    def beginner_instruction(self, step: StepDefinition) -> str:
        if self.client is None:
            return step.subtitle
        try:
            response = self.client.responses.create(
                model=DEFAULT_MODEL,
                input=(
                    "Give one short beginner-friendly coaching instruction, under 30 words, "
                    "for this physical assembly step: "
                    f"{step.instruction_prompt}"
                ),
                temperature=0.3,
            )
            text = getattr(response, "output_text", "") or ""
            return text.strip() or step.subtitle
        except Exception:
            return step.subtitle

    def generate_reference_image(self, step: StepDefinition) -> Optional[str]:
        if self.client is None:
            return None
        try:
            image_result = self.client.images.generate(
                model=DEFAULT_IMAGE_MODEL,
                prompt=step.ai_reference_prompt,
                size="1024x1024",
            )
            data_list = getattr(image_result, "data", None) or []
            if not data_list:
                return None
            first = data_list[0]
            if getattr(first, "b64_json", None):
                return f"data:image/png;base64,{first.b64_json}"
            if getattr(first, "url", None):
                return first.url
            return None
        except Exception:
            return None


planner_service = PlannerService()

def _extract_first_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return {}
    return {}


def gpt_snapshot_judge(step: StepDefinition, image_data_url: str) -> Dict[str, Any]:
    """
    Return shape:
    {
      "is_complete": bool,
      "confidence": float [0..1],
      "reason": str
    }
    """
    client = maybe_get_openai_client()
    if client is None:
        return {
            "is_complete": False,
            "confidence": 0.0,
            "reason": "Live tracking unavailable. You can press Next to continue manually.",
        }

    prompt = f"""
You are a strict electronics build checker for beginners.
Current step:
- id: {step.id}
- title: {step.title}
- goal: {step.goal}
- hint: {step.overlay_hint}

Task:
Decide whether THIS STEP is completed in the provided camera frame.

Rules:
- Be conservative. If uncertain, return incomplete.
- Judge only current step completion, not the whole project.
- Ignore UI overlays/text; focus on physical components in camera image.

extra_rule = ""
if step.id in {"step_led", "step_led_c"}:
    extra_rule = (
        "For this step, pass only if exactly one LED is visibly inserted on the breadboard, "
        "with two legs in different rows across the center gap. "
        "If LED is absent/unclear/not inserted, return is_complete=false."
    )

Return ONLY valid JSON object:
{{
  "is_complete": true/false,
  "confidence": 0.0-1.0,
  "reason": "short reason under 20 words"
}}
"""

    try:
        response = client.responses.create(
            model=DEFAULT_VISION_MODEL,
            temperature=0.0,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
        )
        text = getattr(response, "output_text", "") or ""
        data = _extract_first_json_object(text)

        is_complete = bool(data.get("is_complete", False))
        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        reason = str(data.get("reason", "")).strip() or "No reason provided."

        return {
            "is_complete": is_complete,
            "confidence": confidence,
            "reason": reason,
        }
    except Exception as exc:
        return {
            "is_complete": False,
            "confidence": 0.0,
            "reason": f"Vision check failed: {exc}",
        }


# =========================
# Vision helpers
# =========================

mp_hands = mp.solutions.hands


def decode_image(image_base64: str) -> np.ndarray:
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",", 1)[1]
    binary = base64.b64decode(image_base64)
    image = ImageOpenCV.from_bytes(binary)
    return image


class ImageOpenCV:
    @staticmethod
    def from_bytes(binary: bytes) -> np.ndarray:
        arr = np.frombuffer(binary, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image.")
        return image



def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))



def box_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax1 = a["x"] - a["w"] / 2
    ay1 = a["y"] - a["h"] / 2
    ax2 = a["x"] + a["w"] / 2
    ay2 = a["y"] + a["h"] / 2
    bx1 = b["x"] - b["w"] / 2
    by1 = b["y"] - b["h"] / 2
    bx2 = b["x"] + b["w"] / 2
    by2 = b["y"] + b["h"] / 2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return 0.0 if union <= 0 else inter / union


def ema_bbox(prev: Dict[str, float], nxt: Dict[str, float], alpha: float) -> Dict[str, float]:
    a = clamp01(alpha)
    return {
        "x": clamp01(a * nxt["x"] + (1.0 - a) * prev["x"]),
        "y": clamp01(a * nxt["y"] + (1.0 - a) * prev["y"]),
        "w": clamp01(a * nxt["w"] + (1.0 - a) * prev["w"]),
        "h": clamp01(a * nxt["h"] + (1.0 - a) * prev["h"]),
    }


class PrototypeVisionService:
    """
    Prototype CV service:
    - OpenCV color segmentation as lightweight detector proxy
    - MediaPipe hands to infer active manipulation
    - YOLO placeholder is represented as a hook in self.detect_objects()
    """

    STABILITY_IOU_MATCH = 0.22
    STABILITY_REQUIRED_STREAK = 2
    STABILITY_DROP_AFTER_MISSES = 2
    STABILITY_EMA_ALPHA = 0.45

    def __init__(self) -> None:
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    def detect_objects(self, image_bgr: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        breadboard_boxes = self._detect_breadboard(image_bgr, hsv, w, h)
        led_boxes = self._detect_red_led(hsv, w, h)
        resistor_boxes = self._detect_resistor(hsv, w, h)

        if breadboard_boxes:
            # Restrict tiny components to the breadboard area to avoid
            # false positives from white background and room objects.
            led_boxes = self._filter_inside_breadboard(led_boxes, breadboard_boxes[0]["bbox"])
            resistor_boxes = self._filter_inside_breadboard(resistor_boxes, breadboard_boxes[0]["bbox"])

        detections: Dict[str, List[Dict[str, Any]]] = {
            "led": led_boxes,
            "resistor": resistor_boxes,
            "arduino nano": self._detect_blue_board(hsv, w, h),
            "breadboard": breadboard_boxes,
            "jumper wire": [],
        }
        return detections

    def detect_hands(self, image_bgr: np.ndarray) -> List[Tuple[float, float]]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        points: List[Tuple[float, float]] = []
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                xs = [lm.x for lm in hand.landmark]
                ys = [lm.y for lm in hand.landmark]
                points.append((float(sum(xs) / len(xs)), float(sum(ys) / len(ys))))
        return points

    def analyze_step(self, session: SessionState, image_bgr: np.ndarray, image_width: int, image_height: int) -> Dict[str, Any]:
        step = session.steps[session.current_step_index]
        detections = self.detect_objects(image_bgr)
        hands = self.detect_hands(image_bgr)

        overlay = self._build_overlay(session, step, detections, hands)
        matched, confidence, feedback = self._evaluate_step(step, detections, hands)
        scene_score = self._scene_match_score(step, detections)

        result = {
            "step_id": step.id,
            "matched": matched,
            "confidence": round(confidence, 3),
            "feedback": feedback,
            "hands_detected": len(hands),
            "scene_match_score": round(scene_score, 3),
            "detections": detections,
            "overlay": overlay,
        }
        session.last_analysis = result
        return result

    def _scene_match_score(self, step: StepDefinition, detections: Dict[str, List[Dict[str, Any]]]) -> float:
        hits = 0
        for obj in step.expected_objects:
            if detections.get(obj):
                hits += 1
        return hits / max(1, len(step.expected_objects))

    def _evaluate_step(
        self,
        step: StepDefinition,
        detections: Dict[str, List[Dict[str, Any]]],
        hands: List[Tuple[float, float]],
    ) -> Tuple[bool, float, str]:
        target = self._resolve_target_region(step, detections)

        def best_iou(label: str) -> float:
            candidates = detections.get(label, [])
            if not candidates:
                return 0.0
            return max(box_iou(target, item["bbox"]) for item in candidates if "bbox" in item)

        if step.completion_check == "led_region":
            iou = best_iou("led")
            matched = iou > 0.18
            feedback = "LED placement looks correct." if matched else "Move the LED closer to the highlighted center-gap region."
            return matched, max(iou, 0.15 if hands else 0.05), feedback

        if step.completion_check == "resistor_region":
            iou = best_iou("resistor")
            matched = iou > 0.16
            feedback = "Resistor appears in the correct wiring zone." if matched else "Place the resistor along the highlighted ground-side path."
            return matched, max(iou, 0.15 if hands else 0.05), feedback

        if step.completion_check == "nano_region":
            iou = best_iou("arduino nano")
            matched = iou > 0.15
            feedback = "Arduino Nano appears connected in the right region." if matched else "Move the Arduino Nano into the highlighted right-side area."
            return matched, max(iou, 0.15 if hands else 0.05), feedback

        if step.completion_check == "scene_match":
            score = self._scene_match_score(step, detections)
            matched = score >= 0.75
            feedback = "The overall build looks ready for verification." if matched else "The full scene is not matching yet. Check LED, resistor, and Nano placement."
            return matched, score, feedback

        score = self._scene_match_score(step, detections)
        matched = score >= 0.6
        feedback = "Current scene generally matches the step." if matched else "Current scene still differs from the target step."
        return matched, score, feedback

    def _resolve_target_region(self, step: StepDefinition, detections: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        breadboards = detections.get("breadboard", [])
        if not breadboards:
            return step.target_region

        bb = breadboards[0]["bbox"]

        # relative coordinates inside detected breadboard
        if step.completion_check == "led_region":
            return {
                "x": bb["x"],
                "y": bb["y"] - bb["h"] * 0.12,
                "w": bb["w"] * 0.34,
                "h": bb["h"] * 0.18,
            }

        if step.completion_check == "resistor_region":
            return {
                "x": bb["x"] - bb["w"] * 0.10,
                "y": bb["y"] + bb["h"] * 0.10,
                "w": bb["w"] * 0.42,
                "h": bb["h"] * 0.16,
            }

        if step.completion_check == "nano_region":
            return {
                "x": bb["x"] + bb["w"] * 0.62,
                "y": bb["y"],
                "w": bb["w"] * 0.26,
                "h": bb["h"] * 0.45,
            }

        return step.target_region

    def _raw_candidate_boxes(self, step: StepDefinition, detections: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for label in step.expected_objects:
            items = detections.get(label, [])
            if not items:
                continue
            best = items[0]
            bbox = best.get("bbox")
            if bbox and best.get("score", 0.0) >= 0.55:
                out.append({
                    "label": label,
                    "bbox": dict(bbox),
                    "score": float(best.get("score", 0.0)),
                })
        return out

    def _stabilize_boxes(
        self,
        session: SessionState,
        step_id: str,
        expected_labels: List[str],
        raw_boxes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        root = session.vision_stability
        if root.get("step_id") != step_id:
            root.clear()
            root["step_id"] = step_id
            root["labels"] = {}
        by_label_state: Dict[str, Any] = root["labels"]

        raw_by_label = {item["label"]: item for item in raw_boxes}
        out: List[Dict[str, Any]] = []

        for label in expected_labels:
            state = by_label_state.setdefault(
                label,
                {"streak": 0, "misses": 0, "last_raw": None, "ema": None, "last_score": 0.0},
            )
            raw_item = raw_by_label.get(label)

            if raw_item is None:
                state["misses"] = int(state["misses"]) + 1
                state["streak"] = 0
                state["last_raw"] = None
                if state["misses"] >= self.STABILITY_DROP_AFTER_MISSES:
                    state["ema"] = None
                continue

            state["misses"] = 0
            bbox = raw_item["bbox"]
            state["last_score"] = float(raw_item.get("score", 0.0))

            prev_raw = state["last_raw"]
            if prev_raw is None:
                state["streak"] = 1
            elif box_iou(prev_raw, bbox) >= self.STABILITY_IOU_MATCH:
                state["streak"] = int(state["streak"]) + 1
            else:
                state["streak"] = 1
                state["ema"] = None

            state["last_raw"] = dict(bbox)

            if int(state["streak"]) >= self.STABILITY_REQUIRED_STREAK:
                if state["ema"] is None:
                    state["ema"] = dict(bbox)
                else:
                    state["ema"] = ema_bbox(state["ema"], bbox, self.STABILITY_EMA_ALPHA)
                out.append({
                    "label": label,
                    "bbox": dict(state["ema"]),
                    "score": round(state["last_score"], 3),
                })

        return out

    def _build_overlay(
        self,
        session: SessionState,
        step: StepDefinition,
        detections: Dict[str, List[Dict[str, Any]]],
        hands: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        target = self._resolve_target_region(step, detections)
        overlays = {
            "highlight_box": target,
            "label": step.title,
            "hint": step.overlay_hint,
            "arrows": [self._make_arrow_for_target(target)],
            "detected_boxes": [],
            "hand_points": [{"x": x, "y": y} for x, y in hands],
        }
        raw = self._raw_candidate_boxes(step, detections)
        overlays["detected_boxes"] = self._stabilize_boxes(
            session, step.id, step.expected_objects, raw
        )
        return overlays

    def _make_arrow_for_target(self, target: Dict[str, float]) -> Dict[str, Any]:
        return {
            "x1": clamp01(target["x"]),
            "y1": clamp01(target["y"] - target["h"] * 0.9),
            "x2": clamp01(target["x"]),
            "y2": clamp01(target["y"] - target["h"] * 0.45),
        }

    def _detect_red_led(self, hsv: np.ndarray, w: int, h: int) -> List[Dict[str, Any]]:
        lower1 = np.array([0, 95, 70])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 95, 70])
        upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        return self._mask_to_boxes(
            mask,
            w,
            h,
            min_area=120,
            label_bias=(0.0, 0.0),
            min_score=0.56,
            max_area_ratio=0.02,
            min_aspect=0.45,
            max_aspect=2.2,
        )

    def _detect_resistor(self, hsv: np.ndarray, w: int, h: int) -> List[Dict[str, Any]]:
        lower = np.array([10, 55, 55])
        upper = np.array([28, 180, 225])
        mask = cv2.inRange(hsv, lower, upper)
        return self._mask_to_boxes(
            mask,
            w,
            h,
            min_area=180,
            label_bias=(0.0, 0.0),
            min_score=0.55,
            max_area_ratio=0.03,
            min_aspect=1.25,
            max_aspect=8.0,
        )

    def _detect_blue_board(self, hsv: np.ndarray, w: int, h: int) -> List[Dict[str, Any]]:
        lower = np.array([90, 60, 50])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        return self._mask_to_boxes(
            mask,
            w,
            h,
            min_area=600,
            label_bias=(0.0, 0.0),
            min_score=0.52,
            max_area_ratio=0.5,
            min_aspect=0.25,
            max_aspect=5.0,
        )

    def _detect_breadboard(self, image_bgr: np.ndarray, hsv: np.ndarray, w: int, h: int) -> List[Dict[str, Any]]:
        # Breadboard is usually light/white with low saturation and reasonably high value
        lower = np.array([0, 0, 120])
        upper = np.array([180, 70, 255])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8000:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)

            # Breadboard is roughly rectangular and wider than tall
            if aspect < 0.8 or aspect > 2.5:
                continue

            boxes.append({
                "bbox": {
                    "x": clamp01((x + bw / 2) / w),
                    "y": clamp01((y + bh / 2) / h),
                    "w": clamp01(bw / w),
                    "h": clamp01(bh / h),
                },
                "score": round(min(0.95, 0.55 + area / (w * h)), 3),
            })

        boxes.sort(key=lambda item: item["score"], reverse=True)
        return boxes[:1]

    def _mask_to_boxes(
        self,
        mask: np.ndarray,
        w: int,
        h: int,
        min_area: int,
        label_bias: Tuple[float, float],
        min_score: float = 0.0,
        max_area_ratio: float = 1.0,
        min_aspect: float = 0.0,
        max_aspect: float = 100.0,
    ) -> List[Dict[str, Any]]:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Dict[str, Any]] = []
        frame_area = max(float(w * h), 1.0)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(float(bh), 1.0)
            area_ratio = area / frame_area
            if area_ratio > max_area_ratio:
                continue
            if aspect < min_aspect or aspect > max_aspect:
                continue
            score = round(min(0.95, 0.45 + area_ratio), 3)
            if score < min_score:
                continue
            boxes.append(
                {
                    "bbox": {
                        "x": clamp01((x + bw / 2) / w + label_bias[0]),
                        "y": clamp01((y + bh / 2) / h + label_bias[1]),
                        "w": clamp01(bw / w),
                        "h": clamp01(bh / h),
                    },
                    "score": score,
                }
            )
        boxes.sort(key=lambda item: item["score"], reverse=True)
        return boxes[:5]

    def _filter_inside_breadboard(
        self, items: List[Dict[str, Any]], breadboard_bbox: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        bb_left = breadboard_bbox["x"] - breadboard_bbox["w"] / 2
        bb_top = breadboard_bbox["y"] - breadboard_bbox["h"] / 2
        bb_right = breadboard_bbox["x"] + breadboard_bbox["w"] / 2
        bb_bottom = breadboard_bbox["y"] + breadboard_bbox["h"] / 2

        kept: List[Dict[str, Any]] = []
        for item in items:
            bbox = item.get("bbox")
            if not bbox:
                continue
            cx, cy = bbox["x"], bbox["y"]
            inside = bb_left <= cx <= bb_right and bb_top <= cy <= bb_bottom
            if not inside:
                continue
            # Light edge margin penalty: components near border are often noise.
            edge_margin_x = breadboard_bbox["w"] * 0.04
            edge_margin_y = breadboard_bbox["h"] * 0.04
            near_edge = (
                cx < bb_left + edge_margin_x
                or cx > bb_right - edge_margin_x
                or cy < bb_top + edge_margin_y
                or cy > bb_bottom - edge_margin_y
            )
            if near_edge:
                score = max(0.0, float(item.get("score", 0.0)) - 0.06)
                if score < 0.55:
                    continue
                item = {**item, "score": round(score, 3)}
            kept.append(item)
        kept.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return kept[:3]


vision_service = PrototypeVisionService()


# =========================
# SVG fallback reference generator
# =========================

def generate_svg_reference(step: StepDefinition) -> str:
    title = step.title.replace("&", "&amp;")
    subtitle = step.subtitle.replace("&", "&amp;")
    x = int(step.target_region["x"] * 800)
    y = int(step.target_region["y"] * 600)
    w = int(step.target_region["w"] * 800)
    h = int(step.target_region["h"] * 600)

    svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' width='800' height='600' viewBox='0 0 800 600'>
      <rect width='800' height='600' fill='#f6f7fb'/>
      <rect x='130' y='110' width='420' height='290' rx='24' fill='#ffffff' stroke='#d8dee9' stroke-width='4'/>
      <rect x='570' y='130' width='110' height='180' rx='18' fill='#6fa8dc' stroke='#365f91' stroke-width='4'/>
      <text x='40' y='50' font-family='Arial' font-size='28' font-weight='700' fill='#1f2937'>{title}</text>
      <text x='40' y='82' font-family='Arial' font-size='18' fill='#6b7280'>{subtitle}</text>
      <rect x='{x - w // 2}' y='{y - h // 2}' width='{w}' height='{h}' rx='18' fill='none' stroke='#10b981' stroke-width='6' stroke-dasharray='12 10'/>
      <line x1='{x}' y1='{max(40, y - h)}' x2='{x}' y2='{y - h // 2}' stroke='#f97316' stroke-width='7'/>
      <polygon points='{x},{y - h // 2} {x - 14},{y - h // 2 - 24} {x + 14},{y - h // 2 - 24}' fill='#f97316'/>
      <circle cx='{x}' cy='{y}' r='18' fill='#ef4444' opacity='0.85'/>
      <rect x='145' y='430' width='510' height='110' rx='18' fill='#ffffff' stroke='#e5e7eb'/>
      <text x='170' y='475' font-family='Arial' font-size='22' font-weight='700' fill='#111827'>Target outcome</text>
      <text x='170' y='510' font-family='Arial' font-size='18' fill='#374151'>{step.goal}</text>
      <text x='170' y='535' font-family='Arial' font-size='16' fill='#6b7280'>Fallback reference generated locally because image API was unavailable.</text>
    </svg>
    """
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


# =========================
# Session helpers
# =========================

def inventory_covers(plan: PlanDefinition, inventory: List[str]) -> Tuple[bool, List[str]]:
    inventory_lower = {item.strip().lower() for item in inventory}
    missing = []
    for required in plan.required_materials:
        if required.lower() not in inventory_lower:
            missing.append(required)
    return len(missing) == 0, missing



def build_plan_summaries(inventory: List[str]) -> List[PlanSummary]:
    plans: List[PlanSummary] = []
    for plan in PLANS.values():
        ok, missing = inventory_covers(plan, inventory)
        plans.append(
            PlanSummary(
                id=plan.id,
                name=plan.name,
                badge=plan.badge if ok else "Unavailable",
                time_estimate=plan.time_estimate,
                description=plan.description,
                available=ok,
                missing_parts=missing,
            )
        )
    return plans



def get_step_payload(session: SessionState) -> Dict[str, Any]:
    step = session.steps[session.current_step_index]
    progress_labels = [item.title for item in session.steps]
    current_instruction = step.subtitle
    return {
        "session_id": session.id,
        "plan_id": session.plan_id,
        "plan_name": session.plan_name,
        "current_step_index": session.current_step_index,
        "total_steps": len(session.steps),
        "paused": session.paused,
        "zoom": session.zoom,
        "progress_labels": progress_labels,
        "step": {
            "id": step.id,
            "title": step.title,
            "subtitle": current_instruction,
            "original_subtitle": step.subtitle,
            "icon": step.icon,
            "goal": step.goal,
            "target_region": step.target_region,
            "overlay_hint": step.overlay_hint,
        },
        "last_analysis": session.last_analysis,
    }


# =========================
# FastAPI app
# =========================

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="AR Builder Coach Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_ui():
    return FileResponse("index.html")  

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "openai_configured": bool(OPENAI_API_KEY),
        "anthropic_configured": bool(ANTHROPIC_API_KEY),
        "available_plans": list(PLANS.keys()),
    }


@app.post("/api/plans")
def get_plans(req: InventoryRequest) -> Dict[str, Any]:
    materials = req.materials or ["breadboard", "arduino nano", "led", "220Ω resistor", "jumper wire", "usb cable"]
    print(f"[plans] received materials: {materials}")
    if not ANTHROPIC_API_KEY: 
        print("[plans] No API key, falling back to hardcoded plans")
        return {"plans": [item.model_dump() for item in build_plan_summaries(materials)]}
    
    prompt = f"""
    You are an AR electronics coach for beginners. The user has these materials: {materials}. 

    Generate 2-3 feasible beginner electronics projects they can build using only the materials that they have. For each project, also generate 3-5 step-by-step physical assembly steps.

    Return ONLY a valid JSON object like this: 
        {{
    "plans": [
        {{
        "id": "dynamic_plan_1",
        "name": "Blinking LED",
        "badge": "Beginner",
        "time_estimate": "10-15m",
        "description": "A short description of the project.",
        "required_materials": ["breadboard", "led"],
        "steps": [
            {{
            "id": "step_1",
            "title": "Place the LED",
            "subtitle": "Insert the LED into two rows on the breadboard.",
            "icon": "💡",
            "goal": "LED is visible on the breadboard with legs in separate rows.",
            "completion_check": "led_region",
            "expected_objects": ["breadboard", "led"],
            "overlay_hint": "Place the LED across two separate rows.",
            "target_region": {{"x": 0.5, "y": 0.4, "w": 0.2, "h": 0.15}},
            "instruction_prompt": "Explain how to place an LED on a breadboard.",
            "ai_reference_prompt": "Top-down image of an LED inserted on a breadboard."
            }}
        ]
        }}
    ]
    }}

    Rules:
    - Only suggest projects buildable with the materials listed. Not every part has to be used but do not assume any other parts are there except for the addition of a breadboard.
    - completion_check must be one of: led_region, resistor_region, nano_region, scene_match, generic
    - Steps must be physical, hands-on actions in the correct assembly order.
    - Keep ALL string values under 20 words
    - Return ONLY the JSON, no markdown, no explanation.
    """

    client = OpenAI(
        api_key=ANTHROPIC_API_KEY
    )
   
    '''import urllib.request as _urllib_request
    payload = json.dumps({
        "model": "claude-opus-4-6",
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }'''

    try:
        '''http_req = _urllib_request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload, headers=headers, method="POST"
        )
        with _urllib_request.urlopen(http_req) as resp:
            data = json.loads(resp.read())'''
        response = client.chat.completions.create(
            model="gpt-4o-mini",          
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = response.choices[0].message.content
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        print(f"[plans] claude raw response: {text}")
        print(f"[plans] parsed {len(parsed.get('plans', []))} plans")

        summaries = []
        for plan_data in parsed.get("plans", []):
            steps = [StepDefinition(**s) for s in plan_data.get("steps", [])]

            plan_def = PlanDefinition(
                id=plan_data["id"],
                name=plan_data["name"],
                badge=plan_data["badge"],
                time_estimate=plan_data["time_estimate"],
                description=plan_data["description"],
                required_materials=plan_data.get("required_materials", materials),
                steps=steps,
            )
            DYNAMIC_PLANS[plan_def.id] = plan_def

            summaries.append(PlanSummary(
                id=plan_def.id,
                name=plan_def.name,
                badge=plan_def.badge,
                time_estimate=plan_def.time_estimate,
                description=plan_def.description,
                available=True,
                missing_parts=[],
            ).model_dump())

        print(f"[plans] returning summaries: {summaries}")
        return {"plans": summaries}

    except Exception as exc:
        print(f"[plans] EXCEPTION: {exc}")
        import traceback
        traceback.print_exc()
        return {"plans": [item.model_dump() for item in build_plan_summaries(materials)]}

@app.post("/api/session/start")
def start_session(req: StartSessionRequest) -> Dict[str, Any]:
    inventory = req.inventory or ["breadboard", "arduino nano", "led", "220Ω resistor", "jumper wire", "usb cable"]

    if req.plan_id in PLANS:
        plan = PLANS[req.plan_id]
        steps = plan.steps
        plan_name = plan.name
    elif req.plan_id in DYNAMIC_PLANS:         
        plan = DYNAMIC_PLANS[req.plan_id]
        steps = plan.steps
        plan_name = plan.name
    else:
        dynamic_steps = planner_service.build_dynamic_steps(req.plan_name or req.plan_id, inventory, req.preferences)
        if not dynamic_steps:
            raise HTTPException(status_code=404, detail="Unknown plan and dynamic AI planning failed.")
        steps = dynamic_steps
        plan_name = req.plan_name or req.plan_id

    session_id = str(uuid.uuid4())
    session = SessionState(
        id=session_id,
        plan_id=req.plan_id,
        plan_name=plan_name,
        inventory=inventory,
        preferences=req.preferences,
        steps=steps,
    )
    SESSIONS[session_id] = session
    return get_step_payload(session)


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return get_step_payload(session)


@app.post("/api/session/{session_id}/step/next")
def next_step(session_id: str) -> Dict[str, Any]:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    now = time.time()
    if now - session.last_next_ts < 0.8:
        return get_step_payload(session)
    session.last_next_ts = now

    if session.current_step_index < len(session.steps) - 1:
        session.current_step_index += 1
        session.last_analysis = {}
        session.gpt_eval_state = {}
    return get_step_payload(session)


@app.post("/api/session/{session_id}/step/repeat")
def repeat_step(session_id: str, req: RepeatStepRequest) -> Dict[str, Any]:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    payload = get_step_payload(session)
    payload["replay_seconds"] = req.replay_seconds
    payload["message"] = "Replay current AR guidance."
    return payload


@app.post("/api/session/{session_id}/step/toggle-pause")
def toggle_pause(session_id: str, req: TogglePauseRequest) -> Dict[str, Any]:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    session.paused = (not session.paused) if req.paused is None else req.paused
    return get_step_payload(session)


@app.post("/api/session/{session_id}/step/zoom")
def zoom_step(session_id: str, req: ZoomRequest) -> Dict[str, Any]:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    session.zoom = round(max(1.0, min(2.5, session.zoom + req.delta)), 2)
    return get_step_payload(session)


@app.post("/api/frame/analyze")
def analyze_frame(req: FrameAnalyzeRequest) -> Dict[str, Any]:
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    step = session.steps[session.current_step_index]

    # Per-step GPT state
    state = session.gpt_eval_state
    if state.get("step_id") != step.id:
        state.clear()
        state["step_id"] = step.id
        state["checks"] = 0
        state["consecutive_passes"] = 0
        state["validated"] = False

    # If already validated for this step, don't call GPT repeatedly.
    if state.get("validated"):
        result = {
            "step_id": step.id,
            "matched": True,
            "confidence": 0.99,
            "feedback": "Step already validated. Tap Next when ready.",
            "hands_detected": 0,
            "scene_match_score": 0.99,
            "detections": {},
            "overlay": {
                "highlight_box": step.target_region,
                "label": step.title,
                "hint": "Step completed (GPT validated).",
                "arrows": [],
                "detected_boxes": [],
                "hand_points": [],
            },
            "source": "gpt_snapshot",
        }
        # Never auto-advance. User must tap "Next" manually.
        result["ready_for_next"] = False
        session.last_analysis = result
        return result

    # One screenshot -> one GPT judgement
    judge = gpt_snapshot_judge(step, req.image_base64)
    state["checks"] = int(state.get("checks", 0)) + 1

    single_pass = judge["is_complete"] and judge["confidence"] >= 0.65
    if single_pass:
        state["consecutive_passes"] = int(state.get("consecutive_passes", 0)) + 1
    else:
        state["consecutive_passes"] = 0

    # Debounce: require 2 consecutive passes
    if state["consecutive_passes"] >= 2:
        state["validated"] = True

    matched = bool(state.get("validated", False))
    confidence = judge["confidence"]

    result = {
        "step_id": step.id,
        "matched": matched,
        "confidence": round(confidence, 3),
        "feedback": (
            "Step looks correct. Tap Next to continue."
            if matched
            else f"{judge['reason']} (check #{state['checks']}, pass streak {state['consecutive_passes']}/2)"
        ),
        "hands_detected": 0,
        "scene_match_score": round(confidence, 3),
        "detections": {},
        "overlay": {
            "highlight_box": step.target_region,
            "label": step.title,
            "hint": step.overlay_hint,
            "arrows": [],
            "detected_boxes": [],
            "hand_points": [],
        },
        "source": "gpt_snapshot",
    }

    result["ready_for_next"] = False

    session.last_analysis = result
    return result


@app.post("/api/session/{session_id}/ref-wiring")
def ref_wiring(session_id: str, req: RefWiringRequest) -> Dict[str, Any]:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    step = session.steps[session.current_step_index]
    if step.id in session.reference_cache:
        return {"image_url": session.reference_cache[step.id], "source": "cache"}

    image_url = planner_service.generate_reference_image(step)
    source = "openai"
    if image_url is None:
        image_url = generate_svg_reference(step)
        source = "fallback_svg"

    session.reference_cache[step.id] = image_url
    return {"image_url": image_url, "source": source}


@app.get("/api/reference/{session_id}/{step_id}")
def get_cached_reference(session_id: str, step_id: str) -> Dict[str, Any]:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if step_id not in session.reference_cache:
        raise HTTPException(status_code=404, detail="Reference not generated yet.")
    return {"image_url": session.reference_cache[step_id]}


class AnthropicProxyRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Dict[str, Any]]


@app.post("/api/claude/messages")
def claude_messages(req: AnthropicProxyRequest) -> Dict[str, Any]:
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured in .env")
    '''import urllib.request as _urllib_request
    payload = json.dumps(req.model_dump()).encode()
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }
    http_req = _urllib_request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers=headers,
        method="POST",
    )'''
    try:
        '''with _urllib_request.urlopen(http_req) as resp:
            return json.loads(resp.read())'''
        client = OpenAI(
            api_key=ANTHROPIC_API_KEY
        )
        oai_messages = []
        for msg in req.messages:
            if isinstance(msg["content"], list):
                content = []
                for block in msg["content"]:
                    if block["type"] == "text":
                        content.append({"type": "text", "text": block["text"]})
                    elif block["type"] == "image":
                        src = block["source"]
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"}
                        })
                oai_messages.append({"role": msg["role"], "content": content})
            else:
                oai_messages.append({"role": msg["role"], "content": msg["content"]})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=req.max_tokens,
            messages=oai_messages,
        )
        return {
            "content": [{"type": "text", "text": response.choices[0].message.content}]
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=False, ssl_keyfile="key.pem", ssl_certfile="cert.pem")