import base64
import io
import json
import math
import os
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


# =========================
# Seed plans
# =========================


def seed_plans() -> Dict[str, PlanDefinition]:
    return {
        "plan_a": PlanDefinition(
            id="plan_a",
            name="Plan A (Fastest)",
            badge="Recommended",
            time_estimate="10-12m",
            description="Arduino blink + single LED setup. Direct and efficient.",
            required_materials=["breadboard", "arduino nano", "led", "220Ω resistor", "jumper wire", "usb cable"],
            steps=[
                StepDefinition(
                    id="step_led",
                    title="Place the LED on the breadboard",
                    subtitle="Insert the two legs across the center gap.",
                    icon="💡",
                    goal="The LED should sit in the highlighted upper-middle breadboard area.",
                    completion_check="led_region",
                    expected_objects=["breadboard", "led"],
                    overlay_hint="Highlight the center gap and show a downward placement arrow.",
                    target_region={"x": 0.48, "y": 0.37, "w": 0.18, "h": 0.16},
                    instruction_prompt="Explain how to place one LED across the breadboard center gap for a beginner.",
                    ai_reference_prompt="A top-down electronics wiring reference image showing one LED correctly inserted across the breadboard center gap, clean background, tutorial style.",
                ),
                StepDefinition(
                    id="step_resistor",
                    title="Wire the resistor",
                    subtitle="Connect a 220Ω resistor from the LED cathode row to ground row.",
                    icon="🟫",
                    goal="A resistor should be visible in the lower-left wiring region.",
                    completion_check="resistor_region",
                    expected_objects=["breadboard", "resistor", "led"],
                    overlay_hint="Highlight the ground-side resistor path and show a left-to-right arrow.",
                    target_region={"x": 0.33, "y": 0.56, "w": 0.28, "h": 0.12},
                    instruction_prompt="Explain how to connect a resistor from the LED cathode row to GND on a breadboard.",
                    ai_reference_prompt="A top-down electronics wiring reference image showing a single LED and a correctly placed 220 ohm resistor connected to the ground side on a breadboard, tutorial style.",
                ),
                StepDefinition(
                    id="step_nano",
                    title="Connect the Arduino Nano",
                    subtitle="Route one jumper from a digital pin to the LED row and one to GND.",
                    icon="🧠",
                    goal="The Arduino Nano should appear on the right side and be connected by jumper wires.",
                    completion_check="nano_region",
                    expected_objects=["arduino nano", "jumper wire", "breadboard"],
                    overlay_hint="Highlight the right-side Nano placement zone and show two routing arrows toward the breadboard.",
                    target_region={"x": 0.76, "y": 0.42, "w": 0.2, "h": 0.3},
                    instruction_prompt="Explain how to connect an Arduino Nano to a breadboard LED circuit using one signal jumper and one ground jumper.",
                    ai_reference_prompt="A top-down wiring reference showing an Arduino Nano on the right side connected with jumper wires to a breadboard LED and resistor circuit, tutorial style.",
                ),
                StepDefinition(
                    id="step_test",
                    title="Test the blink setup",
                    subtitle="Make sure the physical scene matches the target layout before uploading code.",
                    icon="✅",
                    goal="The overall scene should match the intended LED blink wiring layout.",
                    completion_check="scene_match",
                    expected_objects=["arduino nano", "breadboard", "led", "resistor"],
                    overlay_hint="Highlight the whole build and show a confirmation ring around the completed circuit.",
                    target_region={"x": 0.5, "y": 0.5, "w": 0.78, "h": 0.7},
                    instruction_prompt="Explain how a beginner should visually verify an Arduino Nano single LED blink circuit before testing.",
                    ai_reference_prompt="A complete top-down electronics build reference showing a correct Arduino Nano single LED blink setup on a breadboard, tutorial style.",
                ),
            ],
        ),
        "plan_c": PlanDefinition(
            id="plan_c",
            name="Plan C (Most Elaborative)",
            badge="Advanced",
            time_estimate="25m",
            description="Includes decorative blinking patterns and power-save mode.",
            required_materials=["breadboard", "arduino nano", "led", "220Ω resistor", "jumper wire", "usb cable"],
            steps=[
                StepDefinition(
                    id="step_led_c",
                    title="Mount the LED",
                    subtitle="Place the LED so its legs sit on separate rows.",
                    icon="💡",
                    goal="LED is inserted in the highlighted region.",
                    completion_check="led_region",
                    expected_objects=["breadboard", "led"],
                    overlay_hint="Show a center placement overlay for the LED.",
                    target_region={"x": 0.49, "y": 0.35, "w": 0.2, "h": 0.15},
                    instruction_prompt="Explain how to mount an LED on a breadboard for a decorative blinking project.",
                    ai_reference_prompt="A clean top-down tutorial reference of one LED inserted correctly on a breadboard.",
                ),
                StepDefinition(
                    id="step_resistor_c",
                    title="Add the resistor path",
                    subtitle="Bridge the LED ground side through a resistor.",
                    icon="🟫",
                    goal="Resistor appears in the target wiring band.",
                    completion_check="resistor_region",
                    expected_objects=["resistor", "breadboard", "led"],
                    overlay_hint="Show a horizontal resistor highlight near the lower breadboard rows.",
                    target_region={"x": 0.34, "y": 0.56, "w": 0.26, "h": 0.11},
                    instruction_prompt="Explain how to place a resistor for a safe LED breadboard circuit.",
                    ai_reference_prompt="Top-down tutorial image showing the resistor correctly connected to an LED on a breadboard.",
                ),
                StepDefinition(
                    id="step_nano_c",
                    title="Seat the Arduino Nano",
                    subtitle="Place the Nano and route the jumper wires.",
                    icon="🧠",
                    goal="Nano appears in the right-side zone with wires connected.",
                    completion_check="nano_region",
                    expected_objects=["arduino nano", "jumper wire"],
                    overlay_hint="Show a placement box on the right and wire routing arrows.",
                    target_region={"x": 0.77, "y": 0.42, "w": 0.22, "h": 0.32},
                    instruction_prompt="Explain how to place and connect an Arduino Nano for a breadboard LED project.",
                    ai_reference_prompt="Top-down tutorial image with Arduino Nano on the right side wired to a breadboard LED circuit.",
                ),
                StepDefinition(
                    id="step_verify_c",
                    title="Verify decorative layout",
                    subtitle="Check that all visible parts match the intended scene.",
                    icon="🎛️",
                    goal="The scene should match the expected decorative layout.",
                    completion_check="scene_match",
                    expected_objects=["arduino nano", "breadboard", "led", "resistor"],
                    overlay_hint="Outline the entire circuit as a verification target.",
                    target_region={"x": 0.5, "y": 0.5, "w": 0.8, "h": 0.72},
                    instruction_prompt="Explain how to visually verify the breadboard layout before demonstrating decorative blink behavior.",
                    ai_reference_prompt="Top-down tutorial image of a complete Arduino Nano LED breadboard layout prepared for decorative blinking behavior.",
                ),
            ],
        ),
    }


PLANS = seed_plans()
SESSIONS: Dict[str, SessionState] = {}


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


class PrototypeVisionService:
    """
    Prototype CV service:
    - OpenCV color segmentation as lightweight detector proxy
    - MediaPipe hands to infer active manipulation
    - YOLO placeholder is represented as a hook in self.detect_objects()
    """

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

        detections: Dict[str, List[Dict[str, Any]]] = {
            "led": self._detect_red_led(hsv, w, h),
            "resistor": self._detect_resistor(hsv, w, h),
            "arduino nano": self._detect_blue_board(hsv, w, h),
            "breadboard": [{"bbox": {"x": 0.5, "y": 0.5, "w": 0.6, "h": 0.45}, "score": 0.55}],
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

        overlay = self._build_overlay(step, detections, hands)
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
        target = step.target_region

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

    def _build_overlay(
        self,
        step: StepDefinition,
        detections: Dict[str, List[Dict[str, Any]]],
        hands: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        target = step.target_region
        overlays = {
            "highlight_box": target,
            "label": step.title,
            "hint": step.overlay_hint,
            "arrows": [self._make_arrow_for_target(target)],
            "detected_boxes": [],
            "hand_points": [{"x": x, "y": y} for x, y in hands],
        }
        for label, items in detections.items():
            for item in items:
                bbox = item.get("bbox")
                if bbox:
                    overlays["detected_boxes"].append({"label": label, "bbox": bbox, "score": item.get("score", 0.0)})
        return overlays

    def _make_arrow_for_target(self, target: Dict[str, float]) -> Dict[str, Any]:
        return {
            "x1": clamp01(target["x"]),
            "y1": clamp01(target["y"] - target["h"] * 0.9),
            "x2": clamp01(target["x"]),
            "y2": clamp01(target["y"] - target["h"] * 0.45),
        }

    def _detect_red_led(self, hsv: np.ndarray, w: int, h: int) -> List[Dict[str, Any]]:
        lower1 = np.array([0, 70, 50])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 70, 50])
        upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        return self._mask_to_boxes(mask, w, h, min_area=120, label_bias=(0.0, 0.0))

    def _detect_resistor(self, hsv: np.ndarray, w: int, h: int) -> List[Dict[str, Any]]:
        lower = np.array([8, 40, 70])
        upper = np.array([25, 190, 220])
        mask = cv2.inRange(hsv, lower, upper)
        return self._mask_to_boxes(mask, w, h, min_area=200, label_bias=(0.0, 0.0))

    def _detect_blue_board(self, hsv: np.ndarray, w: int, h: int) -> List[Dict[str, Any]]:
        lower = np.array([90, 60, 50])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        return self._mask_to_boxes(mask, w, h, min_area=600, label_bias=(0.0, 0.0))

    def _mask_to_boxes(self, mask: np.ndarray, w: int, h: int, min_area: int, label_bias: Tuple[float, float]) -> List[Dict[str, Any]]:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Dict[str, Any]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            boxes.append(
                {
                    "bbox": {
                        "x": clamp01((x + bw / 2) / w + label_bias[0]),
                        "y": clamp01((y + bh / 2) / h + label_bias[1]),
                        "w": clamp01(bw / w),
                        "h": clamp01(bh / h),
                    },
                    "score": round(min(0.95, 0.45 + area / (w * h)), 3),
                }
            )
        boxes.sort(key=lambda item: item["score"], reverse=True)
        return boxes[:5]


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
    current_instruction = planner_service.beginner_instruction(step)
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
    return {"plans": [item.model_dump() for item in build_plan_summaries(materials)]}


@app.post("/api/session/start")
def start_session(req: StartSessionRequest) -> Dict[str, Any]:
    inventory = req.inventory or ["breadboard", "arduino nano", "led", "220Ω resistor", "jumper wire", "usb cable"]

    if req.plan_id in PLANS:
        plan = PLANS[req.plan_id]
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
    if session.current_step_index < len(session.steps) - 1:
        session.current_step_index += 1
        session.last_analysis = {}
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
    image = decode_image(req.image_base64)
    result = vision_service.analyze_step(session, image, req.image_width, req.image_height)
    if result["matched"] and session.current_step_index < len(session.steps) - 1:
        result["ready_for_next"] = True
    else:
        result["ready_for_next"] = session.current_step_index == len(session.steps) - 1 and result["matched"]
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
    import urllib.request as _urllib_request

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
    )
    try:
        with _urllib_request.urlopen(http_req) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=False, ssl_keyfile="key.pem", ssl_certfile="cert.pem")