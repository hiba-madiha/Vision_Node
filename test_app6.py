# test_app5.py  (updated)
import argparse
import base64
import os
import platform
import sys
import threading
import time
from pathlib import Path

import torch
import uvicorn
from digi.xbee.devices import RemoteXBeeDevice, XBeeDevice, XBee64BitAddress, XBeeException
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pymongo import MongoClient  # noqa: F401  # (left for future use)

from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import (
    IMG_FORMATS,
    VID_FORMATS,
    LoadImages,
    LoadScreenshots,
    LoadStreams,
)
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    cv2,
    non_max_suppression,
    scale_boxes,
)
from utils.torch_utils import select_device, smart_inference_mode

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI initialisation
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# ─────────────────────────────────────────────────────────────────────────────
# ZigBee / transmission configuration (tweak as needed)
# ─────────────────────────────────────────────────────────────────────────────
XBEE_PORT = "/dev/ttyUSB0"               # change to COM3 on Windows e.g. "COM3"
XBEE_BAUD = 115200                       # must match coordinator
REMOTE_64BIT = "0013A200422A4D2A"        # coordinator 64-bit address (change if needed)
CHUNK_SIZE = 60                          # number of base64 chars per chunk (ascii-safe)
ACK_TIMEOUT = 0.5                        # seconds to wait for ACK from coordinator
MAX_RETRIES = 3                          # retries per chunk if no ACK received
LABEL_SLEEP = 0.1                        # small pause after sending label/start/end

# ─────────────────────────────────────────────────────────────────────────────
# Globals & configuration (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
opt = None  # set at startup
video_stream_24Hours = 0
video_stream_ondemand = 0
sent_first_frame = False  # ensures first-frame push only once per burst

video_duration = 10  # seconds to keep recording after last detection

is_recording = False
out = None
last_detection_time = 0.0  # unix epoch seconds
video_count = 1

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: wait for ACK from coordinator
# ─────────────────────────────────────────────────────────────────────────────
def _wait_for_ack(xbee_device: XBeeDevice, frame_id: int, chunk_id: int, timeout: float):
    """
    Wait for an ACK message in the format: "ACK|<frame_id>|<chunk_id>"
    Returns True if ACK received, False on timeout.
    """
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            msg = xbee_device.read_data(timeout=0.1)  # non-blocking small increments
            if msg is None:
                continue
            try:
                text = msg.data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            if not text:
                continue
            if text.startswith("ACK|"):
                parts = text.strip().split("|")
                if len(parts) >= 3:
                    try:
                        r_frame = int(parts[1])
                        r_chunk = int(parts[2])
                    except Exception:
                        continue
                    if r_frame == frame_id and r_chunk == chunk_id:
                        return True
        except XBeeException:
            # transient read error: continue trying until timeout
            continue
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Robust sender: base64-encode JPEG, chunk it, send with ACKs
# ─────────────────────────────────────────────────────────────────────────────
def _send_image_over_zigbee(img_path: str, label: str):
    """
    Send label + JPEG image over ZigBee with chunk IDs and frame ID, base64-encoded.
    Uses stop-and-wait ACK per chunk. Coordinator must reply with:
        ACK|<frame_id>|<chunk_id>
    """
    xbee = None
    try:
        # open device
        xbee = XBeeDevice(XBEE_PORT, XBEE_BAUD)
        xbee.open()
        remote = RemoteXBeeDevice(xbee, XBee64BitAddress.from_hex_string(REMOTE_64BIT))

        # send a short label message first (pure text)
        try:
            xbee.send_data(remote, f"Node1:camerazones:{label}")
        except Exception as e:
            LOGGER.warning(f"[ZigBee] label send failed: {e}")
        time.sleep(LABEL_SLEEP)

        # read & base64 encode the JPEG (makes data ASCII-safe)
        with open(img_path, "rb") as fh:
            raw = fh.read()
        b64 = base64.b64encode(raw).decode("ascii")

        # chunk the base64 string (CHUNK_SIZE characters each)
        total_chunks = len(b64) // CHUNK_SIZE + (1 if len(b64) % CHUNK_SIZE else 0)
        frame_id = int(time.time())  # unique per image (epoch seconds)

        # send start control
        xbee.send_data(remote, f"IMG_START|{frame_id}|{total_chunks}")
        time.sleep(LABEL_SLEEP)

        # send chunks with stop-and-wait ACKs
        for i in range(total_chunks):
            start = i * CHUNK_SIZE
            chunk_str = b64[start : start + CHUNK_SIZE]
            packet_txt = f"{frame_id}|{i}|{chunk_str}"
            sent = False
            tries = 0
            while not sent and tries < MAX_RETRIES:
                tries += 1
                try:
                    xbee.send_data(remote, packet_txt)
                except Exception as e:
                    LOGGER.warning(f"[ZigBee] send chunk failed (try {tries}): {e}")
                    time.sleep(0.01)
                    continue

                # wait for ACK
                if _wait_for_ack(xbee, frame_id, i, ACK_TIMEOUT):
                    sent = True
                else:
                    LOGGER.warning(f"[ZigBee] no ACK for frame {frame_id} chunk {i} (try {tries})")
                    time.sleep(0.01)  # small backoff before retry

            if not sent:
                LOGGER.warning(f"[ZigBee] giving up on chunk {i} after {MAX_RETRIES} retries")
                # Option A: abort entire frame
                # xbee.send_data(remote, f"IMG_ABORT|{frame_id}")
                # return
                # Option B: continue and let receiver timeout; here we abort
                break

            # small pacing so coordinator has time to process
            time.sleep(0.01)

        # send end control (even if some chunks failed, send end so coordinator can clean)
        try:
            xbee.send_data(remote, f"IMG_END|{frame_id}")
        except Exception as e:
            LOGGER.warning(f"[ZigBee] send IMG_END failed: {e}")

        LOGGER.info(f"[ZigBee] finished sending frame {frame_id} (expected chunks={total_chunks})")

    except Exception as exc:
        LOGGER.warning(f"[ZigBee] transmission failed: {exc}")
    finally:
        if xbee is not None:
            try:
                xbee.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Core inference / streaming routine (unchanged except for using new sender)
# ─────────────────────────────────────────────────────────────────────────────
@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=500,
    device="",
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):
    global is_recording, out, last_detection_time, video_count, sent_first_frame

    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    torch_device = select_device(device)
    model = DetectMultiBackend(weights, device=torch_device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=torch_device), Profile(device=torch_device), Profile(device=torch_device))

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255.0

        with dt[1]:
            visualize = False
            pred = model(im, augment=augment, visualize=visualize)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            s += "{:g}x{:g} ".format(*im.shape[2:])
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if hide_conf else f"{names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()

            # ───────── First-frame ZigBee push logic ─────────
            if time.time() - last_detection_time > 600:
                sent_first_frame = False  # reset after silence window

            if len(det) > 0 and not sent_first_frame:
                sent_first_frame = True
                last_detection_time = time.time()
                label2 = names[int(det[0][5])]

                # Save JPEG & spawn background thread
                img_tmp = "first_frame1.jpg"

                # Resize & grayscale to reduce size — adjust if you want color / larger resolution
                resize_img = cv2.resize(im0, (100, 100))
                gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(img_tmp, gray_img)

                threading.Thread(
                    target=_send_image_over_zigbee,
                    args=(img_tmp, label2),
                    daemon=True,
                ).start()

            # ───────── Streaming section ─────────
            if video_stream_24Hours == 1 and video_stream_ondemand == 0:
                ret, buffer = cv2.imencode(".jpg", im0)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

            if video_stream_ondemand == 1 and video_stream_24Hours == 0:
                if len(det) > 0:
                    last_detection_time = time.time()
                    if not is_recording:
                        filename = f"video_{video_count}.mp4"
                        h, w = im0.shape[:2]
                        out = cv2.VideoWriter(filename, fourcc, 30, (w, h))
                        is_recording = True
                        LOGGER.info("[REC] started → %s", filename)
                        video_count += 1
                if is_recording:
                    out.write(im0)
                    ret, buffer = cv2.imencode(".jpg", im0)
                    if not ret:
                        continue
                    frame_bytes = buffer.tobytes()
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                    if time.time() - last_detection_time > video_duration:
                        out.release()
                        is_recording = False
                        LOGGER.info("[REC] stopped – idle too long")

            # optional local preview
            view_img = False
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        LOGGER.info(
            f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms"
        )

    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(
        "Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape %s",
        *t,
        (1, 3, *imgsz),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI routes (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return {
        "title": "Object Detection API",
        "endpoints": {
            "/video_feed": "24-hour stream (always on)",
            "/video_ondemand": "only stream when object detected",
        },
    }


@app.get("/video_feed")
async def video_feed():
    global video_stream_24Hours, video_stream_ondemand
    video_stream_24Hours = 1
    video_stream_ondemand = 0
    return StreamingResponse(run(**vars(opt)), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video_ondemand")
async def video_ondemand():
    global video_stream_24Hours, video_stream_ondemand
    video_stream_24Hours = 0
    video_stream_ondemand = 1
    return StreamingResponse(run(**vars(opt)), media_type="multipart/x-mixed-replace; boundary=frame")


# ─────────────────────────────────────────────────────────────────────────────
# CLI & startup glue
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def _on_startup():
    global opt
    opt = _parse_opt()
    check_requirements(exclude=("tensorboard",))


def _parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path")
    parser.add_argument(
        "--source",
        type=str,
        default=ROOT / "data/images",
        help="file/dir/URL/glob/screen/0(webcam)",
    )
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class IDs")

    args = parser.parse_args()
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1
    return args


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
