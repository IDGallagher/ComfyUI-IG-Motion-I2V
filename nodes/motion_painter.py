import asyncio
import base64
from io import BytesIO
import os
import json
from PIL import Image, ImageOps
import numpy as np
from comfy_execution.graph import ExecutionBlocker
from server import PromptServer
from aiohttp import web
import hashlib

# Piping image
PAINTER_DICT = {}  # Painter nodes dict instances

def toBase64ImgUrl(img):
    bytesIO = BytesIO()
    img.save(bytesIO, format="PNG")
    img_types = bytesIO.getvalue()
    img_base64 = base64.b64encode(img_types)
    return f"data:image/png;base64,{img_base64.decode('utf-8')}"

@PromptServer.instance.routes.post("/mi2v/check_canvas_changed")
async def check_canvas_changed(request):
    json_data = await request.json()
    unique_id = json_data.get("unique_id", None)
    is_ok = json_data.get("is_ok", False)

    if unique_id is not None and unique_id in PAINTER_DICT and is_ok == True:
        PAINTER_DICT[unique_id].canvas_set = True
        return web.json_response({"status": "Ok"})

    return web.json_response({"status": "Error"})

async def wait_canvas_change(unique_id, time_out=40):
    for _ in range(time_out):
        if (
            hasattr(PAINTER_DICT[unique_id], "canvas_set")
            and PAINTER_DICT[unique_id].canvas_set == True
        ):
            PAINTER_DICT[unique_id].canvas_set = False
            return True

        await asyncio.sleep(0.1)

    return False

# end - Piping image


class MI2V_MotionPainter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            },
            "optional": {
                "pause": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "VECTOR_DATA")
    FUNCTION = "execute"
    CATEGORY = "Custom Nodes"

    def execute(self, image, unique_id, pause):
        # Piping image input
        if unique_id not in PAINTER_DICT:
            PAINTER_DICT[unique_id] = self

        if image is not None:

            input_images = []

            for imgs in image:
                i = 255.0 * imgs.cpu().numpy()
                i = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                input_images.append(toBase64ImgUrl(i))

            PAINTER_DICT[unique_id].canvas_set = False

            PromptServer.instance.send_sync(
                "mi2v_get_image", {"unique_id": unique_id, "images": input_images}
            )
            if not asyncio.run(wait_canvas_change(unique_id)):
                print(f"MotionPainter_{unique_id}: Failed to get image!")
            else:
                print(f"MotionPainter_{unique_id}: Image received, canvas changed!")
        # end - Piping image input
        # The actual processing will be handled on the client side.
        return (image if not pause else ExecutionBlocker(None), None)

    @classmethod
    def IS_CHANGED(cls, image, unique_id):
        # Use the hash of the image and unique ID to determine if the node has changed.
        m = hashlib.sha256()
        m.update(str(unique_id).encode('utf-8'))
        # Assuming 'image' is a filepath or an object that can be hashed.
        if isinstance(image, str) and os.path.isfile(image):
            with open(image, "rb") as f:
                m.update(f.read())
        return m.hexdigest()
    
# class MI2V_MotionPainter:
#     @classmethod
#     def INPUT_TYPES(cls):
#         input_dir = folder_paths.get_input_directory()
#         files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
#         return {"required":
#                     {"image": (sorted(files), {"image_upload": True})},
#                 }

#     RETURN_TYPES = ("IMAGE", "VECTOR_DATA")
#     FUNCTION = "execute"
#     CATEGORY = "Custom Nodes"

#     def execute(self, image):
#         # The actual processing will be handled on the client side.
#         return (image, None)

#     @classmethod
#     def IS_CHANGED(cls, image):
#         # Use the hash of the image and unique ID to determine if the node has changed.
#         m = hashlib.sha256()
#         # Assuming 'image' is a filepath or an object that can be hashed.
#         if isinstance(image, str) and os.path.isfile(image):
#             with open(image, "rb") as f:
#                 m.update(f.read())
#         return m.hexdigest()
