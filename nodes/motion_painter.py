import asyncio
import base64
import glob
from io import BytesIO
import os
import json
from PIL import Image, ImageOps
import numpy as np
from comfy_execution.graph import ExecutionBlocker
import folder_paths
from server import PromptServer
from aiohttp import web
import hashlib
from ..common.tree import *
from ..common.constants import *

# Directory node save settings
CHUNK_SIZE = 1024
dir_painter_node = os.path.dirname(__file__)
extension_path = os.path.join(os.path.abspath(dir_painter_node))
nodes_settings_path = os.path.join(extension_path, "settings_nodes")


# Create directory settings_nodes if not exists
if not os.path.exists(nodes_settings_path):
    os.mkdir(nodes_settings_path)

    tipsfile = os.path.join(nodes_settings_path, "Stores painter nodes settings.txt")
    with open(tipsfile, "w+", encoding="utf-8") as tipsfile:
        tipsfile.write("Painter node saved settings!")


# Function create file json file
PREFIX = "_setting.json"


def isFileName(filename):
    if (
        not filename
        and filename is not None
        and (type(filename) == str and filename.strip() == "")
    ):
        print("Filename is incorrect")
        return False
    return True


def create_settings_json(filename):
    try:
        json_file = os.path.join(nodes_settings_path, filename)
        if not os.path.isfile(json_file):
            print(f"File settings for '{filename}' is not found! Create file!")
            with open(json_file, "w") as f:
                json.dump({}, f)

    except Exception as e:
        print(f"Error: ${e}")


def get_settings_json(filename, notExistCreate=True):
    if not isFileName(filename):
        return {}

    json_file = os.path.join(nodes_settings_path, filename)
    if os.path.isfile(json_file):
        f = open(json_file, "rb")
        try:
            load_data = json.load(f)
            return load_data
        except Exception as e:
            print("Error load json file: ", e)
            if notExistCreate:
                f.close()
                os.remove(json_file)
                create_settings_json(filename)
        finally:
            f.close()
    else:
        create_settings_json(filename)

    return {}


# Load json file
@PromptServer.instance.routes.get("/mi2v/loading_node_settings/{nodeName}")
async def loadingSettings(request):
    filename = request.match_info.get("nodeName", None)
    if not isFileName(filename):
        load_data = {}
    else:
        load_data = get_settings_json(filename + PREFIX)

    return web.json_response({"settings_nodes": load_data})


# Load json's files
@PromptServer.instance.routes.get("/mi2v/loading_all_node_settings")
async def loadingAllSettings(request):
    load_data = []
    jsonFiles = glob.glob("Paint_*.json", root_dir=nodes_settings_path)

    for f in jsonFiles:
        path_to_file = os.path.join(nodes_settings_path, f)

        if os.path.isfile(path_to_file):
            file = open(path_to_file, "rb")
            try:
                jsonData = json.load(file)
                load_data.append({"name":f.replace(PREFIX,""), "value": jsonData})
            except Exception as e:
                print("Error load json file: ", e)

            finally:
                file.close()
        else:
            print(f"File {f} not file!")

    return web.json_response({"all_settings_nodes": load_data})


# Save data to json file
@PromptServer.instance.routes.post("/mi2v/save_node_settings")
async def saveSettings(request):
    try:
        if not request.content_type.startswith("multipart/"):
            return web.json_response(
                {"error": "multipart/* content type expected"}, status=400
            )

        reader = await request.multipart()
        filename_reader = await reader.next()
        filename = await filename_reader.text()

        data_reader = await reader.next()

        if isFileName(filename):
            filename = filename + PREFIX
            json_file = os.path.join(nodes_settings_path, filename)

            if os.path.isfile(json_file):
                with open(json_file, "wb") as f:
                    while True:
                        chunk = await data_reader.read_chunk(size=CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)

                return web.json_response(
                    {"message": "Painter data saved successfully"}, status=200
                )

            else:
                create_settings_json(filename)
                return web.json_response(
                    {"message": "Painter file settings created!"}, status=200
                )

        else:
            raise Exception("Filename is not found or incorrect!")

    except Exception as e:
        print("Error save json file: ", e)
        return web.json_response({"error": str(e)}, status=500)


# Remove file settings painter node data
@PromptServer.instance.routes.post("/mi2v/remove_node_settings")
async def removeSettings(request):
    try:
        json_data = await request.json()
        filename = json_data.get("name")

        if isFileName(filename):
            filename = filename + PREFIX
            json_file = os.path.join(nodes_settings_path, filename)

            os.remove(json_file)
            return web.json_response(
                {"message": "Painter data removed successfully"}, status=200
            )

    except OSError as e:
        return web.json_response(
            {"error": "Error: %s - %s." % (e.filename, e.strerror)}, status=500
        )


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

class MI2V_MotionPainter(object):
    @classmethod
    def INPUT_TYPES(cls):
        cls.canvas_set = False
        
        work_dir = folder_paths.get_input_directory()
        imgs = [
            img
            for img in os.listdir(work_dir)
            if os.path.isfile(os.path.join(work_dir, img))
        ]

        return {
            "required": {
                "image": ("IMAGE",),
                # "image": (sorted(imgs),),
                "arrows": ("STRING", {"multiline": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "execute"
    CATEGORY = TREE_FLOW

    DESCRIPTION = """Draw arrows to describe the desired motion. Pause execution to load in the correctly sized image, add your arrows, then unpause and run again. LEFT DRAG = Place an Arrow, RIGHT CLICK = Remove an Arrow"""

    def __setattr__(self, name, value):
        print(f"SETTING {name} = {value}")
        super().__setattr__(name, value)

    def execute(self, image, arrows, unique_id):
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
        return (image, arrows)

    # @classmethod
    # def IS_CHANGED(self, image, arrows, unique_id, paupause_execution):
    #     m = hashlib.sha256()
    #     m.update(str(arrows).encode('utf-8'))
    #     return m.hexdigest()

class MI2V_PauseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "motion_vectors": ("STRING", {"default": "", "forceInput": True}),
                "motion_mask": ("MASK", {"default": None, "forceInput": True}),
                "images": ("IMAGE", {"default": None, "forceInput": True}),
                "pause_execution": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")  # Outputs the same data type as input
    FUNCTION = "execute"
    CATEGORY = TREE_FLOW

    DESCRIPTION = """Pause the execution pipeline based on an optional boolean input.
    - **Data**: Any input data that needs to be passed through or paused.
    - **Pause Execution**: Set to `True` to pause the pipeline.
    When `pause_execution` is `True`, the pipeline stops and awaits further instructions.
    """

    def execute(self, motion_vectors="", motion_mask=None, images=None, pause_execution=False):
        """
        Executes the Pause Node logic.

        Args:
            data: The input data to pass through.
            pause_execution (bool): Whether to pause the pipeline.

        Returns:
            tuple: Outputs the data or an ExecutionBlocker.
        """
        if pause_execution:
            return (ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None), )
        else:
            # Pass the data through unchanged
            return (motion_mask, images, motion_vectors,)