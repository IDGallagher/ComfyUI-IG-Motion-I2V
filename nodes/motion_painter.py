import os
import json
from server import PromptServer
from aiohttp import web
import hashlib

# Directory to save node settings
SETTINGS_DIR = os.path.join(os.path.dirname(__file__), "settings_nodes")
os.makedirs(SETTINGS_DIR, exist_ok=True)

PREFIX = "_settings.json"

def is_valid_filename(filename):
    return filename and isinstance(filename, str) and filename.strip() != ""

def get_settings_filepath(filename):
    return os.path.join(SETTINGS_DIR, filename + PREFIX)

def load_settings(filename):
    filepath = get_settings_filepath(filename)
    if os.path.isfile(filepath):
        with open(filepath, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return {}

def save_settings(filename, data):
    filepath = get_settings_filepath(filename)
    with open(filepath, "w") as f:
        json.dump(data, f)

@PromptServer.instance.routes.get("/motion_vector/load_settings/{node_name}")
async def load_settings_route(request):
    filename = request.match_info.get("node_name", None)
    data = load_settings(filename) if is_valid_filename(filename) else {}
    return web.json_response({"settings": data})

@PromptServer.instance.routes.post("/motion_vector/save_settings")
async def save_settings_route(request):
    try:
        data = await request.json()
        filename = data.get("name")
        settings = data.get("settings")
        if is_valid_filename(filename):
            save_settings(filename, settings)
            return web.json_response({"message": "Settings saved successfully"}, status=200)
        else:
            raise ValueError("Invalid filename")
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

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
        }

    RETURN_TYPES = ("IMAGE", "VECTOR_DATA")
    FUNCTION = "execute"
    CATEGORY = "Custom Nodes"

    def execute(self, image, unique_id):
        # The actual processing will be handled on the client side.
        return (image, None)

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
