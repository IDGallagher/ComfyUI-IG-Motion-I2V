import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { fabric } from "./lib/painternode/fabric.js";
import { toRGBA, getColorHEX, LS_Class } from "./lib/painternode/helpers.js";
import {
  isEmptyObject,
} from "./utils.js";

// ================= FUNCTIONS ================

// Save settings in JSON file on the extension folder [big data settings includes images] if true else localStorage
let painters_settings_json = true
//

function resizeCanvas(node, sizes) {
  const { width, height } = sizes ?? node.painter.currentCanvasSize;

  node.painter.canvas.setDimensions({
    width: width,
    height: height,
  });

  node.painter.canvas.getElement().width = width;
  node.painter.canvas.getElement().height = height;

  node.painter.canvas.renderAll();
  app.graph.setDirtyCanvas(true, false);
}
// ================= END FUNCTIONS ================

// ================= CLASS PAINTER ================
class Painter {
  constructor(node, canvas) {
    this.originX = 0;
    this.originY = 0;
    this.drawning = true;
    this.mode = false;
    this.type = "Arrow";
    this.arrows = [];
    this.isDown = false;
    this.arrow = null;

    this.locks = {
      lockMovementX: false,
      lockMovementY: false,
      lockScalingX: false,
      lockScalingY: false,
      lockRotation: false,
    };

    this.currentCanvasSize = { width: 512, height: 512 };
    this.maxNodeSize = 1024;

    this.fonts = {
      Arial: "arial",
      "Times New Roman": "Times New Roman",
      Verdana: "verdana",
      Georgia: "georgia",
      Courier: "courier",
      "Comic Sans MS": "comic sans ms",
      Impact: "impact",
    };

    this.bringFrontSelected = true;

    this.node = node;
    this.history_change = false;
    this.canvas = this.initCanvas(canvas);
    this.image = node.widgets.find((w) => w.name === "image");

    let default_value = this.image.value;
    Object.defineProperty(this.image, "value", {
      set: function (value) {
        this._real_value = value;
      },

      get: function () {
        let value = "";
        if (this._real_value) {
          value = this._real_value;
        } else {
          return default_value;
        }

        if (value.filename) {
          let real_value = value;
          value = "";
          if (real_value.subfolder) {
            value = real_value.subfolder + "/";
          }

          value += real_value.filename;

          if (real_value.type && real_value.type !== "input")
            value += ` [${real_value.type}]`;
        }
        return value;
      },
    });
  }

  initCanvas(canvasEl) {
    this.canvas = new fabric.Canvas(canvasEl, {
      isDrawingMode: false,
      selection: false,
      backgroundColor: "transparent",
      width: 512,
      height: 512,
      fireRightClick: true,
      enablePointerEvents: true,
      stopContextMenu: true,
    });

    this.canvas.backgroundColor = "#000000";

    return this.canvas;
  }

  makeArrow(fromX, fromY, toX, toY) {
    var headlen = 20; // length of head in pixels
    var offset = -10;
    var perpOffset = 3;
    var angle = Math.atan2(toY - fromY, toX - fromX);
    var length = Math.sqrt((toX - fromX)**2 + (toY - fromY)**2);
    
    var newToX = toX - (toX - fromX) * offset / length + perpOffset * Math.sin(angle) * (toY - fromY) / Math.abs(toY - fromY);
    var newToY = toY - (toY - fromY) * offset / length + perpOffset * Math.cos(angle) * (toX - fromX) / Math.abs(toX - fromX);

    // Calculate the arrowhead points
    var arrowX1 = newToX - headlen * Math.cos(angle - Math.PI / 6);
    var arrowY1 = newToY - headlen * Math.sin(angle - Math.PI / 6);
    var arrowX2 = newToX - headlen * Math.cos(angle + Math.PI / 6);
    var arrowY2 = newToY - headlen * Math.sin(angle + Math.PI / 6);
  
    // Create the main line of the arrow
    let strokWidth = 6;
    var line = new fabric.Line([fromX, fromY, toX, toY], {
      stroke: toRGBA("#FF0000", 1.0),
      strokeWidth: strokWidth,
    });
    var line2 = new fabric.Line([fromX+1, fromY+1, toX+1, toY+1], {
        stroke: toRGBA("#000000", 1.0),
        strokeWidth: strokWidth,
    });
    var line3 = new fabric.Line([fromX-1, fromY-1, toX-1, toY-1], {
        stroke: toRGBA("#000000", 1.0),
        strokeWidth: strokWidth,
    });
  
    // Create the arrowhead as a polygon
    var arrowhead = new fabric.Polygon(
      [
        { x: newToX, y: newToY },
        { x: arrowX1, y: arrowY1 },
        { x: arrowX2, y: arrowY2 },
      ],
      {
        fill: toRGBA("#FF0000", 1.0),
        stroke: toRGBA("#000000", 1.0),
        strokeWidth: 1,
      }
    );
  
    // Group the line and arrowhead together
    var arrow = new fabric.Group([line3, line2, line, arrowhead], {
      selectable: false,
      lockMovementX: true,
      lockMovementY: true,
      hasBorders: false,
      hasControls: false,
    });
  
    return arrow;
  }

  propertiesLS() {
    let settingsNode = this.node.LS_Cls.LS_Painters.settings;

    if (!settingsNode) {
      settingsNode = this.node.LS_Cls.LS_Painters.settings = {
        lsSavePainter: true,
        pipingSettings: {
          action: {
            name: "background",
            options: {},
          },
          pipingChangeSize: true,
          pipingUpdateImage: true,
        },
      };
    }

    // Save canvas to localStorage if not exists
    if (typeof settingsNode?.lsSavePainter !== "boolean") {
      settingsNode.lsSavePainter = true;
    }

    // Piping settings localStorage if not exists
    if (!settingsNode?.pipingSettings) {
      settingsNode.pipingSettings = {
        action: {
          name: "background",
          options: {},
        },
        pipingChangeSize: true,
        pipingUpdateImage: true,
      };
    }
  }

  clearCanvas() {
    this.canvas.clear();
    this.canvas.backgroundColor = "#000000";
    this.canvas.requestRenderAll();

    this.canvasSaveSettingsPainter();
  }

  setCanvasSize(new_width, new_height, confirmChange = false) {
    if (
      confirmChange &&
      this.node.isInputConnected(0) &&
      this.node.LS_Cls.LS_Painters.settings.pipingSettings.pipingChangeSize &&
      (new_width !== this.currentCanvasSize.width ||
        new_height !== this.currentCanvasSize.height)
    ) {
      if (confirm("Disable change size piping?")) {
        this.canvas.wrapperEl.querySelector(
          ".pipingChangeSize_checkbox"
        ).checked = false;
        this.node.LS_Cls.LS_Painters.settings.pipingSettings.pipingChangeSize = false;
        this.node.LS_Cls.LS_Save();
      }
    }

    resizeCanvas(this.node, {
      width: new_width,
      height: new_height,
    });

    this.currentCanvasSize = { width: new_width, height: new_height };
    this.node.LS_Cls.LS_Painters.settings["currentCanvasSize"] =
      this.currentCanvasSize;
    this.node.title = `${this.node.type} - ${new_width}x${new_height}`;
    this.canvas.renderAll();
    app.graph.setDirtyCanvas(true, false);
    this.node.onResize();
    this.node.LS_Cls.LS_Save();
  }

  bindEvents() {
    // ----- Canvas Events -----
    this.canvas.on({
      // Mouse button down event
      "mouse:down": (o) => {

        // Right-click to delete the last arrow
        if (o.e.button === 2) {
            console.log(`TEETET ${this.type}`)
            if (this.arrows.length > 0) {
            let lastArrow = this.arrows.pop();
            this.canvas.remove(lastArrow);
            this.canvas.renderAll();
            }
            return;
        }
        
        // Left-click to start drawing an arrow
        if (this.type === "Arrow") {
            let pointer = this.canvas.getPointer(o.e);
            this.isDown = true;
            this.arrowStartX = pointer.x;
            this.arrowStartY = pointer.y;
            return;
        }

        
      },

      // Mouse move event
      "mouse:move": (o) => {
        if (this.isDown && this.type === "Arrow") {
            let pointer = this.canvas.getPointer(o.e);
        
            // Remove the previous temporary arrow
            if (this.arrow) {
              this.canvas.remove(this.arrow);
            }
        
            // Create a new arrow from the start point to the current pointer
            this.arrow = this.makeArrow(
              this.arrowStartX,
              this.arrowStartY,
              pointer.x,
              pointer.y
            );
        
            this.canvas.add(this.arrow);
            this.canvas.renderAll();
          }
        this.canvas.renderAll();
      },

      // Mouse button up event
      "mouse:up": (o) => {
        if (this.isDown && this.type === "Arrow") {
            this.isDown = false;
            this.arrows.push(this.arrow);
            this.arrow = null;
            this.uploadPaintFile(this.node.name);
          }
      },

      "object:added": (o) => {},

      // Object moving event
      "object:moving": (o) => {
        this.canvas.isDrawingMode = false;
      },

      // Object modify event
      "object:modified": () => {
        this.canvas.isDrawingMode = false;
        this.canvas.renderAll();
        this.uploadPaintFile(this.node.name);
      },
    });
    // ----- Canvas Events -----
  }

  // Save canvas data to localStorage or JSON
  canvasSaveSettingsPainter() {
    if (!this.node.LS_Cls.LS_Painters.settings.lsSavePainter) return;

    try {
      const data = this.canvas.toJSON(["mypaintlib"]);
      if (
        this.node.LS_Cls.LS_Painters &&
        !isEmptyObject(this.node.LS_Cls.LS_Painters)
      ) {
        this.node.LS_Cls.LS_Painters.canvas_settings = painters_settings_json
          ? data
          : JSON.stringify(data);

        this.node.LS_Cls.LS_Painters.settings["currentCanvasSize"] =
          this.currentCanvasSize;

        this.node.LS_Cls.LS_Save();
      }
    } catch (e) {
      console.error(e);
    }
  }

  setCanvasLoadData(data) {
    const obj_data =
      typeof data === "string" || data instanceof String
        ? JSON.parse(data)
        : data;

    const canvas_settings = data.canvas_settings;
    const settings = data.settings;

    this.canvas.loadFromJSON(canvas_settings, () => {
      this.canvas.renderAll();
      this.uploadPaintFile(this.node.name);
    //   this.bgColor.value = getColorHEX(data.background).color || "";
    });
  }

  // Load canvas data from localStorage or JSON
  canvasLoadSettingPainter() {
    try {
      if (
        this.node.LS_Cls.LS_Painters &&
        this.node.LS_Cls.LS_Painters.hasOwnProperty("canvas_settings")
      ) {
        const data =
          typeof this.node.LS_Cls.LS_Painters === "string" ||
          this.node.LS_Cls.LS_Painters instanceof String
            ? JSON.parse(this.node.LS_Cls.LS_Painters)
            : this.node.LS_Cls.LS_Painters;
        this.setCanvasLoadData(data);
      }
    } catch (e) {
      console.error(e);
    }
  }

  showImage(name) {
    let img = new Image();
    img.onload = () => {
      this.node.imgs = [img];
      app.graph.setDirtyCanvas(true);
      this.node.onResize();
    };

    let folder_separator = name.lastIndexOf("/");
    let subfolder = "";
    if (folder_separator > -1) {
      subfolder = name.substring(0, folder_separator);
      name = name.substring(folder_separator + 1);
    }

    img.src = api.apiURL(
      `/view?filename=${name}&type=input&subfolder=${subfolder}${app.getPreviewFormatParam()}&${new Date().getTime()}`
    );
    this.node.setSizeForImage?.();
  }

  async uploadPaintFile(fileName) {
    // Upload paint to temp folder ComfyUI
    let activeObj = null;
    if (!this.canvas.isDrawingMode) {
      activeObj = this.canvas.getActiveObject();

      if (activeObj) {
        activeObj.hasControls = false;
        activeObj.hasBorders = false;
        this.canvas.getActiveObjects().forEach((a_obs) => {
          a_obs.hasControls = false;
          a_obs.hasBorders = false;
        });
        this.canvas.renderAll();
      }
    }

    await new Promise((res) => {
      const uploadFile = async (blobFile) => {
        try {
          const resp = await fetch("/upload/image", {
            method: "POST",
            body: blobFile,
          });

          if (resp.status === 200) {
            const data = await resp.json();

            if (!this.image.options.values.includes(data.name)) {
              this.image.options.values.push(data.name);
            }

            this.image.value = data.name;
            this.showImage(data.name);

            if (activeObj && !this.drawning) {
              activeObj.hasControls = true;
              activeObj.hasBorders = true;

              this.canvas.getActiveObjects().forEach((a_obs) => {
                a_obs.hasControls = true;
                a_obs.hasBorders = true;
              });
              this.canvas.renderAll();
            }
            this.canvasSaveSettingsPainter();
            res(true);
          } else {
            alert(resp.status + " - " + resp.statusText);
          }
        } catch (error) {
          console.log(error);
        }
      };

      this.canvas.lowerCanvasEl.toBlob(function (blob) {
        let formData = new FormData();
        formData.append("image", blob, fileName);
        formData.append("overwrite", "true");
        //formData.append("type", "temp");
        uploadFile(formData);
      }, "image/png");
    });

    // - end

    const callb = this.node.callback,
      self = this;
    this.image.callback = function () {
      self.image.value = self.node.name;
      if (callb) {
        return callb.apply(this, arguments);
      }
    };
  }
  
}
// ================= END CLASS PAINTER ================

// ================= CREATE PAINTER WIDGET ============
function PainterWidget(node, inputName, inputData, app) {
  node.name = inputName;
  const widget = {
    type: "painter_widget",
    name: `w${inputName}`,
    callback: () => {},
    draw: function (ctx, _, widgetWidth, y, widgetHeight) {
      const margin = 10,
        left_offset = 1,
        top_offset = 1,
        visible = app.canvas.ds.scale > 0.0 && this.type === "painter_widget",
        w = widgetWidth - margin * 4,
        clientRectBound = ctx.canvas.getBoundingClientRect(),
        transform = new DOMMatrix()
          .scaleSelf(
            clientRectBound.width / ctx.canvas.width,
            clientRectBound.height / ctx.canvas.height
          )
          .multiplySelf(ctx.getTransform())
          .translateSelf(margin, margin + y),
        scale = new DOMMatrix().scaleSelf(transform.a, transform.d);

      let aspect_ratio = 1;
      if (node?.imgs && typeof node.imgs !== undefined) {
        aspect_ratio = node.imgs[0].naturalHeight / node.imgs[0].naturalWidth;
      }
      Object.assign(this.painter_wrap.style, {
        left: `${
          transform.a * margin * left_offset +
          transform.e +
          clientRectBound.left
        }px`,
        top: `${
          transform.d + transform.f + top_offset + clientRectBound.top
        }px`,
        width: `${w * transform.a}px`,
        height: `${w * transform.d * aspect_ratio}px`,
        position: "absolute",
        zIndex: app.graph._nodes.indexOf(node),
      });

      Object.assign(this.painter_wrap.children[0].style, {
        transformOrigin: "0 0",
        transform: scale,
        width: w + "px",
        height: w * aspect_ratio + "px",
      });

      Object.assign(this.painter_wrap.children[1].style, {
        transformOrigin: "0 0",
        transform: scale,
        width: w + "px",
        height: w * aspect_ratio + "px",
      });
      this.painter_wrap.hidden = !visible;
    },
  };

  // Fabric canvas
  let canvasPainter = document.createElement("canvas");
  node.painter = new Painter(node, canvasPainter);

  node.painter.canvas.setWidth(node.painter.currentCanvasSize.width);
  node.painter.canvas.setHeight(node.painter.currentCanvasSize.height);

  resizeCanvas(node, node.painter.canvas);

  widget.painter_wrap = node.painter.canvas.wrapperEl;
  widget.parent = node;
  
  node.painter.image.value = node.name;

  node.painter.propertiesLS();
  node.painter.bindEvents();

  document.body.appendChild(widget.painter_wrap);

  node.addWidget("button", "Clear Canvas", "clear_painer", () => {
    // node.painter.list_objects_panel__items.innerHTML = "";
    node.painter.clearCanvas();
  });

  // Add customWidget to node
  node.addCustomWidget(widget);

  node.onRemoved = () => {
    this.LS_Cls.removeData();
    // When removing this node we need to remove the input from the DOM
    for (let y in node.widgets) {
      if (node.widgets[y].painter_wrap) {
        node.widgets[y].painter_wrap.remove();
      }
    }
  };

  widget.onRemove = () => {
    widget.painter_wrap?.remove();
  };

  node.onResize = function () {
    let [w, h] = this.size;
    let aspect_ratio = 1;

    if (node?.imgs && typeof this.imgs !== undefined) {
      aspect_ratio = this.imgs[0].naturalHeight / this.imgs[0].naturalWidth;
    }
    let buffer = 120;

    if (w > this.painter.maxNodeSize) w = w - (w - this.painter.maxNodeSize);
    if (w < 600) w = 600;

    h = w * aspect_ratio + buffer;
    // console.log(`ON RESIZE ${w} ${h} ${node.painter.type}`)
    this.size = [w, h];
  };

  node.onDrawBackground = function (ctx) {
    if (!this.flags.collapsed) {
      node.painter.canvas.wrapperEl.hidden = false;

    } else {
      node.painter.canvas.wrapperEl.hidden = true;
    }
  };

  node.onConnectInput = () => console.log(`Connected input ${node.name}`);

  // Get piping image input, when node executing...
  api.addEventListener("mi2v_get_image", async ({ detail }) => {
    const { images, unique_id } = detail;

    if (
      !images.length ||
      !node.LS_Cls.LS_Painters.settings.pipingSettings.pipingUpdateImage ||
      +unique_id !== node.id
    ) {
      return;
    }

    await new Promise((res) => {
      const img = new Image();
      img.onload = () => {
        // Change size piping input image
        const { naturalWidth: w, naturalHeight: h } = img;
        if (
          node.LS_Cls.LS_Painters.settings.pipingSettings.pipingChangeSize &&
          (w !== node.painter.currentCanvasSize.width ||
            h !== node.painter.currentCanvasSize.height)
        ) {
          node.painter.setCanvasSize(w, h);
        } else {
          node.title = `${node.type} - ${node.painter.currentCanvasSize.width}x${node.painter.currentCanvasSize.height}`;
        }

        const img_ = new fabric.Image(img, {
          left: 0,
          top: 0,
          angle: 0,
          strokeWidth: 1,
          originX: "left",
          originY: "top",
          pipingImage: true,
        });
        res(img_);
      };
      img.src = images[0];
    })
      .then(async (result) => {
        await new Promise((res) => {
            node.painter.canvas.setBackgroundImage(
              result,
              async () => {
                node.painter.canvas.renderAll();
                await node.painter.uploadPaintFile(node.name);
                res(true);
              },
              {
                scaleX: node.painter.canvas.width / result.width,
                scaleY: node.painter.canvas.height / result.height,
                strokeWidth: 0,
              }
            );
          });
      })
      .then(() => {
        api
          .fetchApi("/mi2v/check_canvas_changed", {
            method: "POST",
            body: JSON.stringify({
              unique_id: node.id.toString(),
              is_ok: true,
            }),
          })
          .then((res) => res.json())
          .then((res) =>
            res?.status === "Ok"
              ? console.log(
                  `%cChange canvas ${node.name}: ${res.status}`,
                  "color: green; font-weight: 600;"
                )
              : console.error(`Error change canvas: ${res.status}`)
          )
          .catch((err) => console.error(`Error change canvas: ${err}`));
      });
  });

  app.canvas.onDrawBackground = function () {
    // Draw node isnt fired once the node is off the screen
    // if it goes off screen quickly, the input may not be removed
    // this shifts it off screen so it can be moved back if the node is visible.
    for (let n in app.graph._nodes) {
      const currnode = app.graph._nodes[n];
      for (let w in currnode.widgets) {
        let wid = currnode.widgets[w];
        if (Object.hasOwn(wid, "painter_widget")) {
          wid.painter_wrap.style.left = -8000 + "px";
          wid.painter_wrap.style.position = "absolute";
        }
      }
    }
  };

  node.painter.type = "Arrow"

  app.graph.setDirtyCanvas(true, false);
  node.onResize();

  node.painter.isDrawingMode = false;
  node.painter.drawning = true;
  
  console.log(`TYPE ${node.painter.type}`)

  return { widget: widget };
}
// ================= END CREATE PAINTER WIDGET ============

// ================= CREATE EXTENSION ================

const extensionName = "mi2v.MotionPainter";

app.registerExtension({
  name: extensionName,
  async init(app) {
  },
  async setup(app) {
    let PainerNode = app.graph._nodes.filter((wi) => wi.type == "MotionPainter");

    if (PainerNode.length) {
      PainerNode.map(async (n) => {
        console.log(`Setup MotionPainter: ${n.name}`);
        const widgetImage = n.widgets.find((w) => w.name == "image");
        await n.LS_Cls.LS_Init(n);
        let painter_ls = n.LS_Cls.LS_Painters;

        if (painter_ls && typeof lsData === "string") {
          painter_ls = JSON.parse(painter_ls);
        }

        if (widgetImage && painter_ls && !isEmptyObject(painter_ls)) {

          painter_ls.hasOwnProperty("objects_canvas") &&
            delete painter_ls.objects_canvas; // remove old property

          if (painter_ls?.settings?.currentCanvasSize) {
            n.painter.currentCanvasSize = painter_ls.settings.currentCanvasSize;

            n.painter.setCanvasSize(
              n.painter.currentCanvasSize.width,
              n.painter.currentCanvasSize.height
            );
          }
          n.painter.canvasLoadSettingPainter();
          console.log(`TTTT ${n.painter.type}`)
          // Resize window
          window.addEventListener("resize", (e) => resizeCanvas(n), false);
        }
      });
    }
  },
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "MotionPainter") {
      // Create node
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        const r = onNodeCreated
          ? onNodeCreated.apply(this, arguments)
          : undefined;

        const node_title = await this.getTitle();
        const node_id = this.id; // used node id as image name,instead of MotionPainter's quantity

        const nodeName = `Paint_${node_id}`;
        const nodeNamePNG = `${nodeName}.png`;

        console.log(`Create MotionPainter: ${nodeName}`);

        this.LS_Cls = new LS_Class(nodeNamePNG, painters_settings_json);

        // Find widget update_node and hide him
        for (const w of this.widgets) {
          if (w.name === "update_node") {
            w.type = "converted-widget";
            w.value =
              this.LS_Cls.LS_Painters.settings?.pipingSettings
                ?.pipingUpdateImage ?? true;
            w.computeSize = () => [0, -4];
            if (!w.linkedWidgets) continue;
            for (const l of w.linkedWidgets) {
              l.type = "converted-widget";
              l.computeSize = () => [0, -4];
            }
          }
        }

        PainterWidget.apply(this, [this, nodeNamePNG, {}, app]);
        this.painter.canvas.renderAll();
        this.painter.uploadPaintFile(nodeNamePNG);
        this.title = `${this.type} - ${this.painter.currentCanvasSize.width}x${this.painter.currentCanvasSize.height}`;

        return r;
      };
    }
  },
});
// ================= END CREATE EXTENSION ================
