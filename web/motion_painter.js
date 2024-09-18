import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { fabric } from "./lib/fabric.js";

console.log("HELLO WORLD");

function resizeCanvas(node, sizes) 
{
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

class MotionPainter {
  constructor(node, canvas) {
    this.originX = 0;
    this.originY = 0;
    this.drawning = true;
    this.mode = false;
    this.type = "Brush";

    this.locks = {
      lockMovementX: false,
      lockMovementY: false,
      lockScalingX: false,
      lockScalingY: false,
      lockRotation: false,
    };

    this.currentCanvasSize = { width: 512, height: 512 };
    this.maxNodeSize = 1024;

    this.bringFrontSelected = true;

    this.node = node;
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

    console.log("INIT");
  }

    initCanvas(canvasEl) {
        this.canvas = new fabric.Canvas(canvasEl, {
            isDrawingMode: true,
            backgroundColor: "transparent",
            width: 512,
            height: 512,
            enablePointerEvents: true,
        });

        // this.canvas.backgroundColor = "#ffffff";
        fabric.util.addListener(
            this.canvas.upperCanvasEl,
            "contextmenu",
            function (e) {
                e.preventDefault();
            }
        );

        return this.canvas;
    }

    setCanvasSize(new_width, new_height, confirmChange = false) {
        if (
            confirmChange &&
            this.node.isInputConnected(0) &&
            // this.node.LS_Cls.LS_Painters.settings.pipingSettings.pipingChangeSize &&
            (new_width !== this.currentCanvasSize.width ||
            new_height !== this.currentCanvasSize.height)
        ) {
            if (confirm("Disable change size piping?")) {
                this.canvas.wrapperEl.querySelector(
                    ".pipingChangeSize_checkbox"
                ).checked = false;
                // this.node.LS_Cls.LS_Painters.settings.pipingSettings.pipingChangeSize = false;
                // this.node.LS_Cls.LS_Save();
            }
        }

        resizeCanvas(this.node, {
            width: new_width,
            height: new_height,
        });

        this.currentCanvasSize = { width: new_width, height: new_height };
        // this.node.LS_Cls.LS_Painters.settings["currentCanvasSize"] = this.currentCanvasSize;
        this.node.title = `${this.node.type} - ${new_width}x${new_height}`;
        this.canvas.renderAll();
        app.graph.setDirtyCanvas(true, false);
        this.node.onResize();
        // this.node.LS_Cls.LS_Save();
    }

    showImage(name) {
        let img = new Image();
        img.onload = () => {
            this.node.imgs = [img];
            app.graph.setDirtyCanvas(true);
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

function MotionPainterWidget(node, inputName, app) {
    node.name = inputName;
    const widget = {
        type: "motion_painter_widget",
        name: `w${inputName}`,
        callback: () => {},
        draw: function (ctx, _, widgetWidth, y, widgetHeight) 
        {
            const margin = 10,
            left_offset = 1,
            top_offset = 1,
            visible = app.canvas.ds.scale > 0.6 && this.type === "motion_painter_widget",
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

            let aspect_ratio = 0.5;
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
                height: `${w * transform.d}px`,
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
        }
    }

    // Fabric canvas
    let canvasPainter = document.createElement("canvas");
    node.painter = new MotionPainter(node, canvasPainter)

    node.painter.canvas.setWidth(node.painter.currentCanvasSize.width);
    node.painter.canvas.setHeight(node.painter.currentCanvasSize.height);

    resizeCanvas(node, node.painter.canvas);

    widget.painter_wrap = node.painter.canvas.wrapperEl;
    widget.parent = node;
    console.log(`NODE name ${node.name}`)
    node.painter.image.value = node.name;
    console.log(`NODE name ${node.painter.image.value}`)
    document.body.appendChild(widget.painter_wrap);
    
    // Add customWidget to node
    node.addCustomWidget(widget);

    node.onRemoved = () => {
        // this.LS_Cls.removeData();
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

    node.onResize = function () 
    {
        let [w, h] = this.size;
        let aspect_ratio = 1;
    
        if (node?.imgs && typeof this.imgs !== undefined) {
          aspect_ratio = this.imgs[0].naturalHeight / this.imgs[0].naturalWidth;
        }
        let buffer = 90;
    
        if (w > this.painter.maxNodeSize) w = w - (w - this.painter.maxNodeSize);
        if (w < 600) w = 600;
    
        h = w * aspect_ratio + buffer;
    
        this.size = [w, h];
    };

    node.onDrawBackground = function (ctx) {
        if (!this.flags.collapsed) {
            node.painter.canvas.wrapperEl.hidden = false;
            if (this.imgs && this.imgs.length) {
                if (app.canvas.ds.scale > 0.8) {
                    let [dw, dh] = this.size;
            
                    let w = this.imgs[0].naturalWidth;
                    let h = this.imgs[0].naturalHeight;
            
                    const scaleX = dw / w;
                    const scaleY = dh / h;
                    const scale = Math.min(scaleX, scaleY, 1);
            
                    w *= scale / 8;
                    h *= scale / 8;
            
                    let x = 5;
                    let y = dh - h - 5;
            
                    ctx.drawImage(this.imgs[0], x, y, w, h);
                    ctx.font = "10px serif";
                    ctx.strokeStyle = "white";
                    ctx.strokeRect(x, y, w, h);
                    ctx.fillStyle = "rgba(255,255,255,0.7)";
                    ctx.fillText("Mask", w / 2, dh - 10);
                }
            }
        } else {
          node.painter.canvas.wrapperEl.hidden = true;
        }
    };

    node.onConnectInput = () => console.log(`Connected input ${node.name}`);

    // Get piping image input, when node executing...
    api.addEventListener("mi2v_get_image", async ({ detail }) => {
        const { images, unique_id } = detail;

        if (!images.length || +unique_id !== node.id) return;

        await new Promise((res) => {
            const img = new Image();
            img.onload = () => {
                // Change size piping input image
                const { naturalWidth: w, naturalHeight: h } = img;
                if (w !== node.painter.currentCanvasSize.width || h !== node.painter.currentCanvasSize.height){
                    node.painter.setCanvasSize(w, h);
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
        }).then(async (result) => {
            await new Promise((res) => {
                node.painter.canvas.setBackgroundImage(
                    result,
                    async () => {
                        node.painter.canvas.renderAll();
                        // await node.painter.uploadPaintFile(node.name);
                        res(true);
                    },
                    {
                        scaleX: node.painter.canvas.width / result.width,
                        scaleY: node.painter.canvas.height / result.height,
                        strokeWidth: 0,
                    }
                );
            });
        }).then(() => {
            api.fetchApi("/mi2v/check_canvas_changed", {
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

    app.canvas.onDrawBackground = function () 
    {
        // Draw node isnt fired once the node is off the screen
        // if it goes off screen quickly, the input may not be removed
        // this shifts it off screen so it can be moved back if the node is visible.
        for (let n in app.graph._nodes) {
            const currnode = app.graph._nodes[n];
            for (let w in currnode.widgets) {
            let wid = currnode.widgets[w];
            if (Object.hasOwn(wid, "motion_painter_widget")) {
                wid.painter_wrap.style.left = -8000 + "px";
                wid.painter_wrap.style.position = "absolute";
            }
            }
        }
    };

    app.graph.setDirtyCanvas(true, false);
    node.onResize();

    return { widget: widget };
}

app.registerExtension({
    name: "MI2V Motion Painter",
    async setup(app) {
        let PainerNode = app.graph._nodes.filter((wi) => wi.type == "MI2V Motion Painter");
    
        if (PainerNode.length) {
            PainerNode.map(async (n) => {
                console.log(`Setup PainterNode: ${n.name}`);
                const widgetImage = n.widgets.find((w) => w.name == "image");
                await n.LS_Cls.LS_Init(n);
                let painter_ls = n.LS_Cls.LS_Painters;
        
                if (painter_ls && typeof lsData === "string") {
                painter_ls = JSON.parse(painter_ls);
                }
        
                if (widgetImage && painter_ls && !isEmptyObject(painter_ls)) {
                // Load settings elements
                n.painter.setValueElementsLS();
        
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
        
                // Resize window
                window.addEventListener("resize", (e) => resizeCanvas(n), false);
                }
            });
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name === "MI2V Motion Painter") {
            console.log("Registering onNodeCreated for MI2V_MotionPainter");
            // Create node
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function () {
                const r = onNodeCreated
                ? onNodeCreated.apply(this, arguments)
                : undefined;
                
                const node_id = this.id;
                const nodeName = `MotionPaint_${node_id}`;
                console.log("onNodeCreated called for node:", this.type);
                // const node = this;

                // // Initialize imageName property
                // node.imageName = "default_image.png"; // Set a default image name or path

                // // Optionally, add a widget to select the image
                // node.addWidget("image", "image", null, (value) => {
                //   node.imageName = value;
                //   node.motionPainter.loadInputImage();
                // });

                MotionPainterWidget.apply(this, [this, nodeName, appInstance]);
                this.painter.canvas.renderAll();
                this.title = `${this.type} - ${this.painter.currentCanvasSize.width}x${this.painter.currentCanvasSize.height}`;
            };
        }
    },
});
