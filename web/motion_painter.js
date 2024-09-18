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
    this.node = node;
    this.canvas = this.initCanvas(canvas);
    this.arrows = [];
    this.isDrawing = false;
    this.startPoint = null;
    this.currentCanvasSize = { width: 512, height: 512 };
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


    this.canvas.backgroundColor = "#ffffff";

    fabric.util.addListener(
      this.canvas.upperCanvasEl,
      "contextmenu",
      function (e) {
        e.preventDefault();
      }
    );

    return this.canvas;
  }
}

function MotionPainterWidget(node, app) {
    const widget = {
        type: "motion_painter_widget",
        name: `motion_painter_widget`,
        callback: () => {},
        draw: function (ctx, _, widgetWidth, y, widgetHeight) 
        {
            const margin = 10,
            left_offset = 8,
            top_offset = 50,
            visible = app.canvas.ds.scale > 0.6 && this.type === "motion_painter_widget",
            w = widgetWidth - margin * 2 - 80,
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

    document.body.appendChild(widget.painter_wrap);
    
    // Add customWidget to node
    node.addCustomWidget(widget);

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
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name === "MI2V Motion Painter") {
            console.log("Registering onNodeCreated for MI2V_MotionPainter");
            // Create node
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function () {
                const r = onNodeCreated
                ? onNodeCreated.apply(this, arguments)
                : undefined;
                
                console.log("onNodeCreated called for node:", this.type);
                // const node = this;

                // // Initialize imageName property
                // node.imageName = "default_image.png"; // Set a default image name or path

                // // Optionally, add a widget to select the image
                // node.addWidget("image", "image", null, (value) => {
                //   node.imageName = value;
                //   node.motionPainter.loadInputImage();
                // });

                MotionPainterWidget.apply(this, [this, appInstance]);
                this.painter.canvas.renderAll();
                this.title = "HELLOOOO";
            };
        }
    },
});
