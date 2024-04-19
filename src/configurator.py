from dataclasses import dataclass
import json
from tkinter import Tk, Label, Button, Entry, StringVar, IntVar, Checkbutton, Listbox, END, MULTIPLE

@dataclass
class VideoProcessingConfig:
    pixelate_state: bool = True  # Whether to apply pixelation to people
    blackout_labels: list = None  # Labels to blackout (must be in coco-classes.json)
    yolo_model_path: str = "models/yolov9c-seg.pt"  # YOLO model path with segmentation
    source_index: int|str = 0  # Index of the video source (0 for default camera)
    img_size: int = 980  # Input image size for YOLO model
    model_confidence: float = 0.20  # Confidence threshold for YOLO model
    pixelate_level: int = 60  # Pixelation level for faces

    def __post_init__(self) -> None:
        if self.blackout_labels is None:
            self.blackout_labels = ["tv", "laptop", "cell phone"]

    def __str__(self) -> str:
        return f"Processing Configuration:\n" \
                f"Pixelate Faces: {self.pixelate_state}\n" \
                f"Blackout Labels: {', '.join(self.blackout_labels)}\n" \
                f"YOLO Model Path: {self.yolo_model_path}\n" \
                f"Source Index: {self.source_index}\n" \
                f"Image Size: {self.img_size}\n" \
                f"Model Confidence: {self.model_confidence}\n"

def configuratorGUI() -> VideoProcessingConfig:
    # Load coco classes from JSON
    with open("./src/misc/coco-classes.json", "r") as file:
        coco_classes = json.load(file)

    # Create a new Tkinter window
    root = Tk()
    root.title("Video Processing Configuration")

    # Variable bindings
    pixelate_state_var = IntVar(value=1)
    source_index_var = StringVar(value="0")
    if int(source_index_var.get()):
        source_index_var = int(source_index_var.get())
    img_size_var = IntVar(value=640)
    model_confidence_var = StringVar(value="0.20")

    # Listbox for blackout labels
    blackout_label_listbox = Listbox(root, selectmode=MULTIPLE, exportselection=0)
    for item in coco_classes:
        blackout_label_listbox.insert(END, coco_classes[item])
    blackout_label_listbox.grid(row=1, column=1, sticky='w', rowspan=4)

    # Helper functions
<<<<<<< HEAD
    def save_config() -> VideoProcessingConfig:
=======
    def save_config():
>>>>>>> 32a323bef03b55026b7238756632548bba10c2e6
        selected_labels = [blackout_label_listbox.get(i) for i in blackout_label_listbox.curselection()]
        global config
        config = VideoProcessingConfig(
            pixelate_state=bool(pixelate_state_var.get()),
            blackout_labels=selected_labels,
            source_index=source_index_var.get(),
            img_size=img_size_var.get(),
            model_confidence=float(model_confidence_var.get())
        )
        root.destroy()  # Close the window after saving
        return config

    # UI setup
    Label(root, text="Pixelate Faces:").grid(row=0, column=0, sticky='w')
    Checkbutton(root, text="Enable", variable=pixelate_state_var).grid(row=0, column=1, sticky='w')

    Label(root, text="Blackout Labels:").grid(row=1, column=0, sticky='nw')

    Label(root, text="Source Index:").grid(row=5, column=0, sticky='w')
    Entry(root, textvariable=source_index_var).grid(row=5, column=1, sticky='w')

    Label(root, text="Image Size:").grid(row=6, column=0, sticky='w')
    Entry(root, textvariable=img_size_var).grid(row=6, column=1, sticky='w')

    Label(root, text="Model Confidence:").grid(row=7, column=0, sticky='w')
    Entry(root, textvariable=model_confidence_var).grid(row=7, column=1, sticky='w')

    Button(root, text="Save Configuration", command=save_config).grid(row=8, columnspan=2)

    # Start the Tkinter loop
    root.mainloop()
    return config

if __name__ == "__main__":
    config = configuratorGUI()
    print(config)
