from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load model
model = YOLO(r"../models/best.pt")

# Input image
image_path = "D:/project/test_samples/input_images/tile_94.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
orig_h, orig_w = image.shape[:2]

# Fixed class colors (RGB)
class_colors = {
    'building_rcc': (255, 255, 255),   # white
    'building_tiled': (255, 0, 0),     # red
    'building_tin': (0, 0, 255),       # blue
    'street_road': (150, 75, 0),       # brown
    'main_road': (128, 128, 128),      # grey
    'waterbody': (0, 255, 255)         # sky blue
}

# Run inference
results = model(image_path)

# Dictionary to store counts
class_counts = {cls: 0 for cls in class_colors.keys()}

if results[0].masks is not None:
    masks = results[0].masks.data.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    names = model.names

    for mask, cls_id in zip(masks, class_ids):
        class_name = names[cls_id]
        if class_name in class_colors:
            class_counts[class_name] += 1  # count each detection

            color = class_colors[class_name]
            resized_mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # Apply mask color
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            for i in range(3):
                colored_mask[:, :, i] = resized_mask * color[i]

            image = cv2.addWeighted(image, 1.0, colored_mask, 0.5, 0)

# --- Create visualization ---
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# Left: Segmentation result
ax[0].imshow(image)
ax[0].axis("off")
ax[0].set_title("Semantic Segmentation Result")

# Right: Color-coded legend with counts
legend_elements = []
for cls, color in class_colors.items():
    rgb_norm = tuple(c/255 for c in color)  # scale 0-1 for matplotlib
    count = class_counts[cls]

    # If color is very light (like white/yellow) → add black edge for visibility
    brightness = sum(color) / 3
    if brightness > 220:  # very light color threshold
        legend_elements.append(
            Patch(facecolor=rgb_norm, edgecolor="black", linewidth=1.5, label=f"{cls}: {count}")
        )
    else:
        legend_elements.append(Patch(facecolor=rgb_norm, label=f"{cls}: {count}"))

ax[1].legend(
    handles=legend_elements,
    loc="center",
    frameon=False,
    fontsize=12
)
ax[1].axis("off")
ax[1].set_title("Detected Object Counts", fontsize=14)

plt.tight_layout()
plt.show()

# Also print counts in terminal
print("\n🔢 Object Counts:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")
