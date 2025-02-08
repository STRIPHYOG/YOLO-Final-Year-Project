import pandas as pd
import tkinter as tk
from tkinter import scrolledtext

# File paths (update according to your local directory)
file_paths = {
    "yolo_v11": r"D:\final yr project\compare\yolo_v11_results.csv",
    "yolo_v9": r"D:\final yr project\compare\yolo_v9_results.csv",
    "yolo_v8": r"D:\final yr project\compare\yolo_v8_results.csv",
    "yolo_v5": r"D:\final yr project\compare\yolo_v5_results.csv",
}

# Extract the highest mAP50-95(B) for each version
best_mAP = {}
detailed_info = {}

for version, path in file_paths.items():
    df = pd.read_csv(path)
    if "metrics/mAP50-95(B)" in df.columns:
        best_mAP[version] = df["metrics/mAP50-95(B)"].max()
        avg_mAP = df["metrics/mAP50-95(B)"].mean()
        worst_mAP = df["metrics/mAP50-95(B)"].min()
        detailed_info[version] = {
            "best": best_mAP[version],
            "average": avg_mAP,
            "worst": worst_mAP,
        }

# Find the best model
best_model = max(best_mAP, key=best_mAP.get)
best_mAP_value = best_mAP[best_model]

# Function to show results in a detailed popup window
def show_results():
    root = tk.Tk()
    root.title("YOLO Model Performance Comparison")

    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=200, height=200, font=("Arial", 12))
    text_area.pack(padx=10, pady=10)

    results_text = f"🚀 **Best Model: {best_model}**\n"
    results_text += f"🏆 **Best mAP50-95(B) Score:** {best_mAP_value:.5f}\n\n"
    
    # Reason why this model is best
    results_text += "**Why is this the best model?**\n"
    results_text += f"- Highest maximum mAP: {best_mAP_value:.5f}\n"
    results_text += f"- Consistently high performance: Avg mAP: {detailed_info[best_model]['average']:.5f}\n"
    results_text += f"- Lowest variation: Worst mAP: {detailed_info[best_model]['worst']:.5f}\n\n"

    # Comparison with others
    results_text += "**Comparison with other models:**\n"
    for model, scores in detailed_info.items():
        if model != best_model:
            results_text += f"\n📌 **{model}**\n"
            results_text += f"- Best mAP: {scores['best']:.5f}\n"
            results_text += f"- Average mAP: {scores['average']:.5f}\n"
            results_text += f"- Worst mAP: {scores['worst']:.5f}\n"
            if scores["best"] < best_mAP_value:
                results_text += "❌ Lower max accuracy than the best model.\n"
            if scores["average"] < detailed_info[best_model]["average"]:
                results_text += "❌ Inconsistent performance in some cases.\n"

    text_area.insert(tk.INSERT, results_text)
    text_area.config(state=tk.DISABLED)  # Make it read-only

    root.mainloop()

# Show the popup
show_results()
