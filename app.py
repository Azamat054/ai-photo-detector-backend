import customtkinter as ctk
from tkinter import filedialog
import sys

# подключаем бекенд
sys.path.append(r"C:\Users\azama\Desktop\project\ai_photo_detector")

from detector import predict_image, predict_video

ctk.set_appearance_mode("dark")

app = ctk.CTk()
app.geometry("500x350")
app.title("AI vs Human Detector")

label = ctk.CTkLabel(app, text="Выберите файл", font=("Arial", 18))
label.pack(pady=20)

result_box = ctk.CTkTextbox(app, width=420, height=120)
result_box.pack(pady=10)


def open_image():
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.png *.jpeg")]
    )
    if not path:
        return
    res = predict_image(path)
    show_result(res)


def open_video():
    path = filedialog.askopenfilename(
        filetypes=[("Video", "*.mp4 *.avi")]
    )
    if not path:
        return
    res = predict_video(path)
    show_result(res)


def show_result(res: dict):
    result_box.delete("1.0", "end")
    for k, v in res.items():
        if isinstance(v, float):
            v = round(v, 4)
        result_box.insert("end", f"{k}: {v}\n")


ctk.CTkButton(app, text="📷 Image", command=open_image).pack(pady=10)
ctk.CTkButton(app, text="🎥 Video", command=open_video).pack(pady=10)

app.mainloop()
