import os
from PyPDF2 import PdfReader, PdfWriter

def crop_pdf_top(input_path, output_path, crop_points=30):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
        
    reader = PdfReader(input_path)
    writer = PdfWriter()
    
    for page in reader.pages:
        mb = page.mediabox
        page.mediabox.upper_right = (mb.upper_right[0], mb.upper_right[1] - crop_points)
        if hasattr(page, 'cropbox'):
            page.cropbox.upper_right = (mb.upper_right[0], mb.upper_right[1] - crop_points)
            
        writer.add_page(page)
        
    with open(output_path, "wb") as f:
        writer.write(f)
    print(f"Successfully cropped {input_path}")

files_to_crop = [
    ("IEEE-tex/Visualization_output", "IEEE_Fig5_Pareto_2D.pdf"),
    ("IEEE-tex/Visualization_output", "IEEE_Fig8_ExtremeWeather.pdf"),
    ("IEEE-tex/Visualization_output", "IEEE_Fig9_Ablation.pdf"),
    ("zh-tex/Visualization", "IEEE_Fig5_Pareto_2D.pdf"),
    ("zh-tex/Visualization", "IEEE_Fig8_ExtremeWeather.pdf"),
    ("zh-tex/Visualization", "IEEE_Fig9_Ablation.pdf")
]

for d, f in files_to_crop:
    filepath = os.path.join(d, f)
    crop_pdf_top(filepath, filepath, crop_points=35)
