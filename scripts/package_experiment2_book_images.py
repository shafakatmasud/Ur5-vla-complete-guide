#!/usr/bin/python3
"""Package visually reviewed experiment-2 frames without altering source photographs."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Inches, Pt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'data/dataset/staging/experiment_2_sort_fruits'
EPISODE = 'episode_20261021_225333'
STAGES = [
    (0, 'Initial workspace', 'Two apples and two distractor cubes before sorting.'),
    (29.8, 'Approach first apple', 'The open gripper aligns with the first apple.'),
    (42, 'Lift first apple', 'The first apple is held above the table.'),
    (68, 'Carry first apple', 'The first apple is transported toward the bowl.'),
    (82.8, 'Place first apple', 'The first apple is visible inside the bowl after release.'),
    (133.4, 'Approach second apple', 'The open gripper approaches the remaining apple.'),
    (143, 'Lift second apple', 'The second apple is held above the table.'),
    (171, 'Carry second apple', 'The second apple is transported toward the bowl.'),
    (185.8, 'Second apple above bowl', 'The second apple is positioned over the bowl before release.'),
    (189.3, 'Completed sorting', 'Both apples are visible inside the bowl; the cubes remain on the table.'),
]


def main():
    folder = EXPERIMENT / 'episodes' / EPISODE
    output = EXPERIMENT / 'reports/Experiment2_Book_Images'
    output.mkdir(exist_ok=False)
    (output / 'original_images').mkdir()
    (output / 'figures').mkdir()
    frames = [json.loads(line) for line in (folder / 'frames.jsonl').open()]
    rows = []
    selected = []
    for i, (target, title, caption) in enumerate(STAGES, 1):
        frame = min(frames, key=lambda f: abs(f['episode_time_s'] - target))
        selected.append((title, caption, frame))
        for camera in ['base', 'wrist']:
            src = folder / frame[f'observation.images.{camera}']
            dst = output / 'original_images' / f'{i:02}_{title.lower().replace(" ", "_")}_{camera}.jpg'
            shutil.copyfile(src, dst)
            assert src.read_bytes() == dst.read_bytes()
            with Image.open(dst) as im:
                width, height = im.size
            rows.append({'stage': i, 'title': title, 'caption': caption, 'camera': camera,
                         'episode': EPISODE, 'frame_index': frame['frame_index'],
                         'original_episode_time_s': frame['episode_time_s'],
                         'original_record_time_utc': frame['record_time_utc'],
                         'camera_source_stamp_ns': frame.get(f'{camera}_image_source_stamp_ns'),
                         'width_px': width, 'height_px': height,
                         'image_file': str(dst.relative_to(output)), 'source_file': str(src),
                         'sha256': hashlib.sha256(dst.read_bytes()).hexdigest()})
    with (output / 'image_sources.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = {'task': 'Sort the fruits', 'recording_condition': 'Manual joystick demonstrations',
                'episode': EPISODE, 'camera_images': 20, 'base_images': 10, 'wrist_images': 10,
                'photographs': 'Byte-identical copies of recorded RGB images; no content edits, crops or artificial enhancement.',
                'time_basis': 'Original recorded episode timing, not the Excel-only 2.1 time scale.',
                'camera_pairing': 'Both views belong to the same dataset record; camera source exposure timestamps may differ.',
                'selection': 'Visually reviewed approach, held-object transport, and bowl-placement frames. IO alone is not used as proof of contact.',
                'images': rows}
    (output / 'image_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    doc = Document()
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Inches(.55)
    section.left_margin = section.right_margin = Inches(.6)
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    doc.styles['Normal'].font.name = 'Calibri'; doc.styles['Normal'].font.size = Pt(10)
    doc.styles['Normal'].paragraph_format.space_after = Pt(4)
    pdf = canvas.Canvas(str(output / 'Experiment2_Picking_and_Placing.pdf'), pagesize=A4)
    pdf.setTitle('Experiment 2 — Picking and placing both apples')
    pdf.setAuthor('UR5 experiment recordings')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    for page in range(5):
        if page:
            doc.add_page_break()
        doc.add_heading('Experiment 2: Sort the fruits', 0)
        doc.add_paragraph('Manual joystick demonstration | Base camera (left), wrist camera (right).')
        pdf.setFont('Helvetica-Bold', 17); pdf.drawString(35, 807, 'Experiment 2: Sort the fruits')
        pdf.setFont('Helvetica', 9); pdf.drawString(35, 790, 'Manual joystick demonstration | Base camera (left), wrist camera (right)')
        panel = Image.new('RGB', (1320, 1120), 'white'); draw = ImageDraw.Draw(panel)
        for k in range(2):
            index = page*2+k
            title, caption, frame = selected[index]
            files = [output / r['image_file'] for r in rows if r['stage'] == index+1]
            doc.add_heading(f'{index+1:02}. {title}', 2)
            table = doc.add_table(rows=1, cols=2)
            for cell, f in zip(table.rows[0].cells, files):
                cell.paragraphs[0].add_run().add_picture(str(f), width=Inches(3.35))
            doc.add_paragraph(caption)
            doc.add_paragraph(f"Source: {EPISODE}, frame {frame['frame_index']}; original t = {frame['episode_time_s']:.2f} s.")
            y = 762-k*345
            pdf.setFont('Helvetica-Bold', 12); pdf.drawString(35, y, f'{index+1:02}. {title}')
            for j, f in enumerate(files):
                pdf.drawImage(str(f), 35+j*267, y-211, width=258, height=193.5, preserveAspectRatio=True)
                im = Image.open(f)
                assert im.size == (640, 480), im.size
                panel.paste(im, (10+j*660, k*560+58))
                draw.text((10+j*660, k*560+32), 'Base camera' if j == 0 else 'Wrist camera', fill='#334155', font=small)
            pdf.setFont('Helvetica', 9); pdf.drawString(35, y-229, caption)
            pdf.setFont('Helvetica', 8); pdf.drawString(35, y-243, f"{EPISODE} | frame {frame['frame_index']} | original t = {frame['episode_time_s']:.2f} s")
            draw.text((10,k*560+3), f'{index+1:02}. {title}', fill='#17365D', font=font)
            draw.text((10,k*560+540), f"Frame {frame['frame_index']} | original t = {frame['episode_time_s']:.2f} s", fill='#334155', font=small)
        doc.add_paragraph('Original RGB images: 640 × 480 pixels. Camera exposures may differ within each paired dataset record. Source images and provenance are included in the ZIP.')
        pdf.setFont('Helvetica', 8); pdf.drawString(35, 45, 'Original 640 x 480 RGB frames; camera exposures may differ. Provenance is provided in image_sources.csv.')
        pdf.drawRightString(560, 30, f'{page+1} / 5'); pdf.showPage()
        panel.save(output / 'figures' / f'Figure_{page+1:02}_paired_sequence.png', dpi=(200,200))
    pdf.save()
    doc.save(output / 'Experiment2_Picking_and_Placing.docx')
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(output.rglob('*')):
            if f.is_file(): z.write(f, f.relative_to(output.parent))
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    print(json.dumps({'output': str(output), 'archive': str(archive), 'images': len(rows), 'bytes': archive.stat().st_size}))


if __name__ == '__main__':
    main()
