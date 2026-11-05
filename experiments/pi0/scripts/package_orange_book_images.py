#!/usr/bin/python3
"""Package a visually reviewed selection of real orange-test camera frames."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import simpleSplit
from docx import Document
from docx.shared import Cm, Pt


# Ten reviewed moments, with both camera views retained for each moment.
SELECTION = [
    (2, 'existing', 1, 'workspace_two_oranges', 'Workspace with two oranges, distractor cubes and the bowl'),
    (19, 'existing', 2, 'two_orange_approach', 'Approach view with both oranges visible'),
    (16, 'extra', 145., 'first_orange_pickup', 'First orange held by the gripper; second orange remains on the table'),
    (16, 'extra', 160., 'first_orange_transport', 'First-orange transport toward the bowl'),
    (16, 'existing', 5, 'first_orange_in_bowl', 'First orange visible inside the bowl in the wrist view'),
    (16, 'existing', 6, 'second_orange_approach', 'Gripper approaching the second orange'),
    (18, 'extra', 603., 'second_orange_pickup', 'Second orange held above the table'),
    (18, 'extra', 620., 'second_orange_over_bowl', 'Second orange above the bowl, with the first orange inside'),
    (14, 'existing', 9, 'two_oranges_in_bowl', 'Two oranges visible inside the bowl in the wrist view'),
    (18, 'existing', 10, 'completed_bowl_view', 'Completed bowl view with two oranges'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.session.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    images = out/'original_images'
    images.mkdir(exist_ok=True)
    records = []
    for pair, (attempt, kind, target, tag, caption) in enumerate(SELECTION, 1):
        trial = root/f'cross_01/attempt_{attempt:02d}'
        if kind == 'existing':
            candidates = json.loads((trial/'report_frames/source_manifest.json').read_text())['panels']
        else:
            candidates = json.loads((root/f'reports/orange_image_review/extra_a{attempt:02d}_index.json').read_text())
        for camera in ['base', 'wrist']:
            if kind == 'existing':
                item = next(x for x in candidates if x['camera'] == camera and x['step'] == target)
                source = trial/'report_frames'/item['file']
            else:
                item = min((x for x in candidates if x['camera'] == camera), key=lambda x: abs(x['capture_time_s']-target))
                source = root/'reports/orange_image_review'/item['file']
            number = len(records)+1
            filename = f'{number:02d}_{tag}_{camera}_A{attempt:02d}_t{item["capture_time_s"]:09.3f}.png'
            destination = images/filename
            shutil.copy2(source, destination)
            assert sha(source) == sha(destination)
            with Image.open(destination) as im:
                width, height = im.size
                assert (width, height) == (640, 480)
            records.append({'image': number, 'pair': pair, 'caption': caption, 'attempt': attempt,
                            'operator_outcome': json.loads((trial/'summary.json').read_text())['operator_outcome'],
                            'camera': camera, 'capture_time_s': item['capture_time_s'],
                            'source_stamp_s': item['source_stamp_s'], 'source_age_s': item.get('source_age_s'),
                            'video_frame_index_zero_based': item['video_frame_index_zero_based'],
                            'original_video': str(trial/'video'/f'{camera}.avi'),
                            'selected_source': str(source), 'file': f'original_images/{filename}',
                            'width': width, 'height': height, 'sha256': sha(destination)})
    assert len(records) == 20
    assert len({r['sha256'] for r in records}) == 20
    assert sum(r['camera'] == 'base' for r in records) == 10
    with (out/'Image_Index.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {'title': 'Sort the oranges — selected report images',
                'reviewed_attempts': list(range(1, 21)), 'reviewed_candidate_images': 400,
                'additional_video_frame_candidates': 20,
                'selection': 'Visually selected for clarity, visible grasp/transport/placement and two-orange scenes; not a representative outcome sample.',
                'image_count': 20, 'base_images': 10, 'wrist_images': 10,
                'image_treatment': 'Original full-frame 640 x 480 PNG images. No crop, enhancement, upscaling, added overlays or generated imagery.',
                'captions': 'Describe visible scenes. Paired camera frames use nearby receive times, not asserted hardware synchronization.',
                'phase_order': 'Presentation order by task phase; source attempt numbers and timestamps are retained.',
                'records': records}
    (out/'Image_Sources.json').write_text(json.dumps(metadata, indent=2)+'\n')
    pdfmetrics.registerFont(TTFont('ReviewFont', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('ReviewBold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    pdf = out/'Sort_The_Oranges_20_Selected_Images.pdf'
    c = canvas.Canvas(str(pdf), pagesize=A4)
    c.setTitle('Sort the oranges — 20 selected report images')
    w, h = A4
    for page in range(5):
        c.setFillColorRGB(.09, .20, .29)
        c.setFont('ReviewBold', 18)
        c.drawString(34, h-40, 'Sort the oranges — selected images')
        c.setFont('ReviewFont', 9)
        c.drawString(34, h-60, '20 original frames | 10 base + 10 wrist | selected across the 20-attempt session')
        for slot in range(2):
            pair = page*2+slot+1
            row = [r for r in records if r['pair'] == pair]
            y = h-98-slot*338
            c.setFont('ReviewBold', 10)
            for j, line in enumerate(simpleSplit(f'{pair:02d}. {row[0]["caption"]}', 'ReviewBold', 10, w-68)):
                c.drawString(34, y-j*13, line)
            y -= 27
            image_width = (w-80)/2
            image_height = image_width*.75
            for col, record in enumerate(row):
                x = 34+col*(image_width+12)
                c.drawImage(str(out/record['file']), x, y-image_height, width=image_width, height=image_height)
                c.setFont('ReviewBold', 9)
                c.drawString(x, y-image_height-15, f'Image {record["image"]:02d} | {record["camera"].capitalize()} camera')
                c.setFont('ReviewFont', 8)
                c.drawString(x, y-image_height-29, f'Attempt {record["attempt"]:02d} | capture t = {record["capture_time_s"]:.3f} s')
        c.setFont('ReviewFont', 8)
        c.drawString(34, 31, 'Illustrative selection; source attempts/timestamps are listed in Image_Index.csv.')
        c.drawRightString(w-34, 31, f'{page+1} / 5')
        c.showPage()
    c.save()
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(1.3)
    section.left_margin = section.right_margin = Cm(1.2)
    document.styles['Normal'].font.name = 'Calibri'
    document.styles['Normal'].font.size = Pt(9)
    for page in range(5):
        if page:
            document.add_page_break()
        document.add_heading('Sort the oranges — selected images', 1)
        document.add_paragraph('Original camera frames selected across all 20 attempts. Image files are also supplied separately.')
        for slot in range(2):
            pair = page*2+slot+1
            row = [r for r in records if r['pair'] == pair]
            document.add_heading(f'{pair:02d}. {row[0]["caption"]}', 2)
            table = document.add_table(rows=1, cols=2)
            for cell, record in zip(table.rows[0].cells, row):
                cell.paragraphs[0].add_run().add_picture(str(out/record['file']), width=Cm(8.7))
                cell.add_paragraph(f'Image {record["image"]:02d} | {record["camera"].capitalize()} camera\nAttempt {record["attempt"]:02d} | t = {record["capture_time_s"]:.3f} s')
    document.save(out/'Sort_The_Oranges_20_Selected_Images.docx')
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                z.write(path, Path(out.name)/path.relative_to(out))
    downloads = Path.home()/'Downloads'
    for path in [archive, pdf, out/'Sort_The_Oranges_20_Selected_Images.docx']:
        shutil.copy2(path, downloads/path.name)
    print(json.dumps({'images': 20, 'base': 10, 'wrist': 10, 'pdf': str(downloads/pdf.name),
                      'word': str(downloads/'Sort_The_Oranges_20_Selected_Images.docx'),
                      'zip': str(downloads/archive.name), 'zip_bytes': archive.stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
