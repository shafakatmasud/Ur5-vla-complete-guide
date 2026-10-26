"""Create dimensioned vector checkerboards for actual-size printing."""
import argparse
import json
from pathlib import Path

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas


def create_target(path, page_size, paper):
    width, height = page_size
    c = Canvas(str(path), pagesize=page_size)
    c.setTitle(f'UR5 calibration checkerboard - {paper} - 20 mm squares')
    c.setAuthor('UR5 experiment calibration preparation')
    c.setFont('Helvetica-Bold', 15)
    c.drawCentredString(width / 2, height - 20 * mm, 'UR5 camera calibration checkerboard')
    c.setFont('Helvetica', 10)
    c.drawCentredString(width / 2, height - 28 * mm, f'{paper} | 9 x 7 squares | square side: 20 mm | inner corners: 8 x 6')
    c.drawCentredString(width / 2, height - 36 * mm, 'PRINT AT ACTUAL SIZE / 100%. Disable Fit, Shrink and borderless enlargement.')

    left = (width - 180 * mm) / 2
    bottom = height - 65 * mm - 140 * mm
    c.setFont('Helvetica-Bold', 9)
    c.drawString(left, bottom + 140 * mm + 8 * mm, 'TOP LEFT - keep this orientation identifiable during capture')
    # Exactly 9 columns by 7 rows. No strokes: square edges do not grow.
    for row in range(7):
        for column in range(9):
            if (row + column) % 2 == 0:
                c.setFillColorRGB(0, 0, 0)
                c.rect(left + column * 20 * mm, bottom + (6 - row) * 20 * mm,
                       20 * mm, 20 * mm, fill=1, stroke=0)

    y = bottom - 17 * mm
    c.setLineWidth(.5)
    bx = (width - 100 * mm) / 2
    c.line(bx, y, bx + 100 * mm, y)
    for x in (bx, bx + 100 * mm):
        c.line(x, y - 2 * mm, x, y + 2 * mm)
    c.setFont('Helvetica', 10)
    c.drawCentredString(width / 2, y - 6 * mm, 'Measure between tick centres: must be 100 mm')
    c.setFont('Helvetica', 9)
    lines = [
        '1. Print ONE page on the matching paper size; use black ink on white matte paper.',
        '2. Check the 100 mm bar and a 20 mm square with a ruler before mounting.',
        '3. Mount flat on rigid board. Keep a clean white border; avoid glossy lamination.',
        '4. Keep the whole pattern visible and sharply focused during calibration capture.',
        'This target alone does not calibrate camera-to-robot or gripper coordinates.',
        'Do not attach it to the robot or move the arm until the capture setup is defined.',
    ]
    for index, line in enumerate(lines):
        c.drawString(left, y - (15 + index * 4.5) * mm, line)
    c.showPage()
    c.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for paper, page_size in [('A4', A4), ('Letter', letter)]:
        path = args.output / f'UR5_checkerboard_20mm_{paper}.pdf'
        create_target(path, page_size, paper)
        print(path)
    (args.output / 'checkerboard_spec.json').write_text(json.dumps({
        'pattern': 'checkerboard', 'square_side_m': .02,
        'squares_columns_rows': [9, 7], 'inner_corners_columns_rows': [8, 6],
        'board_width_height_m': [.18, .14], 'print_scale_percent': 100,
        'scale_bar_m': .1, 'orientation': 'printed TOP LEFT label above first row',
        'purpose': 'physical calibration reference; no calibration performed',
        'reference': 'https://docs.opencv.org/4.12.0/da/d0d/tutorial_camera_calibration_pattern.html',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
