#!/usr/bin/python3
"""Package unchanged trial PNGs and clearly labelled camera contact sheets."""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image as PILImage
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();root=args.session.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    rows=[];manifests={}
    for a in range(1,21):
        folder=root/f'cross_01/attempt_{a:02d}';trial=json.loads((folder/'trial.json').read_text());manifest=json.loads((folder/'report_frames/source_manifest.json').read_text())
        label='Complete success' if trial['operator_outcome']=='pass' else 'Partial success' if trial.get('pickup') is True else 'Failure'
        manifests[a]=manifest
        for panel in manifest['panels']:
            source=folder/'report_frames'/panel['file'];assert source.exists()
            relative=f'attempt_{a:02d}/{panel["camera"]}/{panel["file"]}'
            rows.append({'attempt':a,'outcome':label,'object_colour':'orange' if a==20 else 'blue','camera':panel['camera'],'step':panel['step'],'capture_time_s':panel['capture_time_s'],'source_stamp_s':panel['source_stamp_s'],'archive_path':relative,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_path':str(source)})
    partial=sorted({r['attempt'] for r in rows if r['outcome']=='Partial success'})
    assert partial==[3,4,5,6,9] and len(rows)==400
    headers=list(rows[0])
    def csv_text(subset):
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=headers);writer.writeheader();writer.writerows(subset);return stream.getvalue()
    (out/'image_index.csv').write_text(csv_text(rows))
    instructions=('Original full-frame PNGs from the recorded camera feeds; no generated imagery or scene edits.\n'
        'Each selected attempt has 10 base-camera and 10 wrist-camera images.\n'
        'Partial-success attempts are 3,4,5,6,9. These are full-task failures with some reported progress.\n'
        'Attempts 1-19 used blue; attempt 20 used orange, confirmed by the operator.\n'
        'Frame selection follows command/gripper-event anchors and requires visual review for stage captions.\n'
        'IO does not establish cube contact. Capture-relative timestamps are not verified task completion times.\n'
        'Cameras are not asserted hardware synchronized. PDF pages are downsampled previews; ZIP PNGs are originals.\n'
        'Two software-aborted attempt-19 captures and unrecorded demonstrations are excluded.\n')
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='ImageCaption',fontSize=7,leading=9))
    def pdf(path,subset):
        story=[];buffers=[];attempts=sorted({r['attempt'] for r in subset})
        for a in attempts:
            for camera in ['base','wrist']:
                if story:story.append(PageBreak())
                group=sorted([r for r in subset if r['attempt']==a and r['camera']==camera],key=lambda r:r['step'])
                assert len(group)==10
                story.append(Paragraph(f'Attempt {a:02d} | {camera.capitalize()} camera',styles['Heading1']))
                story.append(Paragraph(escape(f'{group[0]["outcome"]} | {group[0]["object_colour"]} cube | 10 real full-frame images'),styles['BodyText']))
                story.append(Paragraph('Preview images; original PNGs are in the ZIP. Times below are relative to recording start.',styles['ImageCaption']))
                cells=[]
                for r in group:
                    with PILImage.open(r['source_path']) as raw:
                        image=raw.convert('RGB');image.thumbnail((480,360))
                        stream=io.BytesIO();image.save(stream,format='JPEG',quality=88);stream.seek(0);buffers.append(stream)
                    cells.append([Image(stream,width=55*mm,height=41.25*mm),Paragraph(escape(f"{r['step']:02d} | t={r['capture_time_s']:.3f} s"),styles['ImageCaption'])])
                table_rows=[]
                for start in range(0,len(cells),3):
                    block=cells[start:start+3];block += [['','']]*(3-len(block))
                    table_rows += [[v[0] for v in block],[v[1] for v in block]]
                table=Table(table_rows,colWidths=[59*mm]*3);table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.append(table)
        def footer(canvas,doc):
            canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#526373'));canvas.drawString(16*mm,10*mm,'Real camera frames | Candidate stages require visual review');canvas.drawRightString(194*mm,10*mm,str(doc.page))
        SimpleDocTemplate(str(path),pagesize=A4,leftMargin=16*mm,rightMargin=16*mm,topMargin=14*mm,bottomMargin=16*mm,title=path.stem).build(story,onFirstPage=footer,onLaterPages=footer)
    products=[]
    for prefix,subset in [('UR5_All_20_Attempts',rows),('UR5_Partial_Success_Attempts',[r for r in rows if r['attempt'] in partial])]:
        archive=out/(prefix+'_Original_Images.zip')
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
            for r in subset:z.write(r['source_path'],r['archive_path'])
            z.writestr('image_index.csv',csv_text(subset));z.writestr('IMAGE_INFORMATION.txt',instructions)
            for a in sorted({r['attempt'] for r in subset}):z.writestr(f'attempt_{a:02d}/source_manifest.json',json.dumps(manifests[a],indent=2)+'\n')
        path=out/(prefix+'_Camera_Images.pdf');pdf(path,subset)
        with zipfile.ZipFile(archive) as z:
            names=[name for name in z.namelist() if name.endswith('.png')];assert len(names)==len(subset)
            for r in subset:assert hashlib.sha256(z.read(r['archive_path'])).hexdigest()==r['sha256']
        products.append({'zip':str(archive),'pdf':str(path),'original_images':len(subset),'attempts':sorted({r['attempt'] for r in subset}),'original_hashes_verified':True})
    (out/'package_manifest.json').write_text(json.dumps({'products':products,'scene_colour_confirmed':True,'original_images_modified':False,'pdf_previews_downsampled':True},indent=2)+'\n')
    print(json.dumps(products,indent=2))


if __name__=='__main__':main()
