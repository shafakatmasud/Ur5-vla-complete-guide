#!/usr/bin/python3
"""Export real recorded frames for review; no ROS imports or robot commands."""
import argparse
import json
from pathlib import Path
import time
import traceback

import cv2
from cross_grid_results import write_json, utc


def export_trial(folder, count=10):
    folder=Path(folder).resolve()
    if count < 2:
        raise ValueError('At least two frames per camera are required')
    summary=json.loads((folder/'summary.json').read_text())
    session_path=folder.parents[1]/'session.json'
    session=json.loads(session_path.read_text()) if session_path.exists() else {}
    objects_requested=summary.get('objects_requested',session.get('objects_per_attempt',1))
    frames={'base':[], 'wrist':[]}; commands=[]; events=[]
    with (folder/'observations.jsonl').open() as stream:
        for line in stream:
            record=json.loads(line); topic=record['topic']
            if topic in ('camera/base/video_frame','camera/wrist/video_frame'):
                frames[topic.split('/')[1]].append(record)
            elif topic=='/control_abstraction_layer':
                commands.append(record['t_s'])
            elif topic=='/rosout':
                message=record['value'].get('message','')
                if 'BOWL RELEASE:' in message or 'Gripper changed to' in message or 'ASSISTED TRANSPORT COMPLETE:' in message:
                    events.append({'t_s':record['t_s'], 'message':message})
    if any(len(rows)<count for rows in frames.values()):
        raise ValueError('Insufficient recorded frames from both cameras')
    start=min(commands) if commands else max(rows[0]['t_s'] for rows in frames.values())
    end=min(max(commands)+4 if commands else float('inf'),
            *(rows[-1]['t_s'] for rows in frames.values()))
    closes=[x['t_s'] for x in events if 'Gripper changed to CLOSED' in x['message'] and start <= x['t_s'] <= end]
    releases=[x['t_s'] for x in events if ('BOWL RELEASE:' in x['message'] or 'ASSISTED TRANSPORT COMPLETE:' in x['message']) and start <= x['t_s'] <= end]
    if objects_requested>1:
        # Include both successful stages and the later unsuccessful cycle.
        # The old single-cycle selector could stop at the first release.
        anchors=sorted(set([start,end]+[max(start,min(end,t+delta)) for t in closes+releases for delta in (-1.,2.)]))
        if len(anchors)>count:
            anchors=[anchors[round(i*(len(anchors)-1)/(count-1))] for i in range(count)]
        while len(anchors)<count:
            left,right=max(zip(anchors,anchors[1:]),key=lambda pair:pair[1]-pair[0])
            if right-left<1e-6:break
            anchors=sorted(anchors+[(left+right)/2])
        targets=anchors
        method='Whole multi-object command interval, before/after close/release anchors, and largest-gap samples; includes later failed cycles. Events do not prove grasp or placement.'
    elif count==10 and closes and releases and closes[0]<releases[-1]:
        close,release=closes[0],releases[-1]
        targets=[start+(close-start)*f for f in (0,.5,1)]
        targets += [close+(release-close)*f for f in (.08,.3,.6,.85,.98)]
        targets += [release,min(end,release+2)]
        method='Command interval plus gripper-close/release log anchors; events are not proof of object contact or placement.'
    else:
        targets=[start+(end-start)*k/(count-1) for k in range(count)]
        method='Uniform receive-time samples over the observed policy-command interval, with up to four seconds after commands stop.'
    destination=folder/'report_frames';destination.mkdir(exist_ok=True)
    output=[]
    for camera,records in frames.items():
        cap=cv2.VideoCapture(str(folder/'video'/f'{camera}.avi'))
        try:
            used=set()
            for i,target in enumerate(targets,1):
                candidates=sorted(records,key=lambda x:abs(x['t_s']-target))
                record=next((x for x in candidates if x['value']['index'] not in used),None)
                if record is None:raise ValueError('Could not select distinct frames')
                index=record['value']['index'];used.add(index)
                cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read()
                if not ok:raise ValueError(f'Cannot decode {camera} frame {index}')
                filename=f'{camera}_{i:02d}_t{record["t_s"]:09.3f}.png'
                if not cv2.imwrite(str(destination/filename),image):raise OSError('PNG save failed')
                output.append({'camera':camera,'step':i,'file':filename,'capture_time_s':record['t_s'],
                               'source_stamp_s':record['value'].get('source_stamp_s'),
                               'source_age_s':record['value'].get('source_age_s'),
                               'video_frame_index_zero_based':index,'width':image.shape[1],'height':image.shape[0]})
        finally:cap.release()
    manifest={'exported_utc':utc(),'evidence':'REAL CAMERA FRAMES; not generated imagery',
              'visual_review_required':True,'source_trial':str(folder),'operator_outcome':summary.get('operator_outcome'),
              'count_per_camera':count,'selection_method':method,
              'objects_requested':objects_requested,
              'warning':'Candidate report frames. Verify pickup, transport and placement visually before assigning stage captions. Cameras are not asserted hardware synchronized. No crop, enhancement or scene modification.',
              'events':events,'panels':output}
    write_json(destination/'source_manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial',nargs='?',type=Path)
    parser.add_argument('--watch-session',type=Path)
    parser.add_argument('--count',type=int,default=10)
    args=parser.parse_args()
    if args.count<2:parser.error('--count must be at least 2')
    if bool(args.trial)==bool(args.watch_session):parser.error('Specify a trial folder or --watch-session')
    if args.trial:
        result=export_trial(args.trial,args.count);print('Exported',len(result['panels']),'real frames');return
    root=args.watch_session.resolve();failed=set()
    while True:
        for summary in sorted(root.glob('cross_*/attempt_*/summary.json')):
            folder=summary.parent;manifest=folder/'report_frames/source_manifest.json'
            # Active renumbering can leave a temporary recorder alias or move
            # a folder after this scan. Export only the canonical directory.
            if folder.is_symlink() or not summary.exists():continue
            if manifest.exists() or folder in failed:continue
            try:
                result=export_trial(folder,args.count)
                print(f'EXPORTED {folder.name}: {len(result["panels"])} camera frames',flush=True)
            except Exception as error:
                if not folder.exists():continue
                failed.add(folder)
                try:
                    write_json(folder/'report_frame_export_error.json',{'utc':utc(),'error':str(error)})
                except FileNotFoundError:
                    failed.discard(folder)
                    continue
                traceback.print_exc()
        write_json(root/'report_export_status.json',{'updated_utc':utc(),'mode':'watching completed trials',
            'frames_per_camera':args.count,'failed_trials':[str(x) for x in failed],'robot_commands_sent':0})
        time.sleep(2)


if __name__=='__main__':main()
