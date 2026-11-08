"""One-clip tracking experiment. No form scoring or medical interpretation."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import subprocess

NAMES = {11:'left_shoulder',12:'right_shoulder',13:'left_elbow',14:'right_elbow',
         15:'left_wrist',16:'right_wrist',23:'left_hip',24:'right_hip',
         25:'left_knee',26:'right_knee',27:'left_ankle',28:'right_ankle',
         29:'left_heel',30:'right_heel',31:'left_toe',32:'right_toe'}
LINKS = [(12,14),(14,16),(12,24),(24,26),(26,28),(28,30),(30,32),(28,32)]
BAR_MATCH_THRESHOLD = 0.6

class BarEndMatcher:
    """Clip-specific appearance baseline, manually selected references; not a trained detector."""
    def __init__(self, video):
        self.templates=[]
        cap=cv2.VideoCapture(video,cv2.CAP_FFMPEG)
        # Two hand-selected appearances: racked and extended. Coordinates at width=540.
        for frame_id,x,y in [(300,87,397),(436,192,351)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id); ok,im=cap.read()
            if not ok: raise RuntimeError('Cannot read bar reference frame')
            gray=cv2.cvtColor(cv2.resize(im,(540,960)),cv2.COLOR_BGR2GRAY)
            for size in [24,32]:
                patch=gray[y-size//2:y+size//2,x-size//2:x+size//2]
                for scale in [.85,1.,1.15]:
                    self.templates.append(cv2.resize(patch,None,fx=scale,fy=scale))
        cap.release()
    def locate(self,frame,wrist):
        if not wrist or wrist['visibility']<.5 or wrist['presence']<.5: return None,None
        wx,wy=wrist['x'],wrist['y']; gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        x0=max(0,int(wx-100));x1=min(540,int(wx+40))
        y0=max(0,int(wy-80));y1=min(960,int(wy+65)); roi=gray[y0:y1,x0:x1]
        best=None
        for t in self.templates:
            th,tw=t.shape
            if roi.shape[0]<th or roi.shape[1]<tw: continue
            ncc=cv2.matchTemplate(roi,t,cv2.TM_CCOEFF_NORMED)
            yy,xx=np.indices(ncc.shape);cx=xx+x0+tw/2;cy=yy+y0+th/2
            score=ncc-.0015*np.hypot(cx-(wx-20),cy-(wy-25))
            _,s,_,loc=cv2.minMaxLoc(score);x,y=loc
            candidate=(float(s),float(ncc[y,x]),float(x+x0+tw/2),float(y+y0+th/2))
            if best is None or candidate[0]>best[0]: best=candidate
        if best is None: return None,None
        _,quality,x,y=best
        return ((x,y) if quality>=BAR_MATCH_THRESHOLD else None),quality

def label(im, s, p, color=(220,225,230), scale=.48):
    cv2.putText(im,s,p,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)

def run():
    project=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('video',nargs='?',default=str(project/'form_videos/BarbellBenchPress.mp4'))
    ap.add_argument('--model',default=str(project/'models/pose_landmarker_heavy.task'))
    ap.add_argument('--out',default=str(project/'results/latest'))
    ap.add_argument('--start-frame',type=int,default=300)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(a.video,cv2.CAP_FFMPEG)
    if not cap.isOpened(): raise RuntimeError('Cannot open video')
    fps=cap.get(cv2.CAP_PROP_FPS); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sw=int(cap.get(3)); sh=int(cap.get(4)); w=540; h=round(sh*w/sw)
    cap.set(cv2.CAP_PROP_POS_FRAMES,a.start_frame)
    matcher=BarEndMatcher(a.video); history=[]; records=[]; visible=Counter()
    canvas_w=w+380
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo',
         '-vcodec','rawvideo','-pix_fmt','bgr24','-s',f'{canvas_w}x{h}',
         '-r',str(fps),'-i','-','-an','-vcodec','libx264','-preset','fast',
         '-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'bench_tracking.mp4')]
    encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    opts=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=a.model),
          running_mode=vision.RunningMode.VIDEO,num_poses=1,
          min_pose_detection_confidence=.3,min_tracking_confidence=.5)
    t0=time.perf_counter(); misses=0; frame_id=a.start_frame
    try:
      with vision.PoseLandmarker.create_from_options(opts) as model:
        while True:
          ok,frame=cap.read()
          if not ok: break
          frame=cv2.resize(frame,(w,h)); raw=frame.copy()
          result=model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                 data=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)),round(frame_id*1000/fps))
          landmarks={}
          if result.pose_landmarks:
            ps=result.pose_landmarks[0]
            for idx,name in NAMES.items():
              p=ps[idx]; in_frame=0<=p.x<=1 and 0<=p.y<=1
              accepted=in_frame and p.visibility>=.5 and p.presence>=.5
              landmarks[name]={'x':float(p.x*w),'y':float(p.y*h),
                'visibility':float(p.visibility),'presence':float(p.presence),'displayed':accepted and idx%2==0}
              if accepted: visible[name]+=1
            for s,e in LINKS:
              pa=landmarks[NAMES[s]]; pb=landmarks[NAMES[e]]
              if pa['displayed'] and pb['displayed']:
                cv2.line(frame,(round(pa['x']),round(pa['y'])),(round(pb['x']),round(pb['y'])),(215,210,30),2,cv2.LINE_AA)
            offsets={12:(-78,-15),14:(14,0),16:(15,12),24:(-15,25),26:(-60,-10),28:(-70,-3),30:(-60,18),32:(-20,22)}
            for idx in range(12,33,2):
              if idx not in NAMES: continue
              p=landmarks[NAMES[idx]]
              if p['displayed']:
                pt=(round(p['x']),round(p['y'])); cv2.circle(frame,pt,4,(215,230,40),-1,cv2.LINE_AA)
                dx,dy=offsets[idx]; label(frame,NAMES[idx].replace('right_',''),(pt[0]+dx,pt[1]+dy),(215,245,100),.4)
          else: misses+=1
          bar,bar_quality=matcher.locate(raw,landmarks.get('right_wrist'))
          history.append(bar)
          for k in range(1,len(history)):
            if history[k-1] is not None and history[k] is not None:
              cv2.line(frame,tuple(np.rint(history[k-1]).astype(int)),tuple(np.rint(history[k]).astype(int)),(20,165,255),2,cv2.LINE_AA)
          if bar:
            cv2.circle(frame,tuple(np.rint(bar).astype(int)),8,(20,165,255),2,cv2.LINE_AA)
            label(frame,'bar end',(round(bar[0])+12,round(bar[1])-12),(20,195,255))
          records.append({'frame':frame_id,'source_time_s':frame_id/fps,'landmarks':landmarks,
              'bar_match_score':bar_quality,
              'bar':None if bar is None else {'x':bar[0],'y':bar[1],'method':'template_match'}})
          panel=np.full((h,380,3),(27,23,19),np.uint8)
          label(panel,'BENCH / FIRST TRACKING PASS',(22,36),(235,240,245),.56)
          label(panel,f'Source time: {frame_id/fps:.2f}s',(22,72))
          label(panel,'Body: MediaPipe Pose Heavy',(22,108),(215,245,100))
          label(panel,'Visible-side estimates only',(22,132))
          label(panel,'Bar: reference-image matching',(22,172),(20,195,255))
          label(panel,'Point = visible bar-end center',(22,196))
          label(panel,'2D image coordinates; no depth',(22,234))
          label(panel,'BAR-END TRAIL / IMAGE SPACE',(22,292),(235,240,245))
          cv2.rectangle(panel,(30,315),(350,635),(75,70,65),1)
          # Fixed crop: preserve image geometry and draw x increasing right / y down.
          def project(p): return (round(30+(p[0]-50)*320/260),round(315+(p[1]-300)*320/260))
          for k in range(1,len(history)):
            if history[k-1] is not None and history[k] is not None:
              cv2.line(panel,project(history[k-1]),project(history[k]),(20,165,255),2,cv2.LINE_AA)
          if bar: cv2.circle(panel,project(bar),5,(20,210,255),-1,cv2.LINE_AA)
          label(panel,'Right = image right; down = image down',(22,661),scale=.43)
          label(panel,'No position smoothing applied',(22,700))
          label(panel,'Hidden / uncertain joints omitted',(22,724))
          label(panel,'Bar status: '+('matched' if bar else 'uncertain / omitted'),(22,764),(140,180,245))
          label(panel,'No form or injury scoring',(22,788),(140,180,245))
          label(panel,'Two manually selected bar references',(22,820),scale=.44)
          label(panel,'Missing matches leave gaps in trail',(22,844),scale=.44)
          label(panel,'Setup omitted: starts at frame 300',(22,h-30),scale=.44)
          canvas=np.hstack([frame,panel]); encoder.stdin.write(canvas.tobytes())
          if frame_id in [300,360,436,500,581,650,726]:
            cv2.imwrite(str(out/f'preview_{frame_id}.jpg'),canvas)
          if (frame_id-a.start_frame)%100==0: print(f'processed {frame_id}/{total}',flush=True)
          frame_id+=1
    finally:
      cap.release(); encoder.stdin.close(); encoder.wait()
    if encoder.returncode: raise RuntimeError('Video encoding failed')
    with (out/'tracking.jsonl').open('w') as f:
      for row in records: f.write(json.dumps(row,allow_nan=False)+'\n')
    summary={'source':a.video,'source_width':sw,'source_height':sh,'analysis_width':w,'analysis_height':h,
      'source_fps':fps,'source_frame_count':total,'start_frame':a.start_frame,'processed_frames':len(records),
      'bar_reference_frames':[300,436],'bar_match_threshold':BAR_MATCH_THRESHOLD,'pose_missing_frames':misses,
      'landmark_frames_above_display_threshold':dict(visible),'bar_accepted_frames':sum(x['bar'] is not None for x in records),
      'processing_seconds':time.perf_counter()-t0,
      'notes':['Pixel positions in resized image, not physical measurements.',
        'Visibility scores and tracker success are not ground-truth accuracy.',
        'Two manually selected bar templates plus wrist-guided search; specific to this clip.',
        f'Normalized correlation >= {BAR_MATCH_THRESHOLD} accepted; score is not probability of correctness.',
        'Near-side landmarks displayed; both sides preserved in JSONL.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': run()
